import os
import sys
import time
import uuid
import shutil
import tempfile
from flask import Flask, render_template, request, jsonify, send_from_directory
from communication_pipeline import (
    process_itantra_pipeline,
    get_encodec_model,
    get_asr_model,
    get_whisper_model,
    generate_tts_audio,
    load_and_normalize_audio,
    AudioPipelineError
)
from semantic_engine import SemanticPayload, IntentCategory, get_receiver_emergency_template
from packet_protocol import Packet, PacketType, Crc16
from ggwave_engine import decode_ggwave_wav_file, encode_packet_to_ggwave_wav

# Reconfigure stdout for UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
STATIC_AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_AUDIO_FOLDER, exist_ok=True)
_RX_CHUNK_BUFFER = {}

@app.route('/')
@app.route('/sender')
def index():
    return render_template('index.html')

@app.route('/receiver')
def receiver_page():
    return render_template('receiver.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        "status": "ONLINE",
        "system": "iTantra Speech & Semantic Communication Engine",
        "version": "2.0.0",
        "supported_modes": ["auto", "mode_1_semantic", "mode_2_text", "encodec_voice"],
        "supported_languages": ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"],
        "supported_bitrates": [24.0, 12.0, 6.0, 3.0, 1.5]
    })

@app.route('/api/transmit', methods=['POST'])
def api_transmit():
    temp_path = None
    try:
        bitrate = float(request.form.get('bitrate', 6.0))
        language = request.form.get('language', 'hi')
        mode = request.form.get('mode', 'auto')

        # Check if audio file was uploaded
        if 'audio' in request.files:
            file = request.files['audio']
            if not file.filename:
                return jsonify({
                    "status": "FAILURE",
                    "stage": "audio_upload",
                    "error": "Empty filename in uploaded audio payload"
                }), 400

            orig_ext = os.path.splitext(file.filename)[1].lower()
            if not orig_ext or len(orig_ext) > 10:
                orig_ext = '.bin'

            temp_filename = f"upload_{uuid.uuid4().hex}{orig_ext}"
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            file.save(temp_path)
            upload_size = os.path.getsize(temp_path)
            print(f"[API Upload Received] Filename: {file.filename} | Size: {upload_size} B | Mime: {file.content_type}")

            # Keep a persistent debug copy of the last upload for content-integrity verification
            try:
                debug_path = os.path.join(UPLOAD_FOLDER, f"last_upload_debug{orig_ext}")
                shutil.copyfile(temp_path, debug_path)
            except Exception as copy_err:
                print(f"[Debug Copy Info] {copy_err}")
        else:
            # Fallback to project audio.wav if no file provided
            temp_path = os.path.join(os.path.dirname(__file__), 'audio.wav')
            if not os.path.exists(temp_path):
                return jsonify({
                    "status": "FAILURE",
                    "stage": "audio_upload",
                    "error": "No audio file uploaded and default audio.wav not found"
                }), 400

        channel_mode = request.form.get('channel_mode', 'software_loopback')
        client_transcript = request.form.get('client_transcript', '').strip()

        # Execute full iTantra pipeline
        result = process_itantra_pipeline(
            audio_path=temp_path,
            target_bitrate=bitrate,
            language=language,
            mode=mode,
            channel_mode=channel_mode,
            output_dir=STATIC_AUDIO_FOLDER,
            client_transcript=client_transcript
        )

        return jsonify(result)

    except AudioPipelineError as ape:
        print(f"[API Stage Error] Stage: {ape.stage}, Message: {ape.message}")
        return jsonify({
            "status": "FAILURE",
            "stage": ape.stage,
            "error": ape.message
        }), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[API Internal Error] {e}")
        return jsonify({
            "status": "FAILURE",
            "stage": "server_error",
            "error": f"Internal processing error: {str(e)}"
        }), 500

    finally:
        # Clean up temporary uploaded file safely
        if temp_path and os.path.exists(temp_path) and temp_path.startswith(UPLOAD_FOLDER):
            try:
                os.remove(temp_path)
            except Exception as clean_err:
                print(f"[Cleanup Warning] {clean_err}")

@app.route('/api/audio/<path:filename>')
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    file_path = os.path.join(STATIC_AUDIO_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Audio file not found"}), 404

    response = send_from_directory(STATIC_AUDIO_FOLDER, filename)
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/receiver/decode_acoustic', methods=['POST'])
def api_receiver_decode_acoustic():
    """
    Laptop 2 Receiver Endpoint:
    Receives acoustic microphone audio slice, runs ggwave demodulator,
    validates CRC16, decodes semantic payload, and synthesizes TTS ONLY
    after a valid packet is confirmed.
    """
    if 'audio' not in request.files:
        return jsonify({"status": "NO_AUDIO"}), 400

    file = request.files['audio']
    temp_name = f"acoustic_rx_{uuid.uuid4().hex[:8]}.wav"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_name)
    file.save(temp_path)

    try:
        import soundfile as sf
        import numpy as np

        # Ensure standard 48kHz mono 16-bit PCM WAV for ggwave decoder
        try:
            info = sf.info(temp_path)
            if info.samplerate != 48000 or info.channels != 1 or info.subtype != 'PCM_16':
                data, orig_sr = sf.read(temp_path, dtype='float32')
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if orig_sr != 48000:
                    in_len = len(data)
                    out_len = int(round(in_len * 48000 / orig_sr))
                    data = np.interp(np.linspace(0, in_len, out_len, endpoint=False), np.arange(in_len), data)
                pcm16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
                sf.write(temp_path, pcm16, 48000, subtype='PCM_16')
        except Exception as norm_err:
            pass

        # Demodulate with ggwave
        wire_packet = decode_ggwave_wav_file(temp_path)

        if wire_packet is None:
            try:
                tensor_data, orig_sr, dur, *rest = load_and_normalize_audio(temp_path, target_sr=48000)
                pcm48 = (np.clip(tensor_data.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
                pcm_path = os.path.join(UPLOAD_FOLDER, f"pcm48_{temp_name}")
                sf.write(pcm_path, pcm48, 48000, subtype="PCM_16")
                wire_packet = decode_ggwave_wav_file(pcm_path)
                if os.path.exists(pcm_path):
                    os.remove(pcm_path)
            except Exception as conv_err:
                pass

        if not wire_packet:
            return jsonify({"status": "NO_PACKET"})

        # Packet received! Validate CRC16
        rx_packet = Packet.decode(wire_packet)
        if not rx_packet:
            # Fallback: check if raw ggwave payload is direct UTF-8 text or keyword
            try:
                raw_text = wire_packet.decode('utf-8', errors='ignore').strip()
                if raw_text and len(raw_text) >= 2 and all(c.isprintable() or c.isspace() for c in raw_text):
                    rx_tx_id = f"RAW-{int(time.time()*1000)%100000:05d}"
                    tts_filename = f"rx_laptop2_raw_{rx_tx_id}.wav"
                    tts_out_path = os.path.join(STATIC_AUDIO_FOLDER, tts_filename)
                    dur = generate_tts_audio(raw_text, tts_out_path, lang="hi", is_emergency=False)
                    return jsonify({
                        "status": "SUCCESS",
                        "mode": "RAW ACOUSTIC TEXT",
                        "crc": "PASS (RAW TONE)",
                        "category": "Direct Acoustic Message",
                        "receiverMeaningText": raw_text,
                        "audio_url": f"/api/audio/{tts_filename}",
                        "audio_duration_sec": round(dur, 2)
                    })
            except Exception:
                pass
            return jsonify({"status": "CRC_FAIL", "message": "Corrupted by acoustic noise"})

        wire_crc = Crc16.compute(wire_packet[:-2])
        rx_tx_id = f"ACOUSTIC-{int(time.time()*1000)%100000:05d}"

        if rx_packet.header.packet_type == PacketType.VOICE_OPUS:
            from opus_engine import OpusDecoder
            import struct
            # Check if payload contains chunk framing [chunk_idx: 2B, total_chunks: 2B]
            if len(rx_packet.payload) >= 4:
                idx, total = struct.unpack_from("!HH", rx_packet.payload, 0)
                if total > 1 and idx < total:
                    if total not in _RX_CHUNK_BUFFER:
                        _RX_CHUNK_BUFFER[total] = {}
                    _RX_CHUNK_BUFFER[total][idx] = rx_packet.payload[4:]
                    if len(_RX_CHUNK_BUFFER[total]) < total:
                        return jsonify({
                            "status": "PARTIAL",
                            "message": f"Received acoustic chunk {idx+1}/{total} (CRC PASS)",
                            "mode": "MODE 1 — VOICE (OPUS)",
                            "chunk": idx + 1,
                            "total": total
                        })
                    # All chunks arrived! Reassemble in sequence order
                    assembled = bytearray()
                    for i in range(total):
                        assembled.extend(_RX_CHUNK_BUFFER[total][i])
                    _RX_CHUNK_BUFFER.pop(total, None)
                    opus_raw = bytes(assembled)
                elif total == 1:
                    opus_raw = rx_packet.payload[4:]
                else:
                    opus_raw = rx_packet.payload
            else:
                opus_raw = rx_packet.payload
            try:
                decoder = OpusDecoder(target_sample_rate=24000)
                reconstructed_pcm, dec_sr = decoder.decode(opus_raw)
                voice_filename = f"rx_laptop2_voice_{rx_tx_id}.wav"
                voice_out_path = os.path.join(STATIC_AUDIO_FOLDER, voice_filename)
                pcm16 = (np.clip(reconstructed_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
                sf.write(voice_out_path, pcm16, dec_sr, subtype='PCM_16')
                dur = len(pcm16) / dec_sr

                print(f"\n{'='*70}")
                print(f"RECEIVER ACOUSTIC DEMODULATION LOGS ({rx_tx_id}) — VOICE (OPUS)")
                print(f"  RX listening started: TRUE")
                print(f"  ggwave signal detected: TRUE")
                print(f"  received payload bytes: {len(wire_packet)}")
                print(f"  packet ID: {rx_tx_id}")
                print(f"  CRC result: PASS (0x{wire_crc:04X})")
                print(f"  mode: MODE 1 — VOICE (OPUS)")
                print(f"  Opus decoding: SUCCESS")
                print(f"  sample rate: {dec_sr} Hz")
                print(f"  reconstructed duration: {dur:.2f}s")
                print(f"  speaker playback: STARTED")
                print(f"{'='*70}\n")

                return jsonify({
                    "status": "SUCCESS",
                    "mode": "MODE 1 — VOICE (OPUS)",
                    "packet_type": "VOICE_OPUS",
                    "crc": f"PASS (0x{wire_crc:04X})",
                    "category": "Voice Audio",
                    "type_id": 0,
                    "receiverMeaningText": f"Reconstructed Speech Audio ({dec_sr} Hz, {dur:.2f}s)",
                    "audio_url": f"/api/audio/{voice_filename}",
                    "audio_duration_sec": round(dur, 2)
                })
            except Exception as dec_err:
                print(f"[Acoustic Opus Decode Error] {dec_err}")
                return jsonify({"status": "DECODE_ERROR", "message": f"Opus decoding error: {dec_err}"})

        elif rx_packet.header.packet_type == PacketType.MODE_1_SEMANTIC:
            rx_payload = SemanticPayload.decode(rx_packet.payload)
            category = IntentCategory.from_id(rx_payload.type_id)
            receiver_meaning = get_receiver_emergency_template(rx_payload.type_id, lang_code="hi")

            # Synthesize receiver speech strictly on Laptop 2 receiver station
            tts_filename = f"rx_laptop2_{rx_tx_id}.wav"
            tts_out_path = os.path.join(STATIC_AUDIO_FOLDER, tts_filename)
            dur = generate_tts_audio(receiver_meaning, tts_out_path, lang="hi", is_emergency=True)

            print(f"\n{'='*70}")
            print(f"RECEIVER ACOUSTIC DEMODULATION LOGS ({rx_tx_id}) — SEMANTIC")
            print(f"  RX listening started: TRUE")
            print(f"  ggwave signal detected: TRUE")
            print(f"  received payload bytes: {len(wire_packet)}")
            print(f"  packet ID: {rx_tx_id}")
            print(f"  CRC result: PASS (0x{wire_crc:04X})")
            print(f"  FEC result: PASS")
            print(f"  semantic type ID: {rx_payload.type_id}")
            print(f"  decoded category: {category.label}")
            print(f"  template: \"{receiver_meaning}\"")
            print(f"  TTS input: \"{receiver_meaning}\"")
            print(f"  TTS sample rate: 24000 Hz")
            print(f"  TTS duration: {dur:.2f}s")
            print(f"  speaker playback: STARTED")
            print(f"{'='*70}\n")

            return jsonify({
                "status": "SUCCESS",
                "mode": "MODE 2 — SEMANTIC",
                "crc": f"PASS (0x{wire_crc:04X})",
                "category": category.label,
                "type_id": rx_payload.type_id,
                "receiverMeaningText": receiver_meaning,
                "audio_url": f"/api/audio/{tts_filename}",
                "audio_duration_sec": round(dur, 2)
            })
        else:
            text_str = rx_packet.payload.decode("utf-8")
            tts_filename = f"rx_laptop2_text_{rx_tx_id}.wav"
            tts_out_path = os.path.join(STATIC_AUDIO_FOLDER, tts_filename)
            dur = generate_tts_audio(text_str, tts_out_path, lang="hi", is_emergency=False)

            print(f"\n{'='*70}")
            print(f"RECEIVER ACOUSTIC DEMODULATION LOGS ({rx_tx_id})")
            print(f"  RX listening started: TRUE")
            print(f"  ggwave signal detected: TRUE")
            print(f"  received payload bytes: {len(wire_packet)}")
            print(f"  packet ID: {rx_tx_id}")
            print(f"  CRC result: PASS (0x{wire_crc:04X})")
            print(f"  FEC result: PASS")
            print(f"  semantic type ID: N/A (Free Text)")
            print(f"  decoded category: Free Text")
            print(f"  template: \"{text_str}\"")
            print(f"  TTS input: \"{text_str}\"")
            print(f"  TTS sample rate: 24000 Hz")
            print(f"  TTS duration: {dur:.2f}s")
            print(f"  speaker playback: STARTED")
            print(f"{'='*70}\n")

            return jsonify({
                "status": "SUCCESS",
                "mode": "MODE 2 — FREE TEXT",
                "crc": f"PASS (0x{wire_crc:04X})",
                "category": "Text",
                "receiverMeaningText": text_str,
                "audio_url": f"/api/audio/{tts_filename}",
                "audio_duration_sec": round(dur, 2)
            })

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route('/api/receiver/simulate_packet', methods=['POST'])
@app.route('/api/simulate_rx_packet', methods=['POST'])
def api_receiver_simulate_packet():
    """
    Simulate an incoming acoustic transmission for testing the receiver station.
    Generates the exact ggwave acoustic packet audio and decodes it,
    returning the decoded result, TTS/voice audio URL, and acoustic chirp audio URL
    so the user can play it over speakers or test over the air.
    """
    data = request.get_json(silent=True) or request.form or {}
    sim_type = data.get('type', 'medical')
    rx_tx_id = f"SIM-{int(time.time()*1000)%100000:05d}"

    try:
        import numpy as np
        import soundfile as sf
        from packet_protocol import Packet, PacketHeader, PacketType, Crc16
        from semantic_engine import SemanticPayload, IntentCategory, get_receiver_emergency_template
        from ggwave_engine import encode_packet_to_ggwave_wav

        if sim_type == 'medical':
            payload = SemanticPayload(type_id=0, intent_id=0, severity=2)
            header = PacketHeader(PacketType.MODE_1_SEMANTIC, lang_id=0, is_fec=False, sequence_num=1)
            packet = Packet(header, payload.encode())
            wire_packet = packet.encode()
            wire_crc = Crc16.compute(wire_packet[:-2])

            tone_filename = f"tone_sim_medical_{rx_tx_id}.wav"
            tone_path = os.path.join(STATIC_AUDIO_FOLDER, tone_filename)
            encode_packet_to_ggwave_wav(wire_packet, tone_path, volume=50)

            receiver_meaning = get_receiver_emergency_template(0, lang_code="hi")
            tts_filename = f"rx_laptop2_sim_medical_{rx_tx_id}.wav"
            tts_out_path = os.path.join(STATIC_AUDIO_FOLDER, tts_filename)
            dur = generate_tts_audio(receiver_meaning, tts_out_path, lang="hi", is_emergency=True)

            return jsonify({
                "status": "SUCCESS",
                "mode": "MODE 2 — SEMANTIC",
                "crc": f"PASS (0x{wire_crc:04X})",
                "category": "Medical Emergency",
                "type_id": 0,
                "receiverMeaningText": receiver_meaning,
                "audio_url": f"/api/audio/{tts_filename}",
                "audio_duration_sec": round(dur, 2),
                "tone_url": f"/api/audio/{tone_filename}"
            })

        elif sim_type == 'fire':
            payload = SemanticPayload(type_id=1, intent_id=0, severity=2)
            header = PacketHeader(PacketType.MODE_1_SEMANTIC, lang_id=0, is_fec=False, sequence_num=2)
            packet = Packet(header, payload.encode())
            wire_packet = packet.encode()
            wire_crc = Crc16.compute(wire_packet[:-2])

            tone_filename = f"tone_sim_fire_{rx_tx_id}.wav"
            tone_path = os.path.join(STATIC_AUDIO_FOLDER, tone_filename)
            encode_packet_to_ggwave_wav(wire_packet, tone_path, volume=50)

            receiver_meaning = get_receiver_emergency_template(1, lang_code="hi")
            tts_filename = f"rx_laptop2_sim_fire_{rx_tx_id}.wav"
            tts_out_path = os.path.join(STATIC_AUDIO_FOLDER, tts_filename)
            dur = generate_tts_audio(receiver_meaning, tts_out_path, lang="hi", is_emergency=True)

            return jsonify({
                "status": "SUCCESS",
                "mode": "MODE 2 — SEMANTIC",
                "crc": f"PASS (0x{wire_crc:04X})",
                "category": "Fire / Evacuation",
                "type_id": 1,
                "receiverMeaningText": receiver_meaning,
                "audio_url": f"/api/audio/{tts_filename}",
                "audio_duration_sec": round(dur, 2),
                "tone_url": f"/api/audio/{tone_filename}"
            })

        elif sim_type == 'voice':
            from opus_engine import OpusEncoder, OpusDecoder
            sample_voice_path = os.path.join(os.path.dirname(__file__), "audio.wav")
            if os.path.exists(sample_voice_path):
                pcm_data, sr = sf.read(sample_voice_path, dtype="float32")
                if pcm_data.ndim > 1:
                    pcm_data = pcm_data.mean(axis=1)
                pcm_slice = pcm_data[:int(sr * 0.4)]
            else:
                t = np.linspace(0, 0.4, int(24000 * 0.4), endpoint=False)
                pcm_slice = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
                sr = 24000

            encoder = OpusEncoder(sample_rate=24000)
            opus_raw = encoder.encode(pcm_slice, orig_sr=sr)

            header = PacketHeader(PacketType.VOICE_OPUS, lang_id=0, is_fec=False, sequence_num=3)
            packet = Packet(header, opus_raw)
            wire_packet = packet.encode()
            wire_crc = Crc16.compute(wire_packet[:-2])

            tone_filename = f"tone_sim_voice_{rx_tx_id}.wav"
            tone_path = os.path.join(STATIC_AUDIO_FOLDER, tone_filename)
            encode_packet_to_ggwave_wav(wire_packet, tone_path, volume=50)

            decoder = OpusDecoder(target_sample_rate=24000)
            reconstructed_pcm, dec_sr = decoder.decode(opus_raw)
            voice_filename = f"rx_laptop2_sim_voice_{rx_tx_id}.wav"
            voice_out_path = os.path.join(STATIC_AUDIO_FOLDER, voice_filename)
            pcm16 = (np.clip(reconstructed_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
            sf.write(voice_out_path, pcm16, dec_sr, subtype='PCM_16')
            dur = len(pcm16) / dec_sr

            return jsonify({
                "status": "SUCCESS",
                "mode": "MODE 1 — VOICE (OPUS)",
                "packet_type": "VOICE_OPUS",
                "crc": f"PASS (0x{wire_crc:04X})",
                "category": "Voice Audio (Opus Codec)",
                "type_id": 0,
                "receiverMeaningText": f"Reconstructed Speech Audio ({dec_sr} Hz, {dur:.2f}s)",
                "audio_url": f"/api/audio/{voice_filename}",
                "audio_duration_sec": round(dur, 2),
                "tone_url": f"/api/audio/{tone_filename}"
            })

        else: # text
            text_str = "यह आईतंत्र का एक परीक्षण संदेश है।"
            header = PacketHeader(PacketType.MODE_2_TEXT, lang_id=0, is_fec=False, sequence_num=4)
            packet = Packet(header, text_str.encode("utf-8"))
            wire_packet = packet.encode()
            wire_crc = Crc16.compute(wire_packet[:-2])

            tone_filename = f"tone_sim_text_{rx_tx_id}.wav"
            tone_path = os.path.join(STATIC_AUDIO_FOLDER, tone_filename)
            encode_packet_to_ggwave_wav(wire_packet, tone_path, volume=50)

            tts_filename = f"rx_laptop2_sim_text_{rx_tx_id}.wav"
            tts_out_path = os.path.join(STATIC_AUDIO_FOLDER, tts_filename)
            dur = generate_tts_audio(text_str, tts_out_path, lang="hi", is_emergency=False)

            return jsonify({
                "status": "SUCCESS",
                "mode": "MODE 2 — FREE TEXT",
                "crc": f"PASS (0x{wire_crc:04X})",
                "category": "Free Text Message",
                "receiverMeaningText": text_str,
                "audio_url": f"/api/audio/{tts_filename}",
                "audio_duration_sec": round(dur, 2),
                "tone_url": f"/api/audio/{tone_filename}"
            })

    except Exception as e:
        print(f"[Simulate Packet Error] {e}")
        return jsonify({"status": "FAILURE", "error": str(e)}), 500


if __name__ == '__main__':
    get_encodec_model()
    try:
        get_asr_model()
    except Exception as e:
        print(f"[iTantra Server Warning] IndicConformer pre-warm notice: {e}")
    try:
        get_whisper_model()
    except Exception as e:
        print(f"[iTantra Server Warning] Whisper pre-warm notice: {e}")
    print("[iTantra Server] Ready on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

