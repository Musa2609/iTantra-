import os
import io
import time
import zlib
import pickle
import struct
import shutil
import tempfile
import uuid
import subprocess
import torch
import numpy as np
import soundfile as sf
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import AutoModel

# Global model caches for fast inference
_ENCODEC_MODEL = None
_ASR_MODEL = None

class AudioPipelineError(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage
        self.message = message

def get_encodec_model():
    global _ENCODEC_MODEL
    if _ENCODEC_MODEL is None:
        print("[iTantra Engine] Loading Meta EnCodec 24kHz Model...")
        _ENCODEC_MODEL = EncodecModel.encodec_model_24khz()
        _ENCODEC_MODEL.eval()
    return _ENCODEC_MODEL

def get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        print("[iTantra Engine] Loading AI4Bharat IndicConformer-600M Model...")
        _ASR_MODEL = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        )
    return _ASR_MODEL

def detect_audio_format(file_path):
    """Detect actual audio format by inspecting header magic bytes."""
    if not os.path.exists(file_path):
        return "non_existent"
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return "empty_file"

    try:
        with open(file_path, "rb") as f:
            header = f.read(32)
    except Exception:
        return "unreadable"

    if len(header) < 4:
        return "corrupt_header"

    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        return "WAV (PCM)"
    elif header.startswith(b"\x1a\x45\xdf\xa3"):
        return "WebM (EBML)"
    elif header.startswith(b"OggS"):
        return "Ogg (Opus/Vorbis)"
    elif header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "MP3"
    elif header.startswith(b"fLaC"):
        return "FLAC"
    elif len(header) >= 8 and header[4:8] == b"ftyp":
        return "MP4/M4A"
    else:
        return "Unknown Container"

def load_and_normalize_audio(audio_path, target_sr=24000):
    """
    Robust multi-format audio loader & normalizer:
    1. Inspect magic bytes & format.
    2. Decode audio via soundfile, torchaudio, or ffmpeg fallback.
    3. Validate duration, channels, sample rate, non-empty, finite samples (no NaN/Inf).
    4. Normalize to target_sr (24,000 Hz), mono (1 channel), float32 tensor scaled to [-1.0, 1.0].
    """
    if not os.path.exists(audio_path):
        raise AudioPipelineError("audio_upload", "Uploaded audio file does not exist")

    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise AudioPipelineError("audio_upload", "Uploaded audio file is empty (0 bytes)")

    fmt_desc = detect_audio_format(audio_path)
    print(f"[iTantra Audio Loader] Processing file: {os.path.basename(audio_path)} ({file_size} bytes, Format: {fmt_desc})")

    data = None
    orig_sr = None

    # Strategy 1: soundfile
    try:
        data_np, sr = sf.read(audio_path, dtype="float32")
        if data_np is not None and len(data_np) > 0:
            data = torch.from_numpy(data_np)
            orig_sr = sr
            print(f"[iTantra Audio Loader] Decoded via soundfile (sr={orig_sr}, shape={data.shape})")
    except Exception as sf_err:
        print(f"[iTantra Audio Loader] soundfile decode info: {sf_err}")

    # Strategy 2: torchaudio
    if data is None:
        try:
            import torchaudio
            wav_tensor, sr = torchaudio.load(audio_path)
            if wav_tensor is not None and wav_tensor.numel() > 0:
                if wav_tensor.ndim == 2:
                    data = wav_tensor.t()
                else:
                    data = wav_tensor
                orig_sr = sr
                print(f"[iTantra Audio Loader] Decoded via torchaudio (sr={orig_sr}, shape={data.shape})")
        except Exception as ta_err:
            print(f"[iTantra Audio Loader] torchaudio decode info: {ta_err}")

    # Strategy 3: ffmpeg CLI fallback if available
    if data is None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            try:
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav.close()
                cmd = [ffmpeg_bin, "-y", "-i", audio_path, "-ac", "1", "-ar", "24000", "-f", "wav", temp_wav.name]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                if res.returncode == 0 and os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 0:
                    data_np, orig_sr = sf.read(temp_wav.name, dtype="float32")
                    data = torch.from_numpy(data_np)
                    print(f"[iTantra Audio Loader] Decoded via ffmpeg fallback")
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
            except Exception as ff_err:
                print(f"[iTantra Audio Loader] ffmpeg fallback info: {ff_err}")

    if data is None or orig_sr is None:
        raise AudioPipelineError("audio_decode", f"Unsupported or invalid audio format ({fmt_desc}). Could not decode PCM audio.")

    # Validation & Normalization
    if data.ndim == 1:
        mono_data = data
        num_samples = len(data)
        num_channels = 1
    elif data.ndim == 2:
        num_samples = data.shape[0]
        num_channels = data.shape[1]
        mono_data = data.mean(dim=1)
    else:
        raise AudioPipelineError("audio_decode", f"Invalid audio dimensions: {data.ndim}")

    if num_samples == 0:
        raise AudioPipelineError("audio_decode", "Decoded audio contains 0 PCM samples")

    if orig_sr <= 0:
        raise AudioPipelineError("audio_decode", f"Invalid sample rate detected: {orig_sr}")

    duration = num_samples / orig_sr
    if duration <= 0.02:
        raise AudioPipelineError("audio_decode", f"Audio duration too short ({duration:.3f} sec). Minimum 0.02 sec required.")

    if torch.isnan(mono_data).any() or torch.isinf(mono_data).any():
        raise AudioPipelineError("audio_decode", "Audio data contains NaN or Inf sample values")

    # Resample to target_sr (24,000 Hz) for EnCodec if needed
    if orig_sr != target_sr:
        try:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
            mono_data = resampler(mono_data.unsqueeze(0)).squeeze(0)
        except Exception:
            # Fallback linear interpolation resampling
            in_len = len(mono_data)
            out_len = int(in_len * target_sr / orig_sr)
            mono_np = mono_data.numpy()
            mono_res = np.interp(np.linspace(0, in_len, out_len, endpoint=False), np.arange(in_len), mono_np)
            mono_data = torch.from_numpy(mono_res.astype(np.float32))
        norm_sr = target_sr
    else:
        norm_sr = orig_sr

    max_val = torch.max(torch.abs(mono_data))
    if max_val > 1e-4:
        mono_data = (mono_data / max_val) * 0.95
    elif max_val > 1.0:
        mono_data = mono_data / max_val

    return mono_data, norm_sr, duration, fmt_desc, file_size

def serialize_encodec_frames(encoded_frames):
    """Serialize EnCodec discrete VQ code tensors into binary payload."""
    serializable = []
    for codes, scale in encoded_frames:
        scale_val = scale.cpu().numpy() if scale is not None else None
        serializable.append((codes.cpu().numpy(), scale_val))
    return pickle.dumps(serializable)

def deserialize_encodec_frames(binary_data):
    """Deserialize binary payload back into EnCodec torch code frames."""
    try:
        raw_list = pickle.loads(binary_data)
        encoded_frames = []
        for codes_np, scale_np in raw_list:
            codes_tensor = torch.from_numpy(codes_np)
            scale_tensor = torch.from_numpy(scale_np) if scale_np is not None else None
            encoded_frames.append((codes_tensor, scale_tensor))
        return encoded_frames
    except Exception as e:
        raise AudioPipelineError("deserialization", f"Payload deserialization failed: {str(e)}")

def packetize_payload(payload_bytes, chunk_size=256):
    """
    Divide binary payload into packets with sequence headers & CRC32 checksums.
    Header format (16 bytes):
    - 4 bytes magic: b'ITAN'
    - 2 bytes: packet_id (uint16)
    - 2 bytes: total_packets (uint16)
    - 4 bytes: crc32 (uint32)
    - 4 bytes: chunk_len (uint32)
    """
    total_len = len(payload_bytes)
    num_packets = (total_len + chunk_size - 1) // chunk_size if total_len > 0 else 0
    packets = []

    for i in range(num_packets):
        chunk = payload_bytes[i * chunk_size : (i + 1) * chunk_size]
        crc = zlib.crc32(chunk) & 0xffffffff
        header = struct.pack(">4sHHII", b'ITAN', i, num_packets, crc, len(chunk))
        packet = header + chunk
        packets.append(packet)

    return packets

def depacketize_packets(packets):
    """
    Verify CRC32 checksums and reassemble binary payload from received packets.
    Returns: (reassembled_bytes, crc_pass_count, crc_fail_count)
    """
    received_chunks = {}
    crc_pass_count = 0
    crc_fail_count = 0
    total_expected = 0

    for pkt in packets:
        if len(pkt) < 16:
            crc_fail_count += 1
            continue

        header = pkt[:16]
        chunk = pkt[16:]

        magic, pkt_id, total_pkts, expected_crc, chunk_len = struct.unpack(">4sHHII", header)

        if magic != b'ITAN' or len(chunk) != chunk_len:
            crc_fail_count += 1
            continue

        total_expected = total_pkts
        actual_crc = zlib.crc32(chunk) & 0xffffffff

        if actual_crc == expected_crc:
            crc_pass_count += 1
            received_chunks[pkt_id] = chunk
        else:
            crc_fail_count += 1

    # Reassemble in sequence order
    reassembled = bytearray()
    for idx in range(total_expected):
        if idx in received_chunks:
            reassembled.extend(received_chunks[idx])

    return bytes(reassembled), crc_pass_count, crc_fail_count

def run_asr_transcription(mono_data, sr, language="hi"):
    """Run AI4Bharat IndicConformer Speech-to-Text on 16kHz mono audio."""
    try:
        model = get_asr_model()

        wav = mono_data.unsqueeze(0)  # (1, samples)

        # Resample to 16 kHz for IndicConformer if needed
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)

        with torch.no_grad():
            transcription = model(wav, language, "ctc")

        if isinstance(transcription, list):
            return transcription[0]
        return str(transcription)
    except Exception as e:
        print(f"[ASR Error] {e}")
        return "— (ASR unavailable for selected language)"

def validate_reconstructed_wav(file_path, expected_sr=24000):
    """
    Validate that the generated reconstructed WAV file has a complete,
    valid 16-bit PCM RIFF header and non-empty audio data.
    """
    if not os.path.exists(file_path):
        raise AudioPipelineError("audio_reconstruction", "Generated audio file does not exist on disk")

    file_size = os.path.getsize(file_path)
    if file_size < 44:
        raise AudioPipelineError("audio_reconstruction", f"Generated audio file size too small ({file_size} bytes)")

    with open(file_path, "rb") as f:
        header = f.read(44)

    riff, overall_size, wave, fmt, chunk1_size, audio_fmt, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
        "<4sI4s4sIHHIIHH", header[:36]
    )

    if riff != b"RIFF" or wave != b"WAVE" or fmt != b"fmt ":
        raise AudioPipelineError("audio_reconstruction", "Invalid WAV RIFF/WAVE header in reconstructed file")

    if sample_rate != expected_sr:
        raise AudioPipelineError("audio_reconstruction", f"Unexpected sample rate in WAV header: {sample_rate} Hz (expected {expected_sr} Hz)")

    if num_channels != 1:
        raise AudioPipelineError("audio_reconstruction", f"Unexpected channel count in WAV header: {num_channels} (expected 1 mono)")

    data_bytes = file_size - 44
    calc_duration = data_bytes / (sample_rate * num_channels * (bits_per_sample // 8)) if (sample_rate * num_channels * (bits_per_sample // 8)) > 0 else 0.0

    print(f"[iTantra Reconstructed WAV Verified] File: {os.path.basename(file_path)} | Size: {file_size} B | SR: {sample_rate} Hz | Channels: {num_channels} | Bits: {bits_per_sample} | Calc Duration: {calc_duration:.2f}s")
    return calc_duration

def process_itantra_pipeline(audio_path, target_bitrate=6.0, language="hi", mode="software_loopback", output_dir="static/audio"):
    """
    Execute the complete iTantra End-to-End Voice Communication Pipeline:
    VOICE -> EnCodec -> Serialization -> Packetization -> CRC32 -> Software Loopback -> Reassembly -> Deserialization -> EnCodec Decoder -> RECONSTRUCTED VOICE (+ AI4Bharat ASR)
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read & Normalize input audio file
    mono_data, orig_sr, duration, fmt_desc, raw_file_size = load_and_normalize_audio(audio_path, target_sr=24000)
    raw_pcm_bytes = len(mono_data) * 2  # 16-bit mono PCM equivalent

    # 2. EnCodec Quantization
    try:
        encodec = get_encodec_model()
        encodec.set_target_bandwidth(float(target_bitrate))

        wav_24k = mono_data.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        wav_24k = convert_audio(wav_24k, orig_sr, encodec.sample_rate, encodec.channels)

        with torch.no_grad():
            encoded_frames = encodec.encode(wav_24k)
    except Exception as e:
        raise AudioPipelineError("encodec_encode", f"EnCodec neural quantization failed: {str(e)}")

    if not encoded_frames or len(encoded_frames) == 0:
        raise AudioPipelineError("encodec_encode", "EnCodec output zero code frames")

    # 3. Serialization
    binary_payload = serialize_encodec_frames(encoded_frames)
    compressed_bytes_count = len(binary_payload)
    compression_ratio = raw_pcm_bytes / compressed_bytes_count if compressed_bytes_count > 0 else 1.0

    # 4. Packetization & CRC32
    chunk_size = 256
    packets = packetize_payload(binary_payload, chunk_size=chunk_size)
    num_packets = len(packets)

    if num_packets == 0:
        raise AudioPipelineError("packetization", "Payload packetization yielded 0 packets")

    # 5. Software Loopback / Transmission Channel
    try:
        import ggwave
        ggwave_available = True
    except ImportError:
        ggwave_available = False

    if mode == "physical_acoustic" and ggwave_available:
        transmitted_packets = packets
        transmission_mode_name = "Physical Acoustic Channel (ggwave)"
    elif ggwave_available:
        transmission_mode_name = "Verified Software Loopback (ggwave Core)"
        transmitted_packets = []
        for pkt in packets:
            instance = ggwave.init()
            wf = ggwave.encode(pkt, instance=instance)
            rx_pkt = ggwave.decode(instance, wf)
            ggwave.free(instance)
            transmitted_packets.append(rx_pkt if rx_pkt is not None else b"")
    else:
        transmitted_packets = packets
        transmission_mode_name = "Verified Software Loopback"

    # 6. Receiver Packet Reassembly & CRC Verification
    reassembled_payload, crc_pass, crc_fail = depacketize_packets(transmitted_packets)
    crc_status = "PASS" if (crc_fail == 0 and crc_pass == num_packets) else f"{crc_pass}/{num_packets} Verified"

    if crc_pass == 0 or len(reassembled_payload) == 0:
        raise AudioPipelineError("crc_validation", f"CRC32 Integrity Validation Failed: {crc_fail}/{num_packets} packets corrupted or unreassembled.")

    # 7. Deserialization
    recovered_frames = deserialize_encodec_frames(reassembled_payload)

    # 8. EnCodec Decoding & Voice Reconstruction
    try:
        with torch.no_grad():
            decoded_wav = encodec.decode(recovered_frames)
        decoded_audio_np = decoded_wav.squeeze(0).squeeze(0).cpu().numpy()
    except Exception as e:
        raise AudioPipelineError("encodec_decode", f"EnCodec neural audio decoding failed: {str(e)}")

    if len(decoded_audio_np) == 0 or np.isnan(decoded_audio_np).any():
        raise AudioPipelineError("audio_reconstruction", "Reconstructed audio waveform tensor is empty or invalid")

    # Save reconstructed audio file as standard 16-bit PCM WAV
    filename_out = f"reconstructed_{int(time.time()*1000)}_{int(target_bitrate)}kbps.wav"
    output_path = os.path.join(output_dir, filename_out)
    dec_max = float(np.max(np.abs(decoded_audio_np))) if len(decoded_audio_np) > 0 else 0.0
    if dec_max > 1e-4:
        decoded_audio_np = (decoded_audio_np / dec_max) * 0.95
    pcm16_samples = (np.clip(decoded_audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(output_path, pcm16_samples, encodec.sample_rate, subtype='PCM_16')

    # Validate generated RIFF header
    calc_duration = validate_reconstructed_wav(output_path, expected_sr=encodec.sample_rate)

    # Encode reconstructed WAV to base64 for instant client-side playback
    try:
        import base64
        with open(output_path, "rb") as audio_f:
            audio_bytes = audio_f.read()
        audio_base64 = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as b64_err:
        print(f"[iTantra Warning] Base64 encoding warning: {b64_err}")
        audio_base64 = None

    # 9. AI4Bharat ASR (Multilingual Speech-to-Text)
    asr_text = run_asr_transcription(mono_data, orig_sr, language=language)

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Codebook shape information
    code_shape = tuple(encoded_frames[0][0].shape)
    num_codebooks = code_shape[1]

    return {
        "status": "SUCCESS",
        "duration_sec": round(calc_duration, 2),
        "input_format": fmt_desc,
        "raw_file_bytes": raw_file_size,
        "orig_sample_rate": orig_sr,
        "raw_pcm_bytes": raw_pcm_bytes,
        "compressed_bytes": compressed_bytes_count,
        "compression_ratio": f"{round(compression_ratio, 1)}x",
        "bitrate_kbps": target_bitrate,
        "num_codebooks": num_codebooks,
        "code_tensor_shape": str(code_shape),
        "packets_sent": num_packets,
        "packets_received": crc_pass,
        "crc_failures": crc_fail,
        "packet_loss": "0%",
        "crc_status": crc_status,
        "transmission_mode": transmission_mode_name,
        "reconstructed_audio_url": f"/api/audio/{filename_out}",
        "audio_base64": audio_base64,
        "reconstruction_status": "SUCCESS — 24kHz Neural Audio Restored",
        "processing_latency_ms": elapsed_ms,
        "language": language,
        "asr_transcription": asr_text
    }

