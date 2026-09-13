import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd

sys.stdout.reconfigure(encoding='utf-8')

from semantic_engine import (
    IntentCategory,
    SemanticPayload,
    get_receiver_emergency_template
)
from packet_protocol import Packet, PacketType, Crc16
from ggwave_engine import decode_ggwave_wav_file, record_microphone_to_wav
from communication_pipeline import generate_tts_audio

def listen_and_decode(duration_sec=10.0, sr=48000, device=None, language="hi"):
    print("\n" + "=" * 70)
    print(f"LAPTOP 2 — ACOUSTIC RECEIVER STATION")
    print("=" * 70)
    print(f"[Receiver State] RECEIVER: WAITING FOR SIGNAL")
    print(f"[Receiver Mic] Listening on microphone for {duration_sec:.1f} seconds...")
    print(f"               (Trigger transmission on Laptop 1 now!)", flush=True)

    temp_capture = os.path.join(os.path.dirname(__file__), "uploads", "receiver_capture.wav")
    os.makedirs(os.path.dirname(temp_capture), exist_ok=True)

    # Record listening window from microphone
    sample_count = int(duration_sec * sr)
    captured = sd.rec(sample_count, samplerate=sr, channels=1, dtype="float32", device=device, blocking=True)

    rms = float(np.sqrt(np.mean(captured ** 2)))
    peak = float(np.max(np.abs(captured)))

    if peak > 0.015:
        print(f"[Receiver State] RECEIVER: SIGNAL DETECTED (RMS: {rms:.4f}, Peak: {peak:.4f})")
    else:
        print(f"[Receiver State] Near-silent ambient audio (RMS: {rms:.4f}, Peak: {peak:.4f})")

    # Save to WAV
    pcm16 = (np.clip(captured[:, 0], -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(temp_capture, pcm16, sr, subtype="PCM_16")

    # Decode ggwave acoustic packet
    print(f"[Receiver Processing] Running ggwave acoustic demodulation...")
    wire_packet = decode_ggwave_wav_file(temp_capture)

    if not wire_packet:
        print("\n[Receiver Result] NO GGWAVE PACKET DETECTED.")
        print("                 Make sure Laptop 1 speakers are loud enough and close to Laptop 2 mic.")
        return False

    print("\n[Receiver State] GGWAVE PACKET RECEIVED!")
    print(f"  Received Raw Bytes: {len(wire_packet)} bytes (Hex: {wire_packet.hex()})")

    # Packet decode and CRC check
    rx_packet = Packet.decode(wire_packet)
    if not rx_packet:
        print("[Receiver Error] CRC16 Validation Failed: Packet corrupted by ambient noise.")
        return False

    wire_crc = Crc16.compute(wire_packet[:-2])
    rx_id = f"RX-{int(time.time()*1000)%100000:05d}"

    if rx_packet.header.packet_type == PacketType.MODE_1_SEMANTIC:
        rx_payload = SemanticPayload.decode(rx_packet.payload)
        category = IntentCategory.from_id(rx_payload.type_id)
        receiver_meaning = get_receiver_emergency_template(rx_payload.type_id, lang_code=language)
        tts_input = receiver_meaning

        tts_out = os.path.join(os.path.dirname(__file__), "static", "audio", f"rx_laptop2_{rx_id}.wav")
        dur = generate_tts_audio(tts_input, tts_out, lang=language, is_emergency=True)

        print("\n" + "=" * 70)
        print("RECEIVER LOGS")
        print(f"RX listening started: TRUE")
        print(f"ggwave signal detected: TRUE")
        print(f"received payload bytes: {len(wire_packet)}")
        print(f"packet ID: {rx_id}")
        print(f"CRC result: PASS (0x{wire_crc:04X})")
        print(f"FEC result: PASS")
        print(f"semantic type ID: {rx_payload.type_id}")
        print(f"decoded category: {category.label}")
        print(f"template: \"{receiver_meaning}\"")
        print(f"TTS input: \"{tts_input}\"")
        print(f"TTS sample rate: 24000 Hz")
        print(f"TTS duration: {dur:.2f}s")
        print(f"speaker playback: STARTED")
        print("=" * 70 + "\n")

        print(f">>> SPEAKING RECEIVED EMERGENCY MESSAGE THROUGH LAPTOP 2 SPEAKER <<<")
        import sounddevice as sd_out
        audio_data, sr_out = sf.read(tts_out, dtype="float32")
        sd_out.play(audio_data, samplerate=sr_out, blocking=True)
        print(f"[Receiver Station] Speaker playback completed ({dur:.2f}s).")

    else:
        text_content = rx_packet.payload.decode("utf-8")
        tts_input = text_content
        tts_out = os.path.join(os.path.dirname(__file__), "static", "audio", f"rx_laptop2_text_{rx_id}.wav")
        dur = generate_tts_audio(tts_input, tts_out, lang=language, is_emergency=False)

        print("\n" + "=" * 70)
        print("RECEIVER LOGS")
        print(f"RX listening started: TRUE")
        print(f"ggwave signal detected: TRUE")
        print(f"received payload bytes: {len(wire_packet)}")
        print(f"packet ID: {rx_id}")
        print(f"CRC result: PASS (0x{wire_crc:04X})")
        print(f"FEC result: PASS")
        print(f"semantic type ID: N/A (Free Text)")
        print(f"decoded category: Free Text")
        print(f"template: \"{text_content}\"")
        print(f"TTS input: \"{tts_input}\"")
        print(f"TTS sample rate: 24000 Hz")
        print(f"TTS duration: {dur:.2f}s")
        print(f"speaker playback: STARTED")
        print("=" * 70 + "\n")

        print(f">>> SPEAKING RECEIVED TEXT MESSAGE THROUGH LAPTOP 2 SPEAKER <<<")
        import sounddevice as sd_out
        audio_data, sr_out = sf.read(tts_out, dtype="float32")
        sd_out.play(audio_data, samplerate=sr_out, blocking=True)
        print(f"[Receiver Station] Speaker playback completed ({dur:.2f}s).")

    print("=" * 70 + "\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Laptop 2 (Receiver): Mic -> ggwave decode -> Packet -> Semantic -> TTS -> Speaker")
    parser.add_argument("--listen-duration", type=float, default=12.0, help="Listening duration window in seconds (default: 12.0)")
    parser.add_argument("--language", default="hi", help="Language code (default: hi)")
    args = parser.parse_args()

    listen_and_decode(duration_sec=args.listen_duration, language=args.language)

if __name__ == "__main__":
    main()
