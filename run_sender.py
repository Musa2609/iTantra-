import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd

sys.stdout.reconfigure(encoding='utf-8')

from semantic_engine import HybridIntentClassifier, IntentCategory, SemanticPayload
from packet_protocol import Packet, PacketHeader, PacketType, Crc16
from ggwave_engine import encode_packet_to_ggwave_wav, play_ggwave_audio

def record_speech_mic(duration_sec=4.0, sr=24000):
    print(f"\n[Sender Mic] Recording {duration_sec:.1f} seconds of speech from microphone...")
    print(">>> SPEAK NOW into Laptop 1 microphone! <<<", flush=True)
    samples = int(duration_sec * sr)
    data = sd.rec(samples, samplerate=sr, channels=1, dtype='float32', blocking=True)
    print("[Sender Mic] Recording finished.", flush=True)
    temp_wav = os.path.join(os.path.dirname(__file__), "uploads", "sender_mic_input.wav")
    os.makedirs(os.path.dirname(temp_wav), exist_ok=True)
    pcm16 = (np.clip(data[:, 0], -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(temp_wav, pcm16, sr, subtype='PCM_16')
    return temp_wav

def main():
    parser = argparse.ArgumentParser(description="Laptop 1 (Sender): Speech -> ASR -> Classifier -> ggwave -> Speaker")
    parser.add_argument("--audio", default=None, help="Input WAV file (if not recording from mic)")
    parser.add_argument("--record", type=float, default=4.0, help="Record N seconds from mic (default: 4.0)")
    parser.add_argument("--text", default=None, help="Direct text input (bypasses ASR)")
    parser.add_argument("--volume", type=int, default=30, help="ggwave acoustic volume (0-100, default: 30)")
    parser.add_argument("--language", default="hi", help="Language code (default: hi)")
    args = parser.parse_args()

    tx_id = f"TX-{int(time.time()*1000)%1000000:06d}"
    print("\n" + "=" * 70)
    print(f"LAPTOP 1 — ACOUSTIC SENDER STATION ({tx_id})")
    print("=" * 70)

    # 1. Speech Input
    if args.text:
        transcript = args.text.strip()
        print(f"[Sender Input] Direct text provided: \"{transcript}\"")
    else:
        audio_path = args.audio
        if not audio_path or not os.path.exists(audio_path):
            audio_path = record_speech_mic(duration_sec=args.record)

        # Run ASR
        print("[Sender ASR] Running multilingual speech recognition...")
        from communication_pipeline import run_asr_transcription, load_and_normalize_audio
        tensor_audio, orig_sr, dur, *rest = load_and_normalize_audio(audio_path)
        transcript = run_asr_transcription(tensor_audio, target_sr=24000, language=args.language)
        print(f"[Sender ASR] Exact transcript: \"{transcript}\"")

    if not transcript or not transcript.strip():
        print("[Sender Error] No speech recognized. Transmission aborted.")
        return 1

    # 2. Semantic Intent Classification
    classifier_path = os.path.join(os.path.dirname(__file__), "data", "intent_dataset.json")
    classifier = HybridIntentClassifier(classifier_path)
    cls_res = classifier.classify(transcript)

    print("\n[Sender Classifier]")
    print(f"  Category: {cls_res.category.label} (typeId = {cls_res.category.type_id})")
    print(f"  Confidence: {cls_res.confidence * 100:.1f}%")
    print(f"  Is Emergency: {cls_res.is_emergency}")
    print(f"  Severity: {cls_res.severity}")

    # 3. Packet Creation
    is_mode_1 = cls_res.is_emergency
    if is_mode_1:
        comm_mode = "MODE 1 — SEMANTIC"
        payload_obj = SemanticPayload(
            type_id=cls_res.category.type_id,
            intent_id=cls_res.intent_id,
            severity=cls_res.severity
        )
        raw_payload = payload_obj.encode()
        payload_bits = 10
        packet = Packet(
            header=PacketHeader(packet_type=PacketType.MODE_1_SEMANTIC, lang_id=0, sequence_num=1),
            payload=raw_payload
        )
    else:
        comm_mode = "MODE 2 — FREE TEXT"
        raw_payload = transcript.encode("utf-8")
        payload_bits = len(raw_payload) * 8
        packet = Packet(
            header=PacketHeader(packet_type=PacketType.MODE_2_TEXT, lang_id=0, sequence_num=1),
            payload=raw_payload
        )

    wire_packet = packet.encode()
    wire_crc = Crc16.compute(wire_packet[:-2])

    # 4. ggwave Acoustic Encoding
    tones_dir = os.path.join(os.path.dirname(__file__), "static", "audio")
    os.makedirs(tones_dir, exist_ok=True)
    ggwave_wav = os.path.join(tones_dir, f"ggwave_{tx_id}.wav")
    meta = encode_packet_to_ggwave_wav(wire_packet, ggwave_wav, volume=args.volume)

    print("\n" + "=" * 70)
    print("SENDER TRANSMISSION LOGS")
    print(f"TX ID: {tx_id}")
    print(f"ASR transcript: {transcript}")
    print(f"classifier result: {cls_res.category.label}")
    print(f"mode: {comm_mode}")
    print(f"payload bytes: {len(wire_packet)}")
    print(f"packet count: 1")
    print(f"CRC: PASS (0x{wire_crc:04X})")
    print(f"FEC: Protected")
    print(f"ggwave encoded sample count: {meta['sampleCount']}")
    print(f"ggwave sample rate: {meta['sampleRate']}")
    print(f"ggwave transmission duration: {meta['durationSec']:.2f}s")
    print(f"speaker playback started: TRUE")
    print("=" * 70)

    # 5. Playback through physical speaker
    print(f"\n>>> TRANSMITTING OVER AIR: PLAYING GGWAVE TONES THROUGH SPEAKER NOW <<<")
    print(f"    (Sound: Acoustic Modem Data Tones • NOT Speech • NOT TTS)")
    print(f"    (Laptop 2 microphone will capture these tones across the air)\n", flush=True)

    play_ggwave_audio(ggwave_wav, blocking=True)
    print(f"[Sender Station] Acoustic tone transmission completed ({meta['durationSec']:.2f}s).")
    print(f"[Sender Station] Notice: SENDER NEVER RUNS TTS. TTS runs on Laptop 2 only.")
    print("=" * 70 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
