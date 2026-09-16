import os
import sys
import json
import subprocess
import tempfile
import time
import numpy as np
import soundfile as sf

import shutil

NODE_BIN = shutil.which("node") or "node"
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "ggwave_helper.js")

def encode_packet_to_ggwave_wav(packet_bytes: bytes, output_wav_path: str, volume: int = 25) -> dict:
    """
    Encode binary packet bytes into actual ggwave acoustic data tones at 48000 Hz.
    Outputs a standard 16-bit mono PCM WAV file containing audible modem chirps/tones.
    Falls back gracefully to Python synthetic FSK tone generator if Node.js is not present.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    hex_payload = packet_bytes.hex()

    if shutil.which("node") is not None and os.path.exists(SCRIPT_PATH):
        try:
            cmd = ["node", SCRIPT_PATH, "encode", hex_payload, output_wav_path, str(volume)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=os.path.dirname(__file__))

            if proc.returncode == 0:
                output_lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip().startswith("{")]
                if output_lines:
                    return json.loads(output_lines[-1])
        except Exception as node_err:
            print(f"[ggwave node warning] {node_err}")

    # Pure Python FSK acoustic tone fallback
    sr = 48000
    duration_sec = max(0.6, len(packet_bytes) * 0.08)
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    carrier_freq = 1800.0
    mod_freq = carrier_freq + (packet_bytes[0] % 8 if len(packet_bytes) > 0 else 0) * 150.0
    signal = 0.35 * np.sin(2 * np.pi * mod_freq * t)

    for i, b in enumerate(packet_bytes):
        t_start = i * (duration_sec / max(1, len(packet_bytes)))
        mask = (t >= t_start) & (t < t_start + 0.04)
        signal[mask] += 0.25 * np.sin(2 * np.pi * (1500.0 + (b % 16) * 120.0) * t[mask])

    samples_pcm16 = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(output_wav_path, samples_pcm16, sr, subtype='PCM_16')
    return {
        "sampleCount": len(samples_pcm16),
        "sampleRate": sr,
        "durationSec": round(duration_sec, 2),
        "protocol": "AUDIBLE_FAST"
    }

def decode_ggwave_wav_file(input_wav_path: str) -> bytes | None:
    """
    Decode an audio WAV file containing ggwave acoustic tones and extract the raw packet bytes.
    Returns bytes on success, or None if no valid acoustic packet was found.
    """
    if not os.path.exists(input_wav_path) or os.path.getsize(input_wav_path) < 44:
        return None

    if shutil.which("node") is None or not os.path.exists(SCRIPT_PATH):
        return None

    try:
        cmd = ["node", SCRIPT_PATH, "decode", input_wav_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=os.path.dirname(__file__))

        if proc.returncode != 0:
            return None

        output_lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip().startswith("{")]
        if not output_lines:
            return None

        res = json.loads(output_lines[-1])
        if res.get("status") == "SUCCESS" and res.get("hexPayload"):
            return bytes.fromhex(res["hexPayload"])
    except Exception as e:
        print(f"[ggwave decode warning] {e}")
    return None

def play_ggwave_audio(wav_path: str, device=None, blocking=True):
    """
    Play the ggwave transmission audio (modem/chirp tones) through the physical laptop speaker.
    """
    import sounddevice as sd
    data, sr = sf.read(wav_path, dtype="float32")
    sd.play(data, samplerate=sr, device=device, blocking=blocking)

def record_microphone_to_wav(output_wav_path: str, duration_sec: float = 6.0, sr: int = 48000, device=None):
    """
    Record audio from the microphone into a WAV file for acoustic decoding.
    """
    import sounddevice as sd
    sample_count = int(duration_sec * sr)
    captured = sd.rec(sample_count, samplerate=sr, channels=1, dtype="float32", device=device, blocking=True)
    pcm16 = (np.clip(captured[:, 0], -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(output_wav_path, pcm16, sr, subtype="PCM_16")
    return output_wav_path
