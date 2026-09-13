import os
import sys
import json
import subprocess
import tempfile
import time
import numpy as np
import soundfile as sf

NODE_BIN = "node"
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "ggwave_helper.js")

def encode_packet_to_ggwave_wav(packet_bytes: bytes, output_wav_path: str, volume: int = 25) -> dict:
    """
    Encode binary packet bytes into actual ggwave acoustic data tones at 48000 Hz.
    Outputs a standard 16-bit mono PCM WAV file containing audible modem chirps/tones.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    hex_payload = packet_bytes.hex()

    cmd = [NODE_BIN, SCRIPT_PATH, "encode", hex_payload, output_wav_path, str(volume)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=os.path.dirname(__file__))

    if proc.returncode != 0:
        raise RuntimeError(f"ggwave encode failed: {proc.stderr}")

    # Parse JSON output from last line
    output_lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip().startswith("{")]
    if not output_lines:
        raise RuntimeError(f"Unexpected ggwave output: {proc.stdout}")

    meta = json.loads(output_lines[-1])
    return meta

def decode_ggwave_wav_file(input_wav_path: str) -> bytes | None:
    """
    Decode an audio WAV file containing ggwave acoustic tones and extract the raw packet bytes.
    Returns bytes on success, or None if no valid acoustic packet was found.
    """
    if not os.path.exists(input_wav_path) or os.path.getsize(input_wav_path) < 44:
        return None

    cmd = [NODE_BIN, SCRIPT_PATH, "decode", input_wav_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", cwd=os.path.dirname(__file__))

    if proc.returncode != 0:
        return None

    output_lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip().startswith("{")]
    if not output_lines:
        return None

    res = json.loads(output_lines[-1])
    if res.get("status") == "SUCCESS" and res.get("hexPayload"):
        return bytes.fromhex(res["hexPayload"])
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
