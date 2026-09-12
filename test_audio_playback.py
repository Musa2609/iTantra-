import os
import sys
import struct
import io
from app import app
from communication_pipeline import process_itantra_pipeline, validate_reconstructed_wav

sys.stdout.reconfigure(encoding='utf-8')

def test_audio_playback_serving():
    print("==================================================")
    print("  iTantra Audio Playback & Serving Verification")
    print("==================================================")

    # 1. Run pipeline to generate reconstructed audio
    audio_path = os.path.join(os.path.dirname(__file__), "audio.wav")
    static_audio_dir = os.path.join(os.path.dirname(__file__), "static", "audio")

    res = process_itantra_pipeline(audio_path, target_bitrate=6.0, language="hi", mode="software_loopback", output_dir=static_audio_dir)
    assert res['status'] == "SUCCESS"
    
    url = res['reconstructed_audio_url']
    filename = os.path.basename(url)
    full_file_path = os.path.join(static_audio_dir, filename)

    print(f"Generated WAV File: {full_file_path}")
    print(f"Reported Duration: {res['duration_sec']} seconds")

    # 2. Validate WAV Header
    calc_dur = validate_reconstructed_wav(full_file_path, expected_sr=24000)
    print(f"Validated RIFF Header Duration: {calc_dur:.2f} seconds")
    assert calc_dur > 0.1

    # 3. Test HTTP Serving via Flask Test Client
    client = app.test_client()

    print("\n--- Test 3.1: Full GET Audio Resource ---")
    resp = client.get(url)
    print(f"HTTP Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('Content-Type')}")
    print(f"Accept-Ranges: {resp.headers.get('Accept-Ranges')}")
    print(f"Content-Length: {resp.headers.get('Content-Length')}")
    
    assert resp.status_code == 200
    assert resp.headers.get('Content-Type') == 'audio/wav'
    assert resp.headers.get('Accept-Ranges') == 'bytes'
    assert int(resp.headers.get('Content-Length')) == os.path.getsize(full_file_path)

    print("\n--- Test 3.2: HTTP Range Request (Header Metadata Read) ---")
    resp_range = client.get(url, headers={'Range': 'bytes=0-43'})
    print(f"Range HTTP Status: {resp_range.status_code}")
    print(f"Range Returned Bytes: {len(resp_range.data)}")
    assert resp_range.status_code in (200, 206)
    assert resp_range.data.startswith(b"RIFF")

    print("\n==================================================")
    print("  AUDIO PLAYBACK SERVING VERIFICATION: PASS ✓")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = test_audio_playback_serving()
    sys.exit(0 if success else 1)
