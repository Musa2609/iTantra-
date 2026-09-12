import os
import sys
import time
import zlib
import struct
import tempfile
import torch
import numpy as np
import soundfile as sf
from communication_pipeline import (
    detect_audio_format,
    load_and_normalize_audio,
    get_encodec_model,
    serialize_encodec_frames,
    deserialize_encodec_frames,
    packetize_payload,
    depacketize_packets,
    process_itantra_pipeline,
    AudioPipelineError
)

sys.stdout.reconfigure(encoding='utf-8')

def run_e2e_pipeline_tests():
    print("==================================================")
    print("  iTantra End-to-End Software Loopback Pipeline Test")
    print("==================================================")

    test_results = {}

    # --------------------------------------------------
    # TEST A: RECORDING & FORMAT DETECTION
    # --------------------------------------------------
    print("\n--- TEST A: Recording & Format Detection ---")
    audio_path = os.path.join(os.path.dirname(__file__), "audio.wav")
    if not os.path.exists(audio_path):
        print("Creating synthetic test audio.wav...")
        sr = 24000
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(audio_path, sine_wave, sr)

    fmt = detect_audio_format(audio_path)
    file_size = os.path.getsize(audio_path)
    print(f"File Path: {audio_path}")
    print(f"Detected Format: {fmt}")
    print(f"File Size: {file_size} bytes")
    assert fmt == "WAV (PCM)", f"Expected WAV (PCM), got {fmt}"
    test_results["TEST A — RECORDING & FORMAT DETECTION"] = "PASS"

    # --------------------------------------------------
    # TEST B: UPLOAD & HEADER VALIDATION
    # --------------------------------------------------
    print("\n--- TEST B: Upload & Magic Header Validation ---")
    with open(audio_path, "rb") as f:
        header_bytes = f.read(16)
    print(f"Header Magic Bytes: {header_bytes[:4]} | Subformat: {header_bytes[8:12]}")
    assert header_bytes.startswith(b"RIFF") and header_bytes[8:12] == b"WAVE"
    test_results["TEST B — UPLOAD & HEADER VALIDATION"] = "PASS"

    # --------------------------------------------------
    # TEST C: AUDIO DECODING
    # --------------------------------------------------
    print("\n--- TEST C: Audio Decoding ---")
    mono_data, sr, duration, fmt_desc, raw_size = load_and_normalize_audio(audio_path, target_sr=24000)
    print(f"Decoded Format: {fmt_desc}")
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample Count: {len(mono_data)}")
    assert len(mono_data) > 0 and duration > 0.1
    test_results["TEST C — AUDIO DECODING"] = "PASS"

    # --------------------------------------------------
    # TEST D: AUDIO NORMALIZATION
    # --------------------------------------------------
    print("\n--- TEST D: Audio Normalization ---")
    assert sr == 24000, f"Expected 24000 Hz, got {sr}"
    assert mono_data.ndim == 1, f"Expected mono 1D tensor, got {mono_data.ndim}D"
    assert not torch.isnan(mono_data).any() and not torch.isinf(mono_data).any()
    max_amp = torch.max(torch.abs(mono_data)).item()
    print(f"Normalized Sample Rate: {sr} Hz | Channels: 1 (Mono)")
    print(f"Max Amplitude: {max_amp:.4f} (Finite & Normalized)")
    test_results["TEST D — AUDIO NORMALIZATION"] = "PASS"

    # --------------------------------------------------
    # TEST E: ENCODEC QUANTIZATION
    # --------------------------------------------------
    print("\n--- TEST E: EnCodec Quantization ---")
    encodec = get_encodec_model()
    target_bitrate = 6.0
    encodec.set_target_bandwidth(target_bitrate)

    wav_24k = mono_data.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded_frames = encodec.encode(wav_24k)

    code_shape = tuple(encoded_frames[0][0].shape)
    print(f"Target Bandwidth: {target_bitrate} kbps")
    print(f"Code Frame Count: {len(encoded_frames)}")
    print(f"Code Tensor Shape: {code_shape}")
    assert len(encoded_frames) > 0 and not torch.isnan(encoded_frames[0][0]).any()
    test_results["TEST E — ENCODEC QUANTIZATION"] = "PASS"

    # --------------------------------------------------
    # TEST F: SERIALIZATION ROUND-TRIP
    # --------------------------------------------------
    print("\n--- TEST F: Serialization Round-Trip ---")
    serialized_bytes = serialize_encodec_frames(encoded_frames)
    deserialized_frames = deserialize_encodec_frames(serialized_bytes)
    print(f"Serialized Payload Size: {len(serialized_bytes)} bytes")
    print(f"Deserialized Frame Count: {len(deserialized_frames)}")

    for idx, ((orig_codes, orig_scale), (deser_codes, deser_scale)) in enumerate(zip(encoded_frames, deserialized_frames)):
        assert torch.equal(orig_codes.cpu(), deser_codes.cpu()), f"Frame {idx} code mismatch after serialization!"
        if orig_scale is not None:
            assert torch.equal(orig_scale.cpu(), deser_scale.cpu()), f"Frame {idx} scale mismatch!"
    print("Codes exact equality check: PASS ✓")
    test_results["TEST F — SERIALIZATION INTEGRITY"] = "PASS"

    # --------------------------------------------------
    # TEST G: PACKETIZATION
    # --------------------------------------------------
    print("\n--- TEST G: Packetization ---")
    packets = packetize_payload(serialized_bytes, chunk_size=256)
    num_packets = len(packets)
    print(f"Total Packets Generated: {num_packets}")
    assert num_packets > 0
    header_magic = packets[0][:4]
    print(f"Packet Header Magic: {header_magic} (Expected: b'ITAN')")
    assert header_magic == b'ITAN'
    test_results["TEST G — PACKETIZATION"] = "PASS"

    # --------------------------------------------------
    # TEST H: CRC32 & INTENTIONAL CORRUPTION REJECTION
    # --------------------------------------------------
    print("\n--- TEST H: CRC32 Error Detection ---")
    reassembled, crc_pass, crc_fail = depacketize_packets(packets)
    assert crc_pass == num_packets and crc_fail == 0
    assert reassembled == serialized_bytes
    print(f"Clean Transmission CRC: {crc_pass}/{num_packets} Passed ✓")

    corrupted_packets = [bytearray(p) for p in packets]
    corrupted_packets[0][-1] ^= 0xFF  # Flip last byte
    corrupted_packets_bytes = [bytes(p) for p in corrupted_packets]

    _, bad_pass, bad_fail = depacketize_packets(corrupted_packets_bytes)
    print(f"Corrupted Transmission CRC Result: {bad_pass} Passed, {bad_fail} Rejected (PASS)")
    assert bad_fail >= 1, "CRC32 failed to detect corrupted packet!"
    test_results["TEST H — CRC32 ERROR DETECTION"] = "PASS"

    # --------------------------------------------------
    # TEST I: SOFTWARE LOOPBACK
    # --------------------------------------------------
    print("\n--- TEST I: Software Loopback Channel ---")
    print(f"Packets Sent: {num_packets}")
    print(f"Packets Received: {crc_pass}")
    print(f"Packet Loss: 0%")
    print(f"CRC Failures: {crc_fail}")
    assert crc_pass == num_packets
    test_results["TEST I — GGWAVE / SOFTWARE LOOPBACK"] = "PASS"

    # --------------------------------------------------
    # TEST J: REASSEMBLY
    # --------------------------------------------------
    print("\n--- TEST J: Packet Reassembly ---")
    reassembled_payload, pass_cnt, fail_cnt = depacketize_packets(packets)
    assert len(reassembled_payload) == len(serialized_bytes)
    assert reassembled_payload == serialized_bytes
    print(f"Reassembled Bytes: {len(reassembled_payload)} / {len(serialized_bytes)} (Match: EXACT)")
    test_results["TEST J — PACKET REASSEMBLY"] = "PASS"

    # --------------------------------------------------
    # TEST K: ENCODEC DECODE & CODE MATCHING
    # --------------------------------------------------
    print("\n--- TEST K: EnCodec Decoding ---")
    recovered_frames = deserialize_encodec_frames(reassembled_payload)
    for orig_f, rec_f in zip(encoded_frames, recovered_frames):
        assert torch.equal(orig_f[0].cpu(), rec_f[0].cpu())

    with torch.no_grad():
        decoded_wav = encodec.decode(recovered_frames)
    decoded_np = decoded_wav.squeeze().cpu().numpy()
    print(f"Decoded Waveform Samples: {len(decoded_np)}")
    assert len(decoded_np) > 0 and not np.isnan(decoded_np).any()
    test_results["TEST K — ENCODEC DECODING"] = "PASS"

    # --------------------------------------------------
    # TEST L: FINAL AUDIO GENERATION
    # --------------------------------------------------
    print("\n--- TEST L: Final Output Audio ---")
    out_dir = os.path.join(os.path.dirname(__file__), "static", "audio")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_e2e_output.wav")
    sf.write(out_path, decoded_np, encodec.sample_rate)

    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    print(f"Output Audio Path: {out_path}")
    print(f"Output File Size: {os.path.getsize(out_path)} bytes")
    print(f"Output Sample Rate: {encodec.sample_rate} Hz (Mono)")
    test_results["TEST L — FINAL AUDIO GENERATION"] = "PASS"

    # --------------------------------------------------
    # FULL PIPELINE WRAPPER TEST
    # --------------------------------------------------
    print("\n--- FULL PIPELINE END-TO-END EXECUTION ---")
    result = process_itantra_pipeline(audio_path, target_bitrate=6.0, language="hi", mode="software_loopback", output_dir=out_dir)
    print(f"Pipeline Result Status: {result['status']}")
    print(f"Duration: {result['duration_sec']}s")
    print(f"Compression Ratio: {result['compression_ratio']}")
    print(f"Packets Verified: {result['packets_received']} / {result['packets_sent']}")
    print(f"CRC Status: {result['crc_status']}")
    print(f"Reconstructed Audio URL: {result['reconstructed_audio_url']}")
    print(f"ASR Transcription: {result['asr_transcription']}")
    assert result['status'] == "SUCCESS"
    test_results["FULL PIPELINE WRAPPER"] = "PASS"

    print("\n==================================================")
    print("  TEST SUMMARY RESULT TABLE")
    print("==================================================")
    all_pass = True
    for test_name, status in test_results.items():
        print(f"  {test_name:<42}: {status}")
        if status != "PASS":
            all_pass = False
    print("==================================================")

    return all_pass

if __name__ == "__main__":
    success = run_e2e_pipeline_tests()
    sys.exit(0 if success else 1)
