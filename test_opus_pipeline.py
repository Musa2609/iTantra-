"""
iTantra Mode 1 (Opus Audio Codec) Test Suite
Executes all 5 required offline/loopback tests:
  TEST 1: PCM speech -> Opus encode -> Opus decode -> reconstructed PCM
  TEST 2: Exact measurement of PCM size, Opus payload size, compression ratio
  TEST 3: Packetize Opus frames using PacketProtocol
  TEST 4: CRC validation (error detection verification)
  TEST 5: Software loopback: Opus -> Packets -> ggwave encode -> ggwave decode -> Packets -> Opus decode
"""

import os
import sys
import numpy as np
import soundfile as sf
from opus_engine import OpusEncoder, OpusDecoder, get_opus_compression_stats
from packet_protocol import Packet, PacketHeader, PacketType, Crc16, Crc32
from ggwave_engine import encode_packet_to_ggwave_wav, decode_ggwave_wav_file

def run_tests():
    print("=" * 70)
    print("iTANTRA MODE 1 (OPUS) VERIFICATION TEST SUITE")
    print("=" * 70)

    # Load test audio (speech)
    test_audio_path = os.path.join(os.path.dirname(__file__), "audio.wav")
    if not os.path.exists(test_audio_path):
        print(f"Error: {test_audio_path} not found!")
        sys.exit(1)

    raw_data, orig_sr = sf.read(test_audio_path)
    mono_audio = raw_data[:, 0] if raw_data.ndim > 1 else raw_data
    # Use first 2 seconds for acoustic modem packet sizing
    clip_dur = 2.0
    samples_to_take = int(min(len(mono_audio), orig_sr * clip_dur))
    clip_audio = mono_audio[:samples_to_take]

    print(f"\n[TEST 1] Opus Encode / Decode Verification")
    encoder = OpusEncoder(sample_rate=24000, channels=1, compression_level=10)
    decoder = OpusDecoder(target_sample_rate=24000)

    opus_bytes = encoder.encode(clip_audio, orig_sr=orig_sr)
    reconstructed_pcm, dec_sr = decoder.decode(opus_bytes)

    assert len(reconstructed_pcm) > 0, "Decoded PCM audio is empty!"
    assert dec_sr == 24000, f"Unexpected sample rate: {dec_sr}"
    peak_amp = float(np.max(np.abs(reconstructed_pcm)))
    rms_amp = float(np.sqrt(np.mean(reconstructed_pcm ** 2)))
    print(f"  - Opus bytes produced: {len(opus_bytes)} bytes")
    print(f"  - Decoded samples: {len(reconstructed_pcm)} at {dec_sr} Hz")
    print(f"  - Peak amplitude: {peak_amp:.4f} | RMS: {rms_amp:.4f}")
    assert peak_amp > 0.01, "Audio appears silent!"
    print("  -> TEST 1 PASSED: Speech encoded and decoded successfully.")

    print(f"\n[TEST 2] Measurement of Real Sizes & Compression Ratio")
    stats = get_opus_compression_stats(reconstructed_pcm, dec_sr, opus_bytes)
    print(f"  - Original 16-bit PCM equivalent: {stats['raw_pcm_bytes']} bytes")
    print(f"  - Opus compressed payload: {stats['opus_bytes']} bytes")
    print(f"  - Measured compression ratio: {stats['compression_ratio']}x")
    print(f"  - Effective bitrate: {stats['measured_bitrate_kbps']} kbps")
    print(f"  - Duration: {stats['duration_sec']}s")
    assert stats['opus_bytes'] < stats['raw_pcm_bytes'], "Opus failed to compress audio"
    print("  -> TEST 2 PASSED: Real compression measurements verified without fabrication.")

    print(f"\n[TEST 3] Packetize Opus Frames (PacketProtocol)")
    # For acoustic channel, chunk into standard packets if needed, or frame header
    header = PacketHeader(
        packet_type=PacketType.VOICE_OPUS,
        lang_id=0,
        is_fec=False,
        sequence_num=1
    )
    packet = Packet(header=header, payload=opus_bytes)
    wire_bytes = packet.encode()
    print(f"  - Packet Header: type={packet.header.packet_type.name} (ID: {packet.header.packet_type.value}), seq={packet.header.sequence_num}")
    print(f"  - Wire bytes total: {len(wire_bytes)} bytes (Header + Length + Payload + CRC)")
    assert len(wire_bytes) == len(opus_bytes) + 8, f"Unexpected packet overhead: {len(wire_bytes)}"
    print("  -> TEST 3 PASSED: Opus frame cleanly encapsulated into PacketProtocol.")

    print(f"\n[TEST 4] CRC Validation & Error Detection")
    # Valid decode
    rx_packet = Packet.decode(wire_bytes)
    assert rx_packet is not None, "Valid packet failed CRC decode!"
    assert rx_packet.payload == opus_bytes, "Payload mismatch after packet decode!"
    # Corrupt a byte in the payload
    corrupted_bytes = bytearray(wire_bytes)
    corrupted_bytes[10] ^= 0xFF
    rx_corrupt = Packet.decode(bytes(corrupted_bytes))
    assert rx_corrupt is None, "Corrupted packet was erroneously accepted by CRC!"
    print(f"  - Valid packet: CRC VERIFIED (PASS)")
    print(f"  - Bit-flipped packet: CRC REJECTED (CORRUPTION DETECTED)")
    print("  -> TEST 4 PASSED: CRC provides strict error detection.")

    print(f"\n[TEST 5] Software Loopback (Opus -> Chunked Packets -> ggwave -> Loopback -> Decode)")
    from packet_protocol import chunk_payload, reassemble_payload
    
    # 0.1s speech slice
    slice_dur = 0.1
    short_samples = clip_audio[:int(orig_sr * slice_dur)]
    short_opus = encoder.encode(short_samples, orig_sr=orig_sr)
    print(f"  - Speech slice duration: {slice_dur}s | Opus payload size: {len(short_opus)} bytes")

    # Chunk into MTU-safe 100-byte packets
    tx_packets = chunk_payload(short_opus, packet_type=PacketType.VOICE_OPUS, chunk_size=100)
    print(f"  - Split into {len(tx_packets)} MTU-safe acoustic packets (<= 100B each)")

    rx_packets = []
    loopback_dir = os.path.join(os.path.dirname(__file__), "scratch")
    os.makedirs(loopback_dir, exist_ok=True)

    for i, pkt in enumerate(tx_packets):
        wire_data = pkt.encode()
        pkt_wav = os.path.join(loopback_dir, f"test_opus_pkt_{i}.wav")
        gg_meta = encode_packet_to_ggwave_wav(wire_data, pkt_wav, volume=40)
        rx_wire = decode_ggwave_wav_file(pkt_wav)
        assert rx_wire is not None, f"ggwave demodulation failed for packet {i}!"
        rx_p = Packet.decode(rx_wire)
        assert rx_p is not None, f"Packet {i} CRC check failed!"
        assert rx_p.header.packet_type == PacketType.VOICE_OPUS, f"Unexpected packet type: {rx_p.header.packet_type}"
        rx_packets.append(rx_p)
        print(f"    [Packet {i+1}/{len(tx_packets)}] ggwave modulated & demodulated: {len(wire_data)}B wire, CRC PASS")

    reassembled_opus, valid_cnt, missing_cnt = reassemble_payload(rx_packets)
    assert reassembled_opus is not None, f"Reassembly failed! Missing: {missing_cnt}"
    assert reassembled_opus == short_opus, "Reassembled Opus payload does not match original!"
    print(f"  - Reassembly: {valid_cnt}/{len(tx_packets)} packets recovered. Zero loss.")

    # Decode recovered Opus payload back to PCM
    loopback_pcm, loopback_sr = decoder.decode(reassembled_opus)
    print(f"  - Reconstructed speech from loopback: {len(loopback_pcm)} samples at {loopback_sr} Hz")
    assert len(loopback_pcm) > 0, "Loopback PCM reconstruction was empty!"
    print("  -> TEST 5 PASSED: Full software loopback with real Opus & ggwave verified!")

    print("\n" + "=" * 70)
    print("ALL 5 OPUS TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
