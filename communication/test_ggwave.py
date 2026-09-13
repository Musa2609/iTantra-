import argparse
import os
import struct
import sys
import time
import wave
import zlib

PACKET_MAGIC = b"ITG2"
PACKET_HEADER = struct.Struct("!4sHHHI")
# ggwave 0.4.3 truncates messages above 140 bytes; reserve space for our 14-byte header.
PACKET_PAYLOAD_SIZE = 100
GGWAVE_SAMPLE_RATE = 48000
PHYSICAL_LEAD_SECONDS = 3
PHYSICAL_GAP_SECONDS = 0.25


def packetize(data: bytes):
    """Split bytes payload into sequenced packets with magic and CRC32."""
    total = (len(data) + PACKET_PAYLOAD_SIZE - 1) // PACKET_PAYLOAD_SIZE
    if total == 0:
        total = 1
    if total > 0xFFFF:
        raise ValueError("Payload requires too many packets")
    packets = []
    for number, start in enumerate(range(0, max(len(data), 1), PACKET_PAYLOAD_SIZE)):
        payload = data[start:start + PACKET_PAYLOAD_SIZE]
        packets.append(PACKET_HEADER.pack(PACKET_MAGIC, number, total, len(payload), zlib.crc32(payload)) + payload)
    return packets


def reassemble_packets(received_packets, expected_total):
    """Reassemble ordered payload bytes from received packets."""
    expected_numbers = set(range(expected_total))
    if set(received_packets) != expected_numbers:
        missing = sorted(expected_numbers - set(received_packets))
        raise ValueError(f"Missing packet numbers: {missing}")
    return b"".join(received_packets[number] for number in range(expected_total))


def validate_packet(packet, expected_total):
    """Validate packet header magic, total, length, and CRC32 checksum."""
    if len(packet) < PACKET_HEADER.size:
        return None
    magic, number, total, payload_length, checksum = PACKET_HEADER.unpack_from(packet)
    payload = packet[PACKET_HEADER.size:]
    if magic != PACKET_MAGIC or total != expected_total or len(payload) != payload_length:
        return None
    if zlib.crc32(payload) != checksum:
        return None
    return number, payload


def test_error_detection(data: bytes):
    """Verify that CRC32 detects single-bit corruption in packet payload."""
    packets = packetize(data)
    corrupted = bytearray(packets[0])
    corrupted[-1] ^= 1
    return validate_packet(bytes(corrupted), len(packets)) is None


def ggwave_encode_decode(data: bytes, ggwave_module, waveform_path: str):
    """Software loopback test: encode packets to audio, decode, and verify CRC32."""
    packets = packetize(data)
    received = {}
    corrupted_packets = 0
    waveforms = []
    transmitted_packets = 0
    started = time.perf_counter()

    for packet in packets:
        decoded_packet = None
        for _attempt in range(3):
            instance = ggwave_module.init()
            waveform = ggwave_module.encode(packet, instance=instance)
            waveforms.append(waveform)
            transmitted_packets += 1
            try:
                decoded_packet = ggwave_module.decode(instance, waveform)
            finally:
                ggwave_module.free(instance)
            if decoded_packet is not None:
                break

        if decoded_packet is None or len(decoded_packet) < PACKET_HEADER.size:
            corrupted_packets += 1
            continue

        validated = validate_packet(decoded_packet, len(packets))
        if validated is None:
            corrupted_packets += 1
        else:
            number, payload = validated
            received[number] = payload

    elapsed = time.perf_counter() - started
    waveform_bytes = b"".join(waveforms)

    with wave.open(waveform_path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(4)
        output.setframerate(GGWAVE_SAMPLE_RATE)
        output.writeframes(waveform_bytes)

    waveform_duration = len(waveform_bytes) / 4 / GGWAVE_SAMPLE_RATE
    recovered = reassemble_packets(received, len(packets))
    return recovered, elapsed, len(packets), transmitted_packets, corrupted_packets, waveform_duration


def physical_ggwave_encode_decode(data: bytes, ggwave_module):
    """Hardware loopback test on the local speaker/microphone."""
    import numpy as np
    import sounddevice as sd

    packets = packetize(data)
    received = {}
    corrupted_packets = 0
    started = time.perf_counter()
    waveforms = [ggwave_module.encode(packet) for packet in packets]
    samples = np.frombuffer(b"".join(waveforms), dtype=np.float32).reshape(-1, 1)

    instance = ggwave_module.init()
    try:
        captured = sd.playrec(
            samples,
            samplerate=GGWAVE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        decoded_stream = ggwave_module.decode(instance, captured.tobytes())
    finally:
        ggwave_module.free(instance)

    if decoded_stream:
        validated = validate_packet(decoded_stream, len(packets))
        if validated is None:
            corrupted_packets += 1
        else:
            number, payload = validated
            received[number] = payload

    elapsed = time.perf_counter() - started
    recovered = b"" if len(received) != len(packets) else reassemble_packets(received, len(packets))
    return recovered, len(packets), len(packets), corrupted_packets, elapsed, len(received)


def physical_probe(payloads, ggwave_module, device=None):
    """Send and record probe packets through audio hardware."""
    import numpy as np
    import sounddevice as sd

    results = []
    for payload in payloads:
        instance = ggwave_module.init()
        waveform = ggwave_module.encode(payload, instance=instance)
        samples = np.frombuffer(waveform, dtype=np.float32).reshape(-1, 1)
        try:
            captured = sd.playrec(
                samples,
                samplerate=GGWAVE_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device,
                blocking=True,
            )
            decoded = ggwave_module.decode(instance, captured.tobytes())
            results.append((decoded == payload, float(np.sqrt(np.mean(captured ** 2))), float(np.max(np.abs(captured)))))
        finally:
            ggwave_module.free(instance)
    return results


def microphone_level(seconds=3, output_path="microphone_test.wav", device=None):
    """Measure background noise levels through the microphone."""
    import numpy as np
    import sounddevice as sd

    captured = sd.rec(
        int(GGWAVE_SAMPLE_RATE * seconds),
        samplerate=GGWAVE_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        blocking=True,
    )
    samples = np.clip(captured[:, 0], -1, 1)
    with wave.open(output_path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(GGWAVE_SAMPLE_RATE)
        output.writeframes((samples * 32767).astype(np.int16).tobytes())
    return float(np.sqrt(np.mean(captured ** 2))), float(np.max(np.abs(captured)))


def physical_send_two_laptop(data: bytes, ggwave_module, output_device=None):
    """Play acoustic signal over speaker for a two-laptop setup."""
    import numpy as np
    import sounddevice as sd

    packets = packetize(data)
    waveforms = [np.frombuffer(ggwave_module.encode(packet), dtype=np.float32) for packet in packets]
    slot_samples = max(len(waveform) for waveform in waveforms) + int(PHYSICAL_GAP_SECONDS * GGWAVE_SAMPLE_RATE)
    slots = [np.pad(waveform, (0, slot_samples - len(waveform))) for waveform in waveforms]
    signal = np.concatenate(slots)

    print(f"\n[Sender] Ready to transmit {len(packets)} packet(s).")
    for sec in range(PHYSICAL_LEAD_SECONDS, 0, -1):
        print(f"[Sender] Transmission begins in {sec} second(s)...", flush=True)
        time.sleep(1.0)

    print(f"[Sender] >>> BROADCASTING ACOUSTIC TONES NOW <<< (listen to speakers)", flush=True)
    sd.play(signal, samplerate=GGWAVE_SAMPLE_RATE, device=output_device, blocking=True)
    print(f"[Sender] Playback finished.")
    return len(packets), len(signal) / GGWAVE_SAMPLE_RATE


def physical_receive_two_laptop(packet_count: int, ggwave_module, input_device=None):
    """Record acoustic signal from microphone for a two-laptop setup."""
    import numpy as np
    import sounddevice as sd

    probe_payload = b"\x00" * PACKET_PAYLOAD_SIZE
    probe_packet = PACKET_HEADER.pack(
        PACKET_MAGIC,
        0,
        packet_count,
        PACKET_PAYLOAD_SIZE,
        zlib.crc32(probe_payload),
    ) + probe_payload
    probe_waveform = np.frombuffer(ggwave_module.encode(probe_packet), dtype=np.float32)
    slot_samples = len(probe_waveform) + int(PHYSICAL_GAP_SECONDS * GGWAVE_SAMPLE_RATE)
    total_samples = int(PHYSICAL_LEAD_SECONDS * GGWAVE_SAMPLE_RATE) + packet_count * slot_samples + int(PHYSICAL_LEAD_SECONDS * GGWAVE_SAMPLE_RATE)
    print(f"Physical receiver listening for {packet_count} packets...")
    captured = sd.rec(total_samples, samplerate=GGWAVE_SAMPLE_RATE, channels=1, dtype="float32", device=input_device, blocking=True)
    received = {}
    corrupted = 0
    lead_samples = int(PHYSICAL_LEAD_SECONDS * GGWAVE_SAMPLE_RATE)

    for number in range(packet_count):
        start = lead_samples + number * slot_samples
        end = start + slot_samples
        instance = ggwave_module.init()
        try:
            decoded = ggwave_module.decode(instance, captured[start:end, 0].tobytes())
        finally:
            ggwave_module.free(instance)
        validated = validate_packet(decoded, packet_count) if decoded else None
        if validated is None:
            corrupted += 1
        else:
            packet_number, payload = validated
            received[packet_number] = payload

    recovered = b""
    if len(received) == packet_count:
        recovered = reassemble_packets(received, packet_count)
    return recovered, packet_count, len(received), corrupted, total_samples / GGWAVE_SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser(description="Acoustic text packet transmission over ggwave with CRC32 integrity")
    parser.add_argument("--text", default="iTantra acoustic communication: AI4Bharat ASR -> IndicTrans2 -> ggwave -> Indic Parler-TTS", help="Text to packetize and transmit")
    parser.add_argument("--physical", action="store_true", help="Run local hardware loopback (speaker to microphone)")
    parser.add_argument("--physical-probe-only", action="store_true", help="Run short microphone and probe checks")
    parser.add_argument("--physical-send", action="store_true", help="Play acoustic packets through this laptop's speaker")
    parser.add_argument("--physical-receive", action="store_true", help="Record acoustic packets through this laptop's microphone")
    parser.add_argument("--packet-count", type=int, help="Expected packet count for --physical-receive")
    parser.add_argument("--waveform-output", default="ggwave_transmission.wav", help="Output WAV path for generated ggwave signal")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== iTantra Acoustic Packet Transmission Test ===")

    try:
        import ggwave
    except ImportError as exc:
        print(f"Error: ggwave library not found: {exc}")
        return 1

    if args.physical_probe_only:
        try:
            import sounddevice as sd
            input_device, output_device = sd.default.device
            rms, peak = microphone_level(device=input_device)
            print(f"Microphone RMS: {rms:.6f}")
            print(f"Microphone peak: {peak:.6f}")
            probe_payloads = [b"ITG2-probe-1"]
            multi_packets = packetize(b"physical-multi-packet-probe-" * 6)
            probe_payloads.extend(multi_packets[:2])
            one_result = physical_probe([probe_payloads[0]], ggwave, (input_device, output_device))[0]
            multi_results = physical_probe(probe_payloads[1:], ggwave, (input_device, output_device))
            print(f"1-packet ggwave speaker-microphone: {'PASS' if one_result[0] else 'FAIL'}")
            print(f"1-packet capture RMS: {one_result[1]:.6f}")
            print(f"2-packet ggwave speaker-microphone: {'PASS' if all(result[0] for result in multi_results) else 'FAIL'}")
            print(f"2-packet successful packets: {sum(result[0] for result in multi_results)}/{len(multi_results)}")
            return 0 if one_result[0] and all(result[0] for result in multi_results) else 1
        except Exception as error:
            print(f"Physical probe error: {type(error).__name__}: {error}")
            return 1

    if args.physical_send and args.physical_receive:
        parser.error("choose only one of --physical-send or --physical-receive")
    if args.physical_receive and not args.packet_count:
        parser.error("--physical-receive requires --packet-count from the sender")

    payload_bytes = args.text.encode("utf-8")
    print(f"Input text: \"{args.text}\"")
    print(f"Payload size: {len(payload_bytes)} bytes")

    if args.physical_send:
        print("Initializing audio hardware...", flush=True)
        import sounddevice as sd
        output_dev = sd.default.device[1] if sd.default.device[1] is not None else None
        sent_packets, duration = physical_send_two_laptop(payload_bytes, ggwave, output_dev)
        print(f"Physical packets sent: {sent_packets}")
        print(f"Physical signal duration: {duration:.2f} seconds")
        print("PHYSICAL SENDER: COMPLETE")
        return 0

    if args.physical_receive:
        try:
            import sounddevice as sd
            received, total, sent, corrupted, duration = physical_receive_two_laptop(args.packet_count, ggwave, sd.default.device[0])
            physical_passed = len(received) > 0 and received == payload_bytes
            print(f"Physical packets expected: {total}")
            print(f"Physical packets received: {sent}")
            print(f"Physical packets lost: {total - sent}")
            print(f"Physical corrupted packets: {corrupted}")
            print(f"Physical data integrity: {'PASS' if physical_passed else 'FAIL'}")
            if received:
                print(f"Received text: \"{received.decode('utf-8', errors='replace')}\"")
            return 0 if physical_passed else 1
        except Exception as error:
            print(f"PHYSICAL RECEIVER: FAIL ({type(error).__name__}: {error})")
            return 1

    waveform_path = os.path.join(os.path.dirname(__file__), args.waveform_output)
    print("ggwave software loopback transmission started...")

    try:
        received, elapsed, packet_count, transmitted_packets, corrupted_packets, waveform_duration = ggwave_encode_decode(payload_bytes, ggwave, waveform_path)
        exact = payload_bytes == received
    except Exception as error:
        print(f"Transmission error: {type(error).__name__}: {error}")
        return 1

    print("ggwave transmission completed")
    print(f"Number of packets: {packet_count}")
    print(f"Total packets transmitted: {transmitted_packets}")
    print(f"Total transmitted bytes: {transmitted_packets * (PACKET_HEADER.size + PACKET_PAYLOAD_SIZE)} bytes")
    print(f"ggwave waveform duration: {waveform_duration:.2f} seconds")
    print(f"Transmission time: {elapsed:.2f} seconds")
    print(f"Received bytes: {len(received)}")
    print(f"Corrupted packets: {corrupted_packets}")
    print(f"FEC/error detection test: {'PASS' if test_error_detection(payload_bytes) else 'FAIL'} (CRC32)")
    print(f"DATA INTEGRITY: {'PASS' if exact else 'FAIL'}")
    print(f"Recovered text: \"{received.decode('utf-8')}\"")
    print("GGWAVE TEST: PASS" if exact else "GGWAVE TEST: FAIL")

    if args.physical:
        try:
            physical_received, physical_total, physical_sent, physical_corrupted, physical_elapsed, physical_received_count = physical_ggwave_encode_decode(payload_bytes, ggwave)
            physical_passed = physical_received == payload_bytes
            print("Physical metrics:")
            print(f"  Total packets sent: {physical_sent}")
            print(f"  Packets received: {physical_received_count}")
            print(f"  Packets lost: {physical_total - physical_received_count}")
            print(f"  Corrupted packets: {physical_corrupted}")
            print(f"  Successful packet percentage: {physical_received_count / physical_total * 100:.2f}%")
            print(f"  Data matched exactly: {'YES' if physical_passed else 'NO'}")
            print(f"  Physical transmission time: {physical_elapsed:.2f} seconds")
            print(f"PHYSICAL ACOUSTIC TEST: {'PASS' if physical_passed else 'FAIL'}")
        except Exception as error:
            print(f"PHYSICAL ACOUSTIC TEST: FAIL ({type(error).__name__}: {error})")
            return 1
    else:
        print("PHYSICAL ACOUSTIC TEST: NOT RUN")

    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
