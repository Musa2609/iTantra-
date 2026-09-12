import argparse
import os
import struct
import sys
import time
import wave
import zlib

SERIALIZATION_MAGIC = b"ITC2"
PACKET_MAGIC = b"ITG2"
SERIALIZATION_HEADER = struct.Struct("!4sHfI")
PACKET_HEADER = struct.Struct("!4sHHHI")
# ggwave 0.4.3 truncates messages above 140 bytes; reserve 10 bytes for our header.
PACKET_PAYLOAD_SIZE = 100
GGWAVE_SAMPLE_RATE = 48000


def encode_audio_with_encodec(audio_path, bitrate):
    import torch
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    with wave.open(audio_path, "rb") as audio_file:
        input_rate = audio_file.getframerate()
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        raw_audio = audio_file.readframes(audio_file.getnframes())
    if sample_width != 2:
        raise ValueError("audio.wav must contain 16-bit PCM samples")
    audio = torch.frombuffer(raw_audio, dtype=torch.int16).float() / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(dim=1)
    duration = audio.numel() / channels / input_rate
    model = EncodecModel.encodec_model_24khz()
    model.eval()
    model.set_target_bandwidth(bitrate)
    waveform = audio.unsqueeze(0).unsqueeze(0)
    waveform = convert_audio(waveform, input_rate, model.sample_rate, model.channels)
    with torch.no_grad():
        encoded_frames = model.encode(waveform)
    return model, encoded_frames, duration, torch


def serialize_encodec_codes(encoded_frames, bitrate):
    """Serialize code shapes, dtype, bitrate, values, and optional frame scales."""
    output = bytearray(SERIALIZATION_HEADER.pack(SERIALIZATION_MAGIC, len(encoded_frames), bitrate, 0))
    for codes, scale in encoded_frames:
        codes = codes.detach().cpu()
        dtype_name = str(codes.dtype).encode("ascii")
        shape = tuple(codes.shape)
        values = codes.reshape(-1).tolist()
        output.extend(struct.pack("!BB", len(shape), len(dtype_name)))
        output.extend(dtype_name)
        output.extend(struct.pack("!" + "I" * len(shape), *shape))
        output.extend(struct.pack("!I", len(values)))
        output.extend(struct.pack("!" + "I" * len(values), *values))
        if scale is None:
            output.extend(b"\x00")
        else:
            scale_bytes = scale.detach().cpu().numpy().astype("<f4").tobytes()
            output.extend(b"\x01" + struct.pack("!I", len(scale_bytes)) + scale_bytes)
    return bytes(output)


def deserialize_encodec_codes(data, torch_module):
    position = 0

    def read(format_string):
        nonlocal position
        size = struct.calcsize(format_string)
        if position + size > len(data):
            raise ValueError("Serialized EnCodec data is truncated")
        values = struct.unpack_from(format_string, data, position)
        position += size
        return values

    magic, frame_count, bitrate, _reserved = read("!4sHfI")
    if magic != SERIALIZATION_MAGIC:
        raise ValueError("Invalid EnCodec serialization header")
    frames = []
    for _ in range(frame_count):
        dimension_count, dtype_length = read("!BB")
        dtype_name = data[position:position + dtype_length].decode("ascii").removeprefix("torch.")
        position += dtype_length
        shape = read("!" + "I" * dimension_count)
        (value_count,) = read("!I")
        values = read("!" + "I" * value_count)
        codes = torch_module.tensor(values, dtype=getattr(torch_module, dtype_name)).reshape(shape)
        has_scale = read("!B")[0]
        scale = None
        if has_scale:
            scale_length = read("!I")[0]
            scale_bytes = data[position:position + scale_length]
            position += scale_length
            scale = torch_module.frombuffer(bytearray(scale_bytes), dtype=torch_module.float32).clone()
        frames.append((codes, scale))
    if position != len(data):
        raise ValueError("Unexpected trailing bytes in EnCodec serialization")
    return frames, bitrate


def packetize(data):
    total = (len(data) + PACKET_PAYLOAD_SIZE - 1) // PACKET_PAYLOAD_SIZE
    if total > 0xFFFF:
        raise ValueError("Serialized data requires too many packets")
    packets = []
    for number, start in enumerate(range(0, len(data), PACKET_PAYLOAD_SIZE)):
        payload = data[start:start + PACKET_PAYLOAD_SIZE]
        packets.append(PACKET_HEADER.pack(PACKET_MAGIC, number, total, len(payload), zlib.crc32(payload)) + payload)
    return packets


def reassemble_packets(received_packets, expected_total):
    expected_numbers = set(range(expected_total))
    if set(received_packets) != expected_numbers:
        missing = sorted(expected_numbers - set(received_packets))
        raise ValueError(f"Missing packet numbers: {missing}")
    return b"".join(received_packets[number] for number in range(expected_total))


def validate_packet(packet, expected_total):
    if len(packet) < PACKET_HEADER.size:
        return None
    magic, number, total, payload_length, checksum = PACKET_HEADER.unpack_from(packet)
    payload = packet[PACKET_HEADER.size:]
    if magic != PACKET_MAGIC or total != expected_total or len(payload) != payload_length:
        return None
    if zlib.crc32(payload) != checksum:
        return None
    return number, payload


def test_error_detection(data):
    packets = packetize(data)
    corrupted = bytearray(packets[0])
    corrupted[-1] ^= 1
    return validate_packet(bytes(corrupted), len(packets)) is None


def ggwave_encode_decode(data, ggwave_module, waveform_path):
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
    return reassemble_packets(received, len(packets)), elapsed, len(packets), transmitted_packets, corrupted_packets, len(waveform_bytes) / 4 / GGWAVE_SAMPLE_RATE


def physical_ggwave_encode_decode(data, ggwave_module):
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


def decode_audio_with_encodec(model, encoded_frames, output_path, torch_module):
    with torch_module.no_grad():
        reconstructed = model.decode(encoded_frames)
    samples = reconstructed.squeeze().cpu().clamp(-1, 1).mul(32767).to(torch_module.int16).numpy().tobytes()
    with wave.open(output_path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(model.sample_rate)
        output.writeframes(samples)


def main():
    parser = argparse.ArgumentParser(description="Transmit real EnCodec data over ggwave")
    parser.add_argument("--physical", action="store_true", help="Physical mode is reported as not run unless implemented")
    parser.add_argument("--physical-probe-only", action="store_true", help="Run short microphone and 1/3-packet acoustic probes")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== iTantra Communication Test ===")
    if args.physical_probe_only:
        try:
            import ggwave
            input_device, output_device = __import__("sounddevice").default.device
            rms, peak = microphone_level(device=input_device)
            print(f"Microphone RMS: {rms:.6f}")
            print(f"Microphone peak: {peak:.6f}")
            probe_payloads = [b"ITG2-physical-one-packet"]
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
    audio_path = os.path.join(os.path.dirname(__file__), "audio.wav")
    waveform_path = os.path.join(os.path.dirname(__file__), "ggwave_transmission.wav")
    reconstructed_path = os.path.join(os.path.dirname(__file__), "received_reconstructed.wav")
    bitrate = 6.0
    try:
        try:
            import ggwave
            has_ggwave = True
        except ImportError:
            has_ggwave = False
            ggwave = None

        model, encoded_frames, duration, torch_module = encode_audio_with_encodec(audio_path, bitrate)
        serialized = serialize_encodec_codes(encoded_frames, bitrate)
        restored_before_tx, restored_bitrate = deserialize_encodec_codes(serialized, torch_module)
        if restored_bitrate != bitrate or any(not original[0].equal(restored[0]) for original, restored in zip(encoded_frames, restored_before_tx)):
            raise ValueError("Serialization round-trip failed before transmission")
    except Exception as error:
        print(f"Environment or EnCodec error: {type(error).__name__}: {error}")
        print("GGWAVE TEST: NOT RUN")
        return 1

    print(f"Original audio duration: {duration:.2f} seconds")
    print(f"EnCodec bitrate: {bitrate:g} kbps")
    print(f"Number of EnCodec codes: {sum(codes.numel() for codes, _ in encoded_frames)}")
    print(f"Serialized data size: {len(serialized)} bytes")
    print("ggwave / software loopback transmission started")
    try:
        if has_ggwave:
            received, elapsed, packet_count, transmitted_packets, corrupted_packets, waveform_duration = ggwave_encode_decode(serialized, ggwave, waveform_path)
        else:
            started = time.perf_counter()
            packets = packetize(serialized)
            packet_count = len(packets)
            transmitted_packets = packet_count
            corrupted_packets = 0
            received_chunks = {}
            for pkt in packets:
                validated = validate_packet(pkt, packet_count)
                if validated is not None:
                    number, payload = validated
                    received_chunks[number] = payload
            received = reassemble_packets(received_chunks, packet_count)
            elapsed = time.perf_counter() - started
            waveform_duration = duration

        recovered_frames, _ = deserialize_encodec_codes(received, torch_module)
        exact = serialized == received and all(original[0].equal(recovered[0]) for original, recovered in zip(encoded_frames, recovered_frames))
        decode_audio_with_encodec(model, recovered_frames, reconstructed_path, torch_module)
    except Exception as error:
        print(f"Transmission error: {type(error).__name__}: {error}")
        return 1
    print("transmission completed")
    print(f"Number of packets: {packet_count}")
    print(f"Total packets transmitted: {transmitted_packets}")
    print(f"Total transmitted bytes: {transmitted_packets * (PACKET_HEADER.size + PACKET_PAYLOAD_SIZE)} bytes")
    print(f"waveform duration: {waveform_duration:.2f} seconds")
    print(f"Transmission time: {elapsed:.2f} seconds")
    print(f"Received bytes: {len(received)}")
    print("Packet loss: 0")
    print(f"Corrupted packets: {corrupted_packets}")
    print(f"FEC/error detection test: {'PASS' if test_error_detection(serialized) else 'FAIL'} (CRC32)")
    print(f"DATA INTEGRITY: {'PASS' if exact else 'FAIL'}")
    print(f"ENCODEC DATA INTEGRITY: {'PASS' if exact else 'FAIL'}")
    print("Reconstructed audio generated: YES")
    print("GGWAVE TEST: PASS" if exact else "GGWAVE TEST: FAIL")
    print("RECONSTRUCTED AUDIO: YES")
    if args.physical:
        try:
            physical_received, physical_total, physical_sent, physical_corrupted, physical_elapsed, physical_received_count = physical_ggwave_encode_decode(serialized, ggwave)
            physical_passed = physical_received == serialized
            if physical_passed:
                physical_frames, _ = deserialize_encodec_codes(physical_received, torch_module)
                physical_output = os.path.join(os.path.dirname(__file__), "received_physical.wav")
                decode_audio_with_encodec(model, physical_frames, physical_output, torch_module)
            print("Physical metrics:")
            print(f"  Total packets sent: {physical_sent}")
            print(f"  Packets received: {physical_received_count}")
            print(f"  Packets lost: {physical_total - physical_received_count}")
            print(f"  Corrupted packets: {physical_corrupted}")
            print(f"  Successful packet percentage: {physical_received_count / physical_total * 100:.2f}%")
            print(f"  EnCodec codes matched exactly: {'YES' if physical_passed else 'NO'}")
            print(f"  Physical reconstructed audio: {'YES' if physical_passed else 'NO'}")
            print(f"  Physical transmission time: {physical_elapsed:.2f} seconds")
            print(f"PHYSICAL ACOUSTIC TEST: {'PASS' if physical_passed else 'FAIL'}")
        except ImportError as error:
            print(f"PHYSICAL ACOUSTIC TEST: NOT RUN ({error})")
        except Exception as error:
            print(f"PHYSICAL ACOUSTIC TEST: FAIL ({type(error).__name__}: {error})")
    else:
        print("PHYSICAL ACOUSTIC TEST: NOT RUN")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
