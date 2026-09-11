import argparse
import socket
import struct
import sys
import zlib

from test_ggwave import decode_audio_with_encodec, deserialize_encodec_codes, encode_audio_with_encodec, serialize_encodec_codes


MAGIC = b"ITN1"
HEADER = struct.Struct("!4sQI")
MAX_PAYLOAD = 64 * 1024 * 1024


def send_payload(connection, payload):
    connection.sendall(HEADER.pack(MAGIC, len(payload), zlib.crc32(payload)) + payload)


def receive_payload(connection):
    magic, payload_size, checksum = HEADER.unpack(read_exact(connection, HEADER.size))
    if magic != MAGIC:
        raise ValueError("Invalid iTantra internet transport header")
    if payload_size > MAX_PAYLOAD:
        raise ValueError(f"Payload is too large: {payload_size} bytes")
    payload = read_exact(connection, payload_size)
    if zlib.crc32(payload) != checksum:
        raise ValueError("Internet transport CRC32 check failed")
    return payload


def read_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Connection closed before the complete payload arrived")
        chunks.extend(chunk)
    return bytes(chunks)


def send_encodec(host, port, audio_path, bitrate):
    model, frames, duration, _torch_module = encode_audio_with_encodec(audio_path, bitrate)
    del model
    payload = serialize_encodec_codes(frames, bitrate)
    with socket.create_connection((host, port), timeout=30) as connection:
        send_payload(connection, payload)
        if read_exact(connection, 4) != b"OKAY":
            raise RuntimeError("Receiver rejected the payload")
    print(f"Sent EnCodec data: {len(payload)} bytes")
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"EnCodec bitrate: {bitrate:g} kbps")
    print("Internet transmission: PASS")


def receive_encodec(bind_host, port, output_path):
    import torch
    from encodec import EncodecModel

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, port))
        server.listen(1)
        print(f"Listening on {bind_host}:{port}")
        connection, address = server.accept()
        print(f"Connected: {address[0]}:{address[1]}")
        with connection:
            payload = receive_payload(connection)
            frames, bitrate = deserialize_encodec_codes(payload, torch)
            model = EncodecModel.encodec_model_24khz()
            model.eval()
            decode_audio_with_encodec(model, frames, output_path, torch)
            connection.sendall(b"OKAY")
    print(f"Received EnCodec data: {len(payload)} bytes")
    print(f"EnCodec bitrate: {bitrate:g} kbps")
    print(f"EnCodec codes: {sum(codes.numel() for codes, _ in frames)}")
    print(f"Reconstructed audio: {output_path}")
    print("Internet reception: PASS")


def self_test():
    payload = b"internet-transport-integrity" * 20
    left, right = socket.socketpair()
    try:
        send_payload(left, payload)
        recovered = receive_payload(right)
    finally:
        left.close()
        right.close()
    if recovered != payload:
        raise AssertionError("Self-test payload mismatch")
    print("Internet framing self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description="Reliable internet transport for iTantra EnCodec data")
    parser.add_argument("--send", metavar="HOST", help="Send audio.wav to a receiver")
    parser.add_argument("--receive", metavar="HOST", help="Listen on HOST for one transmission")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--audio", default="audio.wav")
    parser.add_argument("--output", default="received_internet.wav")
    parser.add_argument("--bitrate", type=float, default=6.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.self_test:
        self_test()
        return 0
    if bool(args.send) == bool(args.receive):
        parser.error("choose exactly one of --send or --receive, or use --self-test")
    if args.send:
        send_encodec(args.send, args.port, args.audio, args.bitrate)
    else:
        receive_encodec(args.receive, args.port, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
