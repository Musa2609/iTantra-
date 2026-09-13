"""
iTantra Binary Packet Protocol & Error Detection / Recovery (Python Port)
Parity with Android PacketProtocol.kt, Crc16.kt, and FecEngine.kt.
"""

import struct
from enum import IntEnum
from typing import Optional, Tuple, List, Union
from semantic_engine import SemanticPayload

MAGIC_BYTE = 0x49  # ASCII 'I'

class PacketType(IntEnum):
    VOICE_OPUS = 0       # Mode 1: Voice / Audio Codec (Opus frames)
    MODE_1_SEMANTIC = 1  # Mode 2: Semantic Intent (10-bit compact frame)
    MODE_2_TEXT = 2      # Mode 2: Free Text (UTF-8)
    MODE_3_ENCODEC = 3   # Legacy neural codec

# Mode aliases for clarity
MODE_1_VOICE_OPUS = PacketType.VOICE_OPUS
MODE_2_SEMANTIC = PacketType.MODE_1_SEMANTIC

class Crc32:
    """CRC32 (IEEE 802.3) for error detection."""
    @staticmethod
    def compute(data: bytes) -> int:
        import zlib
        return zlib.crc32(data) & 0xFFFFFFFF

class Crc16:
    """CRC16-CCITT (Polynomial 0x1021, Initial 0xFFFF)"""
    @staticmethod
    def compute(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

class PacketHeader:
    def __init__(self, packet_type: PacketType, lang_id: int = 0, is_fec: bool = False, sequence_num: int = 0):
        self.packet_type = packet_type
        self.lang_id = lang_id & 0x07
        self.is_fec = is_fec
        self.sequence_num = sequence_num & 0xFFFF

    def encode(self) -> bytes:
        ctrl = (self.packet_type & 0x03) | ((self.lang_id & 0x07) << 2) | ((1 if self.is_fec else 0) << 5)
        return struct.pack("!BBH", MAGIC_BYTE, ctrl, self.sequence_num)

    @classmethod
    def decode(cls, data: bytes) -> Tuple["PacketHeader", int]:
        if len(data) < 4:
            raise ValueError("Header too short")
        magic, ctrl, seq = struct.unpack_from("!BBH", data, 0)
        if magic != MAGIC_BYTE:
            raise ValueError(f"Invalid magic byte: {hex(magic)}, expected {hex(MAGIC_BYTE)}")
        ptype = PacketType(ctrl & 0x03)
        lang_id = (ctrl >> 2) & 0x07
        is_fec = bool((ctrl >> 5) & 0x01)
        return cls(packet_type=ptype, lang_id=lang_id, is_fec=is_fec, sequence_num=seq), 4

class Packet:
    def __init__(self, header: PacketHeader, payload: bytes):
        self.header = header
        self.payload = payload

    def encode(self) -> bytes:
        header_bytes = self.header.encode()
        body = header_bytes + struct.pack("!H", len(self.payload)) + self.payload
        crc = Crc16.compute(body)
        return body + struct.pack("!H", crc)

    @classmethod
    def decode(cls, data: bytes) -> Optional["Packet"]:
        if len(data) < 8:  # 4 header + 2 length + 2 crc
            return None

        # Verify CRC16
        body = data[:-2]
        expected_crc = struct.unpack("!H", data[-2:])[0]
        actual_crc = Crc16.compute(body)
        if actual_crc != expected_crc:
            print(f"[PacketProtocol] CRC Check FAILED: expected {hex(expected_crc)}, got {hex(actual_crc)}")
            return None

        header, offset = PacketHeader.decode(data)
        payload_len = struct.unpack_from("!H", data, offset)[0]
        offset += 2
        payload = data[offset:offset + payload_len]
        return cls(header=header, payload=payload)

PacketProtocol = Packet

class FecEngine:
    """
    Systematic XOR-Parity FEC Engine
    Group Size = 3 (2 Data Frames + 1 Parity Frame)
    Recovers 1 dropped or CRC-corrupted packet per group.
    """
    @staticmethod
    def generate_parity(packet1: bytes, packet2: bytes) -> bytes:
        max_len = max(len(packet1), len(packet2))
        p1 = packet1.ljust(max_len, b"\x00")
        p2 = packet2.ljust(max_len, b"\x00")
        return bytes(b1 ^ b2 for b1, b2 in zip(p1, p2))

    @staticmethod
    def recover_missing(known_packet: bytes, parity_packet: bytes, original_len: int) -> bytes:
        max_len = max(len(known_packet), len(parity_packet))
        k = known_packet.ljust(max_len, b"\x00")
        p = parity_packet.ljust(max_len, b"\x00")
        recovered = bytes(b1 ^ b2 for b1, b2 in zip(k, p))
        return recovered[:original_len]

def chunk_payload(payload: bytes, packet_type: PacketType, chunk_size: int = 120, lang_id: int = 0) -> List[Packet]:
    """
    Split arbitrary payload (e.g. Opus bitstream or text) into sequence-numbered Packets
    that fit comfortably within the 140-byte acoustic modem MTU.
    Each packet contains: [chunk_index: 2B, total_chunks: 2B, chunk_data].
    """
    if len(payload) == 0:
        return []
    total_chunks = (len(payload) + chunk_size - 1) // chunk_size
    packets = []
    for idx in range(total_chunks):
        chunk_data = payload[idx * chunk_size : (idx + 1) * chunk_size]
        framed_payload = struct.pack("!HH", idx, total_chunks) + chunk_data
        header = PacketHeader(
            packet_type=packet_type,
            lang_id=lang_id,
            is_fec=False,
            sequence_num=idx
        )
        packets.append(Packet(header=header, payload=framed_payload))
    return packets

def reassemble_payload(packets: List[Packet]) -> Tuple[Optional[bytes], int, int]:
    """
    Reassemble received Packets into complete binary payload.
    Returns: (reassembled_bytes, valid_count, missing_count).
    """
    chunks = {}
    total_expected = None
    valid_count = 0
    
    for pkt in packets:
        if pkt is None or len(pkt.payload) < 4:
            continue
        idx, total = struct.unpack_from("!HH", pkt.payload, 0)
        chunk_data = pkt.payload[4:]
        chunks[idx] = chunk_data
        total_expected = total
        valid_count += 1
        
    if total_expected is None or len(chunks) == 0:
        return None, 0, 0
        
    missing_count = total_expected - len(chunks)
    if missing_count > 0:
        return None, valid_count, missing_count

    reassembled = bytearray()
    for i in range(total_expected):
        if i in chunks:
            reassembled.extend(chunks[i])
        else:
            return None, valid_count, missing_count
            
    return bytes(reassembled), valid_count, 0

