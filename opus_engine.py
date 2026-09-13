"""
iTantra Opus Audio Codec Engine
Provides OpusEncoder and OpusDecoder for Mode 1 Voice Communication.
Uses libsndfile (RFC 6716 compliant Opus implementation) to produce real compressed Opus bitstreams and reconstruct PCM speech.
"""

import io
import os
import struct
import numpy as np
import soundfile as sf
from typing import Tuple, List, Optional

# Supported Opus standard sample rates (RFC 6716)
SUPPORTED_SAMPLE_RATES = [8000, 12000, 16000, 24000, 48000]

class OpusCodecError(Exception):
    pass

class OpusEncoder:
    """
    Real Opus Audio Encoder.
    Compresses PCM audio into compliant Opus frames/bitstream.
    Configurable parameters: sample_rate, channels, compression_level.
    """
    def __init__(self, sample_rate: int = 24000, channels: int = 1, compression_level: int = 10):
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise OpusCodecError(
                f"Unsupported sample rate {sample_rate} Hz. Opus supports: {SUPPORTED_SAMPLE_RATES}"
            )
        if channels not in [1, 2]:
            raise OpusCodecError(f"Unsupported channels {channels}. Opus supports mono (1) or stereo (2).")

        self.sample_rate = sample_rate
        self.channels = channels
        self.compression_level = max(0, min(10, compression_level))

    def encode(self, pcm_data: np.ndarray, orig_sr: Optional[int] = None) -> bytes:
        """
        Encode raw float or int16 PCM audio samples into an Opus bitstream.
        Returns bytes representing the compressed Opus data.
        """
        if pcm_data is None or len(pcm_data) == 0:
            raise OpusCodecError("Cannot encode empty audio buffer")

        # Convert to 1D float32 array if needed
        samples = np.asarray(pcm_data, dtype=np.float32)
        if samples.ndim > 1:
            if self.channels == 1:
                samples = np.mean(samples, axis=1 if samples.shape[1] < samples.shape[0] else 0)
            elif samples.shape[1] != self.channels:
                samples = samples[:, :self.channels]

        # Resample to encoder sample_rate if needed
        if orig_sr is not None and orig_sr != self.sample_rate:
            in_len = len(samples)
            out_len = int(round(in_len * self.sample_rate / orig_sr))
            samples = np.interp(
                np.linspace(0, in_len, out_len, endpoint=False),
                np.arange(in_len),
                samples
            ).astype(np.float32)

        # Normalize clipping bounds [-1.0, 1.0]
        max_val = np.max(np.abs(samples)) if len(samples) > 0 else 0.0
        if max_val > 1.0:
            samples = samples / max_val

        # Encode via soundfile OGG/Opus container
        bio = io.BytesIO()
        try:
            with sf.SoundFile(
                bio,
                mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                format='OGG',
                subtype='OPUS'
            ) as sf_out:
                sf_out.write(samples)
        except Exception as e:
            raise OpusCodecError(f"Opus encoding failed: {str(e)}")

        return bio.getvalue()


class OpusDecoder:
    """
    Real Opus Audio Decoder.
    Decodes compressed Opus bitstream/frames into reconstructed PCM audio samples.
    """
    def __init__(self, target_sample_rate: Optional[int] = None):
        self.target_sample_rate = target_sample_rate

    def decode(self, opus_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Decode Opus bytes into (pcm_samples, sample_rate).
        Returns float32 PCM samples in range [-1.0, 1.0] and sample rate.
        """
        if not opus_bytes or len(opus_bytes) == 0:
            raise OpusCodecError("Cannot decode empty Opus payload")

        bio = io.BytesIO(opus_bytes)
        try:
            data, sr = sf.read(bio, dtype='float32')
        except Exception as e:
            raise OpusCodecError(f"Opus decoding failed: {str(e)}")

        # Resample if target sample rate requested
        if self.target_sample_rate is not None and self.target_sample_rate != sr:
            in_len = len(data)
            out_len = int(round(in_len * self.target_sample_rate / sr))
            data = np.interp(
                np.linspace(0, in_len, out_len, endpoint=False),
                np.arange(in_len),
                data
            ).astype(np.float32)
            sr = self.target_sample_rate

        return data, sr


def get_opus_compression_stats(pcm_samples: np.ndarray, sr: int, opus_bytes: bytes) -> dict:
    """
    Calculate and return exact, un-invented compression metrics.
    """
    raw_pcm_bytes = len(pcm_samples) * 2  # 16-bit PCM equivalent
    opus_size = len(opus_bytes)
    ratio = raw_pcm_bytes / max(1, opus_size)
    duration_sec = len(pcm_samples) / sr if sr > 0 else 0.0
    bitrate_kbps = (opus_size * 8.0 / duration_sec / 1000.0) if duration_sec > 0 else 0.0

    return {
        "raw_pcm_bytes": raw_pcm_bytes,
        "opus_bytes": opus_size,
        "compression_ratio": round(ratio, 2),
        "duration_sec": round(duration_sec, 2),
        "measured_bitrate_kbps": round(bitrate_kbps, 2),
        "sample_rate": sr
    }
