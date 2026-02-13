"""Audio codec processor — normalize and resample to canonical PCM16 mono 16kHz.

LiveKit commonly operates at 48kHz; the STT path needs 16kHz mono PCM16.
This processor handles the conversion.
"""

from __future__ import annotations

import struct

from voice_agent.logging import get_logger

logger = get_logger("audio_codec")

# Canonical format for STT path
CANONICAL_SAMPLE_RATE = 16000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


def resample_linear(
    pcm16_bytes: bytes,
    src_rate: int,
    dst_rate: int,
) -> bytes:
    """Simple linear-interpolation resampler for PCM16 mono audio.

    Good enough for speech.  For production, consider a higher-quality
    resampler like libsamplerate or scipy.signal.resample_poly.
    """
    if src_rate == dst_rate:
        return pcm16_bytes

    # Unpack to int16 samples
    n_samples = len(pcm16_bytes) // 2
    if n_samples == 0:
        return b""

    samples = struct.unpack(f"<{n_samples}h", pcm16_bytes)

    ratio = dst_rate / src_rate
    out_len = int(n_samples * ratio)
    if out_len == 0:
        return b""

    out_samples: list[int] = []
    for i in range(out_len):
        src_pos = i / ratio
        idx = int(src_pos)
        frac = src_pos - idx

        s0 = samples[min(idx, n_samples - 1)]
        s1 = samples[min(idx + 1, n_samples - 1)]
        val = int(s0 + frac * (s1 - s0))
        # Clamp to int16 range
        val = max(-32768, min(32767, val))
        out_samples.append(val)

    return struct.pack(f"<{len(out_samples)}h", *out_samples)


def stereo_to_mono(pcm16_bytes: bytes) -> bytes:
    """Convert stereo PCM16 to mono by averaging channels."""
    n_samples = len(pcm16_bytes) // 4  # 2 channels × 2 bytes
    if n_samples == 0:
        return b""

    stereo = struct.unpack(f"<{n_samples * 2}h", pcm16_bytes)
    mono: list[int] = []
    for i in range(0, len(stereo), 2):
        avg = (stereo[i] + stereo[i + 1]) // 2
        mono.append(avg)

    return struct.pack(f"<{len(mono)}h", *mono)


def normalize_audio(
    pcm16_bytes: bytes,
    src_rate: int,
    src_channels: int,
    dst_rate: int = CANONICAL_SAMPLE_RATE,
    dst_channels: int = CANONICAL_CHANNELS,
) -> bytes:
    """Normalize audio to canonical format (PCM16, mono, 16kHz).

    Args:
        pcm16_bytes: Raw PCM16 audio bytes.
        src_rate: Source sample rate.
        src_channels: Source channel count.
        dst_rate: Target sample rate (default 16000).
        dst_channels: Target channels (default 1).

    Returns:
        Normalized PCM16 bytes.
    """
    data = pcm16_bytes

    # 1. Convert to mono if needed
    if src_channels == 2 and dst_channels == 1:
        data = stereo_to_mono(data)
    elif src_channels > 2:
        logger.warning("Unsupported channel count %d, taking first channel", src_channels)
        n_samples = len(data) // (src_channels * 2)
        all_samples = struct.unpack(f"<{n_samples * src_channels}h", data)
        mono = [all_samples[i * src_channels] for i in range(n_samples)]
        data = struct.pack(f"<{len(mono)}h", *mono)

    # 2. Resample if needed
    if src_rate != dst_rate:
        data = resample_linear(data, src_rate, dst_rate)

    return data


def split_into_frames(
    pcm16_bytes: bytes,
    sample_rate: int,
    frame_duration_ms: int = 20,
) -> list[bytes]:
    """Split PCM16 audio into fixed-duration frames.

    Args:
        pcm16_bytes: Raw PCM16 mono audio.
        sample_rate: Sample rate of the audio.
        frame_duration_ms: Duration of each frame in milliseconds.

    Returns:
        List of PCM16 byte chunks.
    """
    frame_samples = (sample_rate * frame_duration_ms) // 1000
    frame_bytes = frame_samples * 2  # 16-bit

    frames: list[bytes] = []
    for offset in range(0, len(pcm16_bytes), frame_bytes):
        chunk = pcm16_bytes[offset : offset + frame_bytes]
        if len(chunk) == frame_bytes:
            frames.append(chunk)
        elif chunk:
            # Pad last frame with silence
            chunk += b"\x00" * (frame_bytes - len(chunk))
            frames.append(chunk)

    return frames
