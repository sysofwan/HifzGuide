"""Decode streamed Tadabur audio to 16 kHz mono — the model's expected input.

Tadabur clips arrive as raw WAV bytes when the ``datasets`` audio feature is read
with ``decode=False`` (which also avoids the ``torchcodec`` runtime dependency that
``datasets``' built-in decoder pulls in). This module decodes those bytes with
``soundfile`` and resamples/downmixes to 16 kHz mono with ``librosa`` so the
waveform matches what the ``SeamlessM4TFeatureExtractor`` — and thus the model —
expects. Keeping this as the single loader means the Phase 3 filter reuses the exact
same 16 kHz-mono preprocessing as this smoke test.

It is also where the sealed panel's seal sits for audio (:mod:`tadabur.panel_seal`): every
audio file the tools read goes through :func:`read_audio` or :func:`read_audio_bytes`,
and every byte buffer through :func:`decode_to_mono_16k`, each of which refuses panel
audio outside an authorized block.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .panel_seal import SealedPanelError, refuse_sealed, refuse_sealed_bytes

# The Muaalem feature extractor / model operate at 16 kHz; resample everything here.
TARGET_SAMPLE_RATE = 16000


def read_audio(path: Path, **kwargs):
    """``soundfile.read`` of an audio file, after the panel seal's check. The tools read
    audio files only through here and :func:`read_audio_bytes`."""
    return sf.read(str(refuse_sealed(Path(path))), **kwargs)


def read_audio_bytes(path: Path) -> bytes:
    """An audio file's raw bytes, after the panel seal's check (for
    :func:`decode_to_mono_16k`)."""
    return refuse_sealed(Path(path)).read_bytes()


def decode_to_mono_16k(raw_audio: bytes) -> np.ndarray:
    """Decode WAV ``raw_audio`` to a 16 kHz mono float32 waveform.

    Downmixes multi-channel audio by averaging channels and resamples to
    ``TARGET_SAMPLE_RATE`` only when the source rate differs, so already-16 kHz mono
    clips pass through unresampled.
    """
    return decode_to_mono_16k_with_rate(raw_audio)[0]


def decode_to_mono_16k_with_rate(raw_audio: bytes) -> tuple[np.ndarray, int]:
    """:func:`decode_to_mono_16k` plus the source's sample rate.

    The pass-1 tlog facts record the stored audio's own rate beside the 16 kHz duration,
    so the decode returns both instead of dropping the rate.
    """
    return decode_to_mono_16k_with_rate_and_decoder(raw_audio)[:2]


def decode_to_mono_16k_with_rate_and_decoder(
        raw_audio: bytes) -> tuple[np.ndarray, int, str]:
    """``decode_to_mono_16k_with_rate`` and which decoder read the bytes.

    ``soundfile``/libsndfile covers WAV, FLAC and the other native containers. When it refuses
    the container — tlog's unclean split ships MP4/AAC behind ``.wav`` paths — the pinned PyAV
    fallback demuxes the same bytes; its wheel bundles ffmpeg, so no system ffmpeg is needed.
    Both decoders end in :func:`_to_mono_16k`, so they differ in demuxing, not in resampling.
    """
    data = refuse_sealed_bytes(raw_audio)
    try:
        return (*_decode_with_soundfile(data), "soundfile")
    except (sf.LibsndfileError, OSError, ValueError) as soundfile_error:
        try:
            return (*_decode_with_pyav(data), "pyav")
        except SealedPanelError:
            raise
        except Exception as pyav_error:  # noqa: BLE001 - undecodable bytes are a row fact
            raise ValueError(
                f"no decoder read these bytes: soundfile: {soundfile_error}; "
                f"pyav: {pyav_error!r}") from pyav_error


def decoder_versions() -> dict[str, str | None]:
    """The installed versions of the decoders pass 1 records: soundfile, libsndfile and PyAV.

    PyAV is ``None`` when it is absent; the tlog pass refuses to run in that state
    (:func:`require_pyav`) rather than quietly counting MP4 rows as undecodable.
    """
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {"soundfile": installed("soundfile"), "libsndfile": sf.__libsndfile_version__,
            "pyav": installed("av")}


def require_pyav() -> str:
    """The installed PyAV version, or an error: the tlog pass needs the fallback for MP4/AAC."""
    installed = decoder_versions()["pyav"]
    if installed is None:
        raise RuntimeError(
            "PyAV is not installed; the tlog pass needs it for the unclean split's MP4/AAC rows "
            "(pip install -r tools/requirements-train.txt)")
    return installed


def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    return _to_mono_16k(waveform, sample_rate), sample_rate


def _decode_with_pyav(data: bytes) -> tuple[np.ndarray, int]:
    """PyAV's decode of a container libsndfile refuses, at the source's own rate."""
    import av

    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("no audio stream")
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        if sample_rate is None:
            raise ValueError("no sample rate")
        resampler = av.AudioResampler(format="fltp", layout="mono")
        chunks = [_flatten(resampled)
                  for frame in container.decode(stream)
                  for resampled in resampler.resample(frame)]
        chunks += [_flatten(resampled) for resampled in resampler.resample(None)]
    waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return _to_mono_16k(waveform, sample_rate), sample_rate


def _flatten(frame) -> np.ndarray:
    return frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)


def _to_mono_16k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
    return np.ascontiguousarray(waveform, dtype=np.float32)
