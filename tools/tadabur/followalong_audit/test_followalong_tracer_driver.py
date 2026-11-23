"""The app-arm driver ``tools/muraja_harness/transcribe_checks.swift`` (#307).

It runs Muraja's offline transcriber (the scheduler over the app's own ``MuaalemInference``) and
writes every update's **full** hop and overlap as check requests, where Muraja's ``.log``
(``TranscribeLog.swift:43-46``) cuts each overlap to 30 characters. These tests drive the built
binary with a fake model (``--fake-ids``: every inference returns those phoneme ids), so they need
the CoreML-free half of the build only. They skip where no harness is built (no Mac build), and
fail where one is built without the driver: ``build.sh`` records it in ``harness.json`` as
``transcribe_checks_binary``.

Usage the tests pin:  ``transcribe_checks [--fake-ids <i,j,...>] [--fake-fail] <16 kHz mono wav>``
prints one JSON object on stdout: ``{"checks": [...], "inference_latency_ms": <number>, ...}``,
each check ``{"hop", "overlap", "flush"}`` plus ``"stop": true`` on the session's last. The latency
is ``TRANSCRIBE_INFERENCE_LATENCY_MS`` (0 when unset). A clip that cannot be transcribed exits
non-zero and says why on stderr.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from pathlib import Path

import pytest

from training.decoding import clusters
from training.muraja_policy import CURRENT_COMMIT, HARNESS_ROOT

MANIFEST = HARNESS_ROOT / CURRENT_COMMIT / "harness.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="the Muraja harness is not built here")

#: ids 2, 32, 3, 34 decode to ب + fatha, ت + kasra: 40 Characters (80 scalars) from 80 ids.
FAKE_IDS = ",".join(["2", "32", "3", "34"] * 20)
FAKE_TEXT = "بَتِ" * 20


@pytest.fixture(scope="module")
def driver() -> Path:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert "transcribe_checks_binary" in record, \
        f"{MANIFEST} has no transcribe_checks_binary: rebuild with tools/muraja_harness/build.sh"
    binary = Path(record["transcribe_checks_binary"])
    assert binary.is_file(), f"{binary} is missing"
    return binary


@pytest.fixture(scope="module")
def wav(tmp_path_factory) -> Path:
    """Twelve seconds of a loud tone: never silent to the transcriber's speech-energy proxy."""
    path = tmp_path_factory.mktemp("driver") / "tone.wav"
    samples = b"".join(int(0.3 * 32767 * math.sin(i * 0.05)).to_bytes(2, "little", signed=True)
                       for i in range(12 * 16000))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(samples)
    return path


def run(driver: Path, wav: Path, *flags: str, latency: str | None = None):
    env = {key: value for key, value in os.environ.items() if key != "TRANSCRIBE_INFERENCE_LATENCY_MS"}
    if latency is not None:
        env["TRANSCRIBE_INFERENCE_LATENCY_MS"] = latency
    return subprocess.run([str(driver), *flags, str(wav)], capture_output=True, env=env, check=False)


def checks_of(driver: Path, wav: Path, **kwargs) -> list[dict]:
    done = run(driver, wav, "--fake-ids", FAKE_IDS, **kwargs)
    assert done.returncode == 0, done.stderr.decode("utf-8")
    return json.loads(done.stdout)["checks"]


def test_the_fixture_is_longer_than_the_thirty_character_clip():
    assert len(clusters(FAKE_TEXT)) == 40 > 30


def test_every_window_and_preview_overlap_survives_in_full(driver, wav):
    checks = checks_of(driver, wav)
    windows = [c for c in checks if c["hop"] and not c["flush"] and not c.get("stop")]
    previews = [c for c in checks if not c["hop"] and not c["flush"]]
    assert len(windows) >= 2 and len(previews) >= 5
    assert all(c["hop"] == FAKE_TEXT and c["overlap"] == FAKE_TEXT for c in windows)
    assert all(c["overlap"] == FAKE_TEXT for c in previews)
    assert not any(0 < len(clusters(c["overlap"])) <= 30 for c in checks)


def test_the_session_ends_with_the_stop_carrying_the_final_phonemes(driver, wav):
    last = checks_of(driver, wav)[-1]
    assert last == {"hop": FAKE_TEXT, "overlap": "", "flush": False, "stop": True}


def test_a_check_holds_only_the_four_check_fields(driver, wav):
    assert all(set(c) <= {"hop", "overlap", "flush", "stop"} for c in checks_of(driver, wav))


def test_the_driver_reads_the_inference_latency_itself(driver, wav):
    for latency, expected in ((None, 0), ("40", 40)):
        done = run(driver, wav, "--fake-ids", FAKE_IDS, latency=latency)
        assert json.loads(done.stdout)["inference_latency_ms"] == expected


def test_two_runs_give_byte_identical_check_json(driver, wav):
    first = run(driver, wav, "--fake-ids", FAKE_IDS, latency="40")
    second = run(driver, wav, "--fake-ids", FAKE_IDS, latency="40")
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_a_clip_whose_inference_fails_exits_non_zero_and_says_why(driver, wav):
    done = run(driver, wav, "--fake-fail")
    assert done.returncode != 0
    assert "fail" in done.stderr.decode("utf-8").lower()
