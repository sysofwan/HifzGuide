"""#267 steps 1-2's gates: each a refusal with a message, never a judgement.

* **Identity:** the model revision and the full 64-hex ``.nemo`` SHA-256 are pinned. The issue
  quotes ``d29d19d7…2e31``; only the full hash is a match, so a file that merely resembles the
  quote is refused.
* **Mapping:** at least 95% of the sites and edits are located (step 2); below that the run
  stops and reports the unlocated items.
* **Parity (only if ONNX is used):** the greedy CTC text from onnxruntime must equal NeMo's on
  at least 49 of at least 50 Ship A clips. A broken decoder must never reach the comparison.
* **Card sanity:** the CTC WER on a 200-clip EveryAyah sample is reported, and the run stops
  only above twice the card's 7.61, which would mean a wrong head, sample rate or normaliser.
"""

from __future__ import annotations

#: The pinned Hub revision (``nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0``, 2026-11-04).
REVISION = "7f32349d952f42a28dce979ba73270aa2bbdfa89"
#: The ``.nemo`` file's SHA-256, full: d29d19d7c054a5fc010ac6815e9cbb0dd1b21a30e0a7f7f2982e1fecaf0c3e31.
NEMO_SHA256 = "d29d19d7c054a5fc010ac6815e9cbb0dd1b21a30e0a7f7f2982e1fecaf0c3e31"

#: The mapping gate's bar (step 2).
MAPPING_MINIMUM = 0.95
#: The parity gate's bars (step 1; ONNX only).
PARITY_CLIPS = 50
PARITY_EQUAL = 49
#: The card's Arabic CTC WER and the sample the sanity line is read on.
CARD_WER = 7.61
CARD_CLIPS = 200


class ScreenStop(RuntimeError):
    """A pre-registered gate failed: the run stops and reports, it does not judge."""


def require_identity(record: dict) -> None:
    """Refuse a model record that is not the pinned revision and full ``.nemo`` hash."""
    if record.get("revision") != REVISION:
        raise ScreenStop(
            f"identity: revision {record.get('revision')!r} is not the pinned {REVISION}")
    if record.get("nemo_sha256") != NEMO_SHA256:
        raise ScreenStop(
            f"identity: nemo_sha256 {record.get('nemo_sha256')!r} is not the pinned "
            f"{NEMO_SHA256}; the issue's quoted prefix and suffix are not a match")


def mapping_gate(located: int, total: int) -> None:
    """Pass at >= 95% located; otherwise stop with the counts."""
    if total <= 0:
        raise ScreenStop(f"mapping gate: nothing to locate ({located}/{total})")
    share = located / total
    if share < MAPPING_MINIMUM:
        raise ScreenStop(
            f"mapping gate: {located}/{total} located ({100 * share:.1f}%) is below "
            f"{100 * MAPPING_MINIMUM:.0f}%; the unlocated items are listed and the run stops")


def parity_gate(equal: int, clips: int) -> None:
    """Pass on at least 50 clips with at least 49 equal texts; otherwise stop."""
    if clips < PARITY_CLIPS or equal < PARITY_EQUAL:
        raise ScreenStop(
            f"parity gate: {equal}/{clips} clips equal; the gate needs at least "
            f"{PARITY_EQUAL}/{PARITY_CLIPS}")


def card_gate(wer: float, clips: int) -> None:
    """Report the card sanity line; stop only above twice the card's WER on its 200 clips."""
    if clips != CARD_CLIPS:
        raise ScreenStop(f"card sanity: the line is read on {CARD_CLIPS} clips, not {clips}")
    if wer > 2 * CARD_WER:
        raise ScreenStop(
            f"card sanity: CTC WER {wer:.2f} is above twice the card's {CARD_WER} "
            f"({2 * CARD_WER:.2f}); the head, sample rate or normaliser is wrong")
