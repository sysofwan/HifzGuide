"""#267 step 1's card sanity line: CTC WER on a 200-clip EveryAyah sample.

The card reports EveryAyah WER 6.55 (RNNT) and 7.61 (CTC). This line exists to catch a wrong
head, sample rate or normaliser, not to grade the model: it is reported, and the run stops only
above twice the CTC figure (:func:`tadabur.fastconformer_screen.gates.card_gate`).

The sample is the Quran-Lab benchmark's ``everyayah_heldout`` source — the first
:data:`SAMPLE` rows by key, so the line is reproducible without a random seed — scored against
the benchmark's own reference text through :func:`training.zipformer_gate.card_norm`, the
card's normalisation (NFC, its diacritic/tatweel strip and letter folding). Word error is
pooled over reference words, never averaged over clips.

Two caveats are reported with the line and not hidden: the benchmark's studio reciters are not
NVIDIA's test split, and the card does not publish its own text normalisation, so
:func:`card_norm` is the closest documented rule (it is the one the card's phoneme table uses).

The audio is decoded by the isolated NeMo environment, so this module imports the model only in
:func:`measure`; the normalisation and the WER are torch-free and tested without it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from training.zipformer_gate import card_norm, edit_counts

#: The sample's size and source, fixed before any decode (step 1's "200-clip sample").
SAMPLE = 200
SOURCE = "everyayah_heldout"
#: What the line is measured against, written into its record.
NORMALISATION = ("training.zipformer_gate.card_norm: NFC, diacritics and tatweel stripped, "
                 "hamza/alif/ya/ta-marbuta spellings folded, whitespace collapsed")


def words(text: str) -> list[str]:
    """A text as the card reads it: :func:`card_norm`'s key, split into words."""
    return card_norm(text).split()


def wer(pairs: Iterable[tuple[str, str]]) -> dict:
    """Word error rate, pooled over reference words, for ``(reference, hypothesis)`` pairs."""
    clips = errors = reference_words = 0
    for reference, hypothesis in pairs:
        reference_words_here = words(reference)
        if not reference_words_here:
            continue
        edits, length = edit_counts(reference_words_here, words(hypothesis))
        clips += 1
        errors += edits
        reference_words += length
    return {"clips": clips, "errors": errors, "reference_words": reference_words,
            "wer": 100.0 * errors / reference_words if reference_words else None}


def sample_rows(manifest: Path, sample: int = SAMPLE) -> list[dict]:
    """The sample's rows: the manifest's EveryAyah source, the first ``sample`` by key."""
    rows = {}
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") == SOURCE:
            rows[row["id"]] = row
    chosen = [rows[key] for key in sorted(rows)][:sample]
    if len(chosen) < sample:
        from tadabur.fastconformer_screen.gates import ScreenStop

        raise ScreenStop(f"card sanity: {manifest} holds {len(chosen)} {SOURCE} clips, "
                         f"not the {sample} the line is read on")
    return chosen


def measure(manifest: Path, decode: Callable[[Path], str], *, sample: int = SAMPLE) -> dict:
    """Decode the sample's clips and score them: the line's numbers, without the gate."""
    rows = sample_rows(manifest, sample)
    texts = {row["id"]: decode(Path(manifest).parent / row["audio"]) for row in rows}
    record = wer((row["reference_text"], texts[row["id"]]) for row in rows)
    record.update({"source": SOURCE, "sample": sample, "normalisation": NORMALISATION,
                   "clips": [row["id"] for row in rows]})
    return record


def record(measured: dict, *, identity: dict) -> dict:
    """The committed line: the measurement, the model identity and the gate's outcome."""
    from tadabur.fastconformer_screen.gates import CARD_CLIPS, CARD_WER

    passed = measured["clips"] == CARD_CLIPS and measured["wer"] <= 2 * CARD_WER
    return {"issue": 267, "identity": identity, "card": {"wer": CARD_WER, "clips": CARD_CLIPS},
            "measured": measured, "gate_failed": not passed,
            "rule": "reported; the run stops only above twice the card's CTC WER"}


def main(argv: Sequence[str] | None = None) -> None:
    """Measure the line on the box and write its record; a failed gate stops the run."""
    import argparse

    from tadabur.fastconformer_screen.decode import load_ctc_model, transcribe
    from tadabur.fastconformer_screen.gates import card_gate

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="the benchmark's Hub revision")
    args = parser.parse_args(argv)

    from tadabur.audio import decode_to_mono_16k, read_audio_bytes

    model = load_ctc_model(args.nemo, "cpu")

    def decode(path: Path) -> str:
        return transcribe(model, decode_to_mono_16k(read_audio_bytes(path)))

    measured = measure(args.manifest, decode)
    written = record(measured, identity={"nemo": Path(args.nemo).name,
                                         "benchmark": "Quran-Lab/quranic-asr-benchmark",
                                         "benchmark_revision": args.revision})
    args.out.write_text(json.dumps(written, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: written[key] for key in ("measured", "gate_failed")}, indent=1),
          flush=True)
    card_gate(measured["wer"], measured["clips"])   # raises above twice the card's WER
