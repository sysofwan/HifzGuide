"""Does the model *hear* tashkeel, or infer it from the canonical text? (ADR-0003)

`training.tashkeel_eval` shows the fine-tune reproducing ~98% of the reference's short
vowels. That number alone cannot distinguish two very different models:

* one that **hears** the harakah in the audio, and
* one that has learned the **Quran's text** and reconstructs the harakah from the
  surrounding consonants.

The Quranic text is fixed and public, and the reference labels are generated
deterministically from it, so the consonant skeleton very nearly determines the
vowelization. A text-inferring model therefore scores ~98% on any corpus of *correct*
recitation while being useless for the product: Muraja is a recitation **checker**, and a
model that reconstructs the canonical vowel will silently "correct" a student's mistake
instead of flagging it. The capability only fails where it matters, so no amount of
held-out correct recitation can expose it.

This module isolates the two by restricting attention to **ambiguous skeletons**: consonant
skeletons that appear in training with *more than one* vowelization. There, the text prior is
genuinely uncertain, so the two models make different predictions:

* the text-inferring model can do no better than the training text's majority vowelization,
* a model that hears the audio should stay near its overall accuracy.

The majority-vowelization prior is computed from the **training** split's references (text
only, no audio, no model) and evaluated on the **val** split, which is exactly what a
memorizer could have learned. It is the control the headline number lacks: on this corpus it
scores ~0.71, far enough below ~0.98 that the two hypotheses are cleanly separable.

Usage::

    python -m training.minimal_pairs \
        --manifest tadabur/audit_run/seg_v21/manifest_raw.jsonl \
        --labels   tadabur/audit_run/seg_v21/windowed_labels_v3.jsonl \
        --audio-dir tadabur/audit_run/segment_audio_v2 \
        --model    tadabur/audit_run/seg_v21/rung3_v2/merged \
        --out      tadabur/audit_run/seg_v21/minimal_pairs_rung3.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tadabur.smith_waterman import smith_waterman
from training.tashkeel_eval import SHORT_VOWELS


def skeleton(word: str) -> str:
    """``word`` with every short vowel removed — what the text prior keys on."""
    return "".join(c for c in word if c not in SHORT_VOWELS)


def vowelization(word: str) -> str:
    """``word``'s short-vowel sequence, in order — what the model must get right."""
    return "".join(c for c in word if c in SHORT_VOWELS)


@dataclass(frozen=True)
class WordOccurrence:
    """One reference word in one segment, with the model's vowels for it."""

    skeleton: str
    reference_vowels: str
    decoded_vowels: str

    @property
    def correct(self) -> bool:
        return self.decoded_vowels == self.reference_vowels


def split_clips(labels_path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """The train / val clip ids, read from the windowed label file's own splits.

    The prior must be learnable from *training* material only, so the split has to be the
    one the model actually trained on rather than a fresh partition.
    """
    from training.windowed_labels import read_labels

    by_split = read_labels(labels_path)
    return (
        frozenset(w.clip_audio_filename for w in by_split.get("train", [])),
        frozenset(w.clip_audio_filename for w in by_split.get("val", [])),
    )


def read_segments(manifest_path: Path) -> list[dict]:
    """Manifest rows carrying both a raw reference and its verified word offsets."""
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("raw_reference_phonemes") and row.get("raw_word_offsets"):
            rows.append(row)
    return rows


def text_prior(segments: list[dict], clips: frozenset[str]) -> dict[str, Counter]:
    """Skeleton → vowelization counts, from reference text alone.

    No audio and no model: this is precisely the knowledge a text-memorizing model could
    have absorbed from the training split.
    """
    prior: dict[str, Counter] = defaultdict(Counter)
    for row in segments:
        if row["clip_audio_filename"] not in clips:
            continue
        reference, offsets = row["raw_reference_phonemes"], row["raw_word_offsets"]
        for start, end in zip(offsets, offsets[1:]):
            word = reference[start:end]
            if any(c in SHORT_VOWELS for c in word):
                prior[skeleton(word)][vowelization(word)] += 1
    return prior


def decoded_words(decode: str, reference: str, offsets: list[int]) -> list[tuple[str, str]]:
    """Pair each reference word with the decode's vowels over that word's span.

    The decode is aligned to the whole reference once, then each word's reference span is
    projected through the alignment onto a decode span. A word the alignment never reached
    yields an empty vowel string, which scores as wrong rather than being quietly dropped —
    failing to decode a word is not evidence of hearing its vowel.
    """
    alignment = smith_waterman(decode, reference)
    ref_to_query = {
        alignment.ref_start + i: q
        for i, q in enumerate(alignment.ref_to_query)
        if q is not None and q >= 0
    }

    pairs = []
    for start, end in zip(offsets, offsets[1:]):
        word = reference[start:end]
        if not any(c in SHORT_VOWELS for c in word):
            continue
        query_positions = [ref_to_query[i] for i in range(start, end) if i in ref_to_query]
        span = (
            decode[min(query_positions) : max(query_positions) + 1]
            if query_positions
            else ""
        )
        pairs.append((word, vowelization(span)))
    return pairs


def score(
    occurrences: list[WordOccurrence], prior: dict[str, Counter]
) -> dict:
    """Model accuracy vs the text-only prior, overall and on ambiguous skeletons."""
    ambiguous = {s for s, counts in prior.items() if len(counts) > 1}

    def prior_guess(sk: str) -> str:
        counts = prior.get(sk)
        return counts.most_common(1)[0][0] if counts else ""

    def block(items: list[WordOccurrence]) -> dict:
        if not items:
            return {"words": 0}
        model_hits = sum(1 for o in items if o.correct)
        prior_hits = sum(1 for o in items if prior_guess(o.skeleton) == o.reference_vowels)
        return {
            "words": len(items),
            "model_accuracy": round(model_hits / len(items), 4),
            "text_prior_accuracy": round(prior_hits / len(items), 4),
            "model_minus_prior": round((model_hits - prior_hits) / len(items), 4),
        }

    on_ambiguous = [o for o in occurrences if o.skeleton in ambiguous]
    unambiguous = [o for o in occurrences if o.skeleton not in ambiguous]
    return {
        "all_words": block(occurrences),
        "ambiguous_skeletons": block(on_ambiguous),
        "unambiguous_skeletons": block(unambiguous),
        "distinct_ambiguous_skeletons": len(ambiguous),
    }


def verdict(report: dict, margin: float = 0.10) -> dict:
    """Did the model beat what the canonical text alone can explain?

    The test is only meaningful when the prior is actually uncertain, so a corpus whose
    ambiguous slice is too small or too predictable is reported as inconclusive rather than
    as a pass.
    """
    amb = report["ambiguous_skeletons"]
    if amb.get("words", 0) < 100:
        return {"conclusive": False, "reason": "too few ambiguous-skeleton words to judge"}
    if amb["text_prior_accuracy"] > 0.95:
        return {
            "conclusive": False,
            "reason": "the text prior alone already explains these words",
        }
    hears = amb["model_minus_prior"] >= margin
    return {
        "conclusive": True,
        "hears_tashkeel": bool(hears),
        "margin_required": margin,
        "model_accuracy": amb["model_accuracy"],
        "text_prior_accuracy": amb["text_prior_accuracy"],
        "model_minus_prior": amb["model_minus_prior"],
        "interpretation": (
            "model beats what the canonical text can explain — it is using the audio"
            if hears
            else "model is at or below the text prior — the tashkeel number may be "
            "reconstruction from known text, not hearing"
        ),
    }


def segment_audio_path(audio_dir: Path, audio_filename: str) -> Path:
    """Resolve a segment's audio under either staged layout, as ``ClipAudioCache`` does."""
    from tadabur.audit_sampler import local_audio_path

    for candidate in (audio_dir / local_audio_path(audio_filename), audio_dir / audio_filename):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"segment audio for {audio_filename!r} not found under {audio_dir} under either "
        "the hash-prefixed (tadabur.audit_sampler) or plain name — stage it first."
    )


def _decode_segments(model_id: str, rows: list[dict], audio_dir: Path, batch_size: int,
                     device: str) -> list[str]:
    """Decode each segment's own audio file with ``model_id``."""
    from tadabur.audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
    from tadabur.inference import MuaalemPhonemeModel

    model = MuaalemPhonemeModel.load(model_id, device=device)
    decodes: list[str] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        waves = [
            np.asarray(
                decode_to_mono_16k(
                    segment_audio_path(audio_dir, r["audio_filename"]).read_bytes()
                ),
                dtype=np.float32,
            )
            for r in chunk
        ]
        decodes.extend(d.phonemes for d in model.decode_batch(waves, TARGET_SAMPLE_RATE))
        if (start // batch_size) % 20 == 0:
            print(f"  decoded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    del model
    return decodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="scored segment manifest carrying raw_word_offsets.")
    parser.add_argument("--labels", type=Path, required=True,
                        help="windowed labels JSONL — supplies the train/val clip split.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="per-segment audio directory.")
    parser.add_argument("--model", required=True, help="checkpoint to test.")
    parser.add_argument("--limit", type=int, default=0,
                        help="score at most this many val segments (0 = all).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin", type=float, default=0.10,
                        help="how far above the text prior counts as hearing.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    segments = read_segments(args.manifest)
    train_clips, val_clips = split_clips(args.labels)
    prior = text_prior(segments, train_clips)
    val_rows = [r for r in segments if r["clip_audio_filename"] in val_clips]
    if args.limit:
        val_rows = val_rows[: args.limit]
    if not val_rows:
        raise SystemExit("no val segments found — is --labels the split the model trained on?")
    print(
        f"{len(segments)} segments; prior over {len(prior)} skeletons from train; "
        f"decoding {len(val_rows)} val segments.",
        flush=True,
    )

    decodes = _decode_segments(
        args.model, val_rows, args.audio_dir, args.batch_size, args.device
    )

    occurrences: list[WordOccurrence] = []
    for row, decode in zip(val_rows, decodes):
        for word, decoded in decoded_words(
            decode, row["raw_reference_phonemes"], row["raw_word_offsets"]
        ):
            occurrences.append(
                WordOccurrence(skeleton(word), vowelization(word), decoded)
            )

    report = score(occurrences, prior)
    report["model"] = args.model
    report["val_segments"] = len(val_rows)
    report["verdict"] = verdict(report, args.margin)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
