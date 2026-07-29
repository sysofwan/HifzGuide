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
skeletons that appear in training with *more than one* vowelization. There, a memorizer
keyed on the word alone is genuinely uncertain.

That is not enough on its own. The model is bidirectional and decodes the surrounding
consonants reliably, so it can condition its vowel guess on the neighbouring words without
hearing a single harakah — and the Quran's text is fixed, so that context very nearly
determines the vowelization. The module therefore scores **two** text-only baselines built
from the training split's references (no audio, no model):

* ``unigram`` — majority vowelization for the word's own skeleton, and
* ``context`` — majority vowelization given both neighbouring skeletons.

On this corpus the unigram prior scores ~0.71 but the context prior scores ~0.97, above
every checkpoint measured. Judging against the unigram prior alone would therefore have
"proved" the model hears tashkeel when all it had beaten was a strawman. :func:`verdict`
judges against the **strongest** baseline and declines to rule when that baseline already
explains the words, because at that point the corpus simply cannot separate the two
hypotheses.

Two structural limits are reported rather than hidden. The split is by clip, so it separates
*reciters* but not Quranic *content*: :func:`ayah_overlap` measures how much of the val text
also appears in train. And every clip here is **correct** recitation, so no observation can
show what the model does when the audio and the canonical text disagree. Settling the
question needs counterfactual audio — the same word in the same context with a different
vowel actually spoken — which this corpus does not contain.

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
    """One reference word site in one segment, with the model's vowels for it."""

    site: WordSite
    decoded_vowels: str

    @property
    def skeleton(self) -> str:
        return self.site.skeleton

    @property
    def reference_vowels(self) -> str:
        return self.site.reference_vowels

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


@dataclass(frozen=True)
class WordSite:
    """One vowel-bearing reference word together with its textual neighbours.

    The neighbours are what make the memorizer baseline honest. The model is bidirectional
    and decodes the consonants of the surrounding words reliably, so it can condition a
    vowel guess on that context without hearing the harakah at all — which is exactly the
    failure mode under test.
    """

    skeleton: str
    prev_skeleton: str
    next_skeleton: str
    reference_vowels: str

    @property
    def context_key(self) -> tuple[str, str, str]:
        return (self.prev_skeleton, self.skeleton, self.next_skeleton)


BOUNDARY = ("^", "$")


def reference_sites(reference: str, offsets: list[int]) -> list[tuple[int, int, WordSite]]:
    """Every vowel-bearing word in one segment, with its span and its neighbours."""
    words = [reference[s:e] for s, e in zip(offsets, offsets[1:])]
    skeletons = [skeleton(w) for w in words]
    sites = []
    for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if not any(c in SHORT_VOWELS for c in words[i]):
            continue
        sites.append((
            start,
            end,
            WordSite(
                skeleton=skeletons[i],
                prev_skeleton=skeletons[i - 1] if i > 0 else BOUNDARY[0],
                next_skeleton=skeletons[i + 1] if i + 1 < len(skeletons) else BOUNDARY[1],
                reference_vowels=vowelization(words[i]),
            ),
        ))
    return sites


@dataclass
class TextPriors:
    """What a model could predict from the canonical text alone, at two strengths.

    ``unigram`` keys on the word's own skeleton; ``context`` also keys on both neighbouring
    skeletons. The unigram prior alone is a **strawman**: the Quran's text is fixed, so
    knowing the neighbouring words very nearly determines the vowelization. Measuring
    against the weak baseline only rules out a context-free memorizer, which is not the
    model anyone was worried about.
    """

    unigram: dict[str, Counter]
    context: dict[tuple[str, str, str], Counter]

    @property
    def ambiguous_skeletons(self) -> set[str]:
        return {sk for sk, counts in self.unigram.items() if len(counts) > 1}

    def guess_unigram(self, site: WordSite) -> str:
        counts = self.unigram.get(site.skeleton)
        return counts.most_common(1)[0][0] if counts else ""

    def guess_context(self, site: WordSite) -> str:
        """The context prediction, falling back to the unigram guess when unseen."""
        counts = self.context.get(site.context_key)
        return counts.most_common(1)[0][0] if counts else self.guess_unigram(site)


def text_prior(segments: list[dict], clips: frozenset[str]) -> TextPriors:
    """Build both memorizer baselines from reference text alone.

    No audio and no model: precisely the knowledge a text-memorizing model could have
    absorbed from the training split.
    """
    unigram: dict[str, Counter] = defaultdict(Counter)
    context: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for row in segments:
        if row["clip_audio_filename"] not in clips:
            continue
        for _, _, site in reference_sites(
            row["raw_reference_phonemes"], row["raw_word_offsets"]
        ):
            unigram[site.skeleton][site.reference_vowels] += 1
            context[site.context_key][site.reference_vowels] += 1
    return TextPriors(unigram=dict(unigram), context=dict(context))


def ayah_overlap(segments: list[dict], train: frozenset[str], val: frozenset[str]) -> dict:
    """How much of the val *text* the train split also contains.

    The split is by clip, so it separates reciters but not Quranic content. A val ayah that
    also appears in train is an ayah a text-memorizer could have learned outright, which
    caps what any held-out score here can prove.
    """
    def ayahs(clips: frozenset[str]) -> set:
        return {
            r["surah_ayah"] for r in segments if r["clip_audio_filename"] in clips
        }

    missing = [r for r in segments if "surah_ayah" not in r]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest rows lack 'surah_ayah' — the ayah-overlap limit "
            "cannot be measured, and reporting zero overlap would understate it."
        )

    train_ayahs, val_ayahs = ayahs(train), ayahs(val)
    shared = train_ayahs & val_ayahs
    return {
        "train_ayahs": len(train_ayahs),
        "val_ayahs": len(val_ayahs),
        "shared_ayahs": len(shared),
        "val_ayahs_also_in_train": (
            round(len(shared) / len(val_ayahs), 4) if val_ayahs else None
        ),
    }


def decoded_words(decode: str, reference: str, offsets: list[int]) -> list[tuple[WordSite, str]]:
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
    for start, end, site in reference_sites(reference, offsets):
        query_positions = [ref_to_query[i] for i in range(start, end) if i in ref_to_query]
        span = (
            decode[min(query_positions) : max(query_positions) + 1]
            if query_positions
            else ""
        )
        pairs.append((site, vowelization(span)))
    return pairs


def score(occurrences: list[WordOccurrence], priors: TextPriors) -> dict:
    """Model accuracy against both memorizer baselines, on the ambiguous slice.

    The **context** prior is the one that decides the question. The unigram prior is kept
    only to show how much of the model's apparent advantage is an artifact of a weak
    baseline.
    """
    ambiguous = priors.ambiguous_skeletons

    def block(items: list[WordOccurrence]) -> dict:
        if not items:
            return {"words": 0}
        n = len(items)
        model = sum(1 for o in items if o.correct) / n
        unigram = sum(1 for o in items if priors.guess_unigram(o.site) == o.reference_vowels) / n
        context = sum(1 for o in items if priors.guess_context(o.site) == o.reference_vowels) / n
        return {
            "words": n,
            "model_accuracy": round(model, 4),
            "unigram_prior_accuracy": round(unigram, 4),
            "context_prior_accuracy": round(context, 4),
            "model_minus_unigram_prior": round(model - unigram, 4),
            "model_minus_context_prior": round(model - context, 4),
        }

    return {
        "all_words": block(occurrences),
        "ambiguous_skeletons": block([o for o in occurrences if o.skeleton in ambiguous]),
        "unambiguous_skeletons": block([o for o in occurrences if o.skeleton not in ambiguous]),
        "distinct_ambiguous_skeletons": len(ambiguous),
    }


def verdict(report: dict, margin: float = 0.10) -> dict:
    """Did the model beat what the canonical text alone can explain?

    Judged against the **strongest** available text-only baseline. Beating only the weak
    unigram prior rules out a context-free memorizer and nothing more, so a model that
    fails to clear the context prior leaves the question open rather than answered.

    The test is only meaningful when that baseline is genuinely uncertain, so a corpus whose
    ambiguous slice is too small or whose text already explains the vowels is reported as
    inconclusive rather than as a pass.
    """
    amb = report["ambiguous_skeletons"]
    if amb.get("words", 0) < 100:
        return {"conclusive": False, "reason": "too few ambiguous-skeleton words to judge"}

    bar = max(amb["unigram_prior_accuracy"], amb["context_prior_accuracy"])
    if bar > 0.95:
        return {
            "conclusive": False,
            "reason": (
                "the canonical text alone already explains these words "
                f"(best text-only baseline {bar:.4f}) — this corpus cannot separate "
                "hearing from text reconstruction; a counterfactual test is required"
            ),
            "best_text_baseline": bar,
            "model_accuracy": amb["model_accuracy"],
        }

    hears = (amb["model_accuracy"] - bar) >= margin
    return {
        "conclusive": True,
        "hears_tashkeel": bool(hears),
        "margin_required": margin,
        "model_accuracy": amb["model_accuracy"],
        "best_text_baseline": bar,
        "model_minus_best_baseline": round(amb["model_accuracy"] - bar, 4),
        "interpretation": (
            "model beats what the canonical text can explain — it is using the audio"
            if hears
            else "model is at or below the strongest text-only baseline — the tashkeel "
            "number may be reconstruction from known text, not hearing"
        ),
    }


def segment_audio_path(audio_dir: Path, audio_filename: str) -> Path | None:
    """Resolve a segment's audio under either staged layout, or ``None`` if unstaged.

    ``ClipAudioCache`` raises here, but a partially staged segment directory is normal:
    callers pre-filter and *report* how many segments they dropped, so missing audio
    shrinks the sample visibly rather than biasing it silently.
    """
    from tadabur.audit_sampler import local_audio_path

    for candidate in (audio_dir / local_audio_path(audio_filename), audio_dir / audio_filename):
        if candidate.is_file():
            return candidate
    return None


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
    staged = [r for r in val_rows if segment_audio_path(args.audio_dir, r["audio_filename"])]
    unstaged = len(val_rows) - len(staged)
    val_rows = staged
    if args.limit:
        val_rows = val_rows[: args.limit]
    if not val_rows:
        raise SystemExit("no val segments found — is --labels the split the model trained on?")
    print(
        f"{len(segments)} segments; prior over {len(prior.unigram)} skeletons from train; "
        f"decoding {len(val_rows)} val segments ({unstaged} skipped, audio not staged).",
        flush=True,
    )

    decodes = _decode_segments(
        args.model, val_rows, args.audio_dir, args.batch_size, args.device
    )

    occurrences: list[WordOccurrence] = []
    for row, decode in zip(val_rows, decodes):
        for site, decoded in decoded_words(
            decode, row["raw_reference_phonemes"], row["raw_word_offsets"]
        ):
            occurrences.append(WordOccurrence(site, decoded))

    report = score(occurrences, prior)
    report["model"] = args.model
    report["val_segments"] = len(val_rows)
    report["val_segments_unstaged"] = unstaged
    report["ayah_overlap"] = ayah_overlap(segments, train_clips, val_clips)
    report["verdict"] = verdict(report, args.margin)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
