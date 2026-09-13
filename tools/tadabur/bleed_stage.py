"""Staging the timed decode and VAD intervals the bleed re-cut reads.

:mod:`tadabur.filter` writes two things per clean re-read (Muraja ADR-0016 decision 2):
the reject sink row — decode string, gate signals — and the 16 kHz mono WAV. Neither
carries a *time*. :mod:`tadabur.bleed_recut` needs both: per-frame CTC class ids, so
:func:`~tadabur.waqf_detect.collapse_with_times` can put an onset on every decoded
phoneme, and the recitation VAD's clean speech intervals, so the cut can prefer a pause
over a signal. This module is the pass that produces them.

It is deliberately **shard-free**. The shards are 2.4 GB each and ``--delete-shards``
throws them away as the filter walks them, so re-reading the corpus from parquet to get
a timed decode would re-download the whole run. The staged WAV is the same 16 kHz mono
waveform the gate scored, so decoding *that* reproduces the stored decode; the run
reports the rate at which it does, and a drift is a signal the staging is not reading
the audio the gate saw.

Only clips matching :func:`~tadabur.rejects.is_clean_re_read` are staged — the corpus
population. Prevalence over the *whole* reject pile is :mod:`tadabur.bleed_detect`'s
job, and it needs no audio at all.

The VAD and the phoneme model never co-reside on the GPU: the VAD runs first over every
clip and is freed before the phoneme model loads.

Usage:
  python -m tadabur.bleed_stage --rejects corpus_run/rejects.jsonl
    --clips corpus_run/clips --out corpus_run/decodes.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rejects import RejectRecord, read_reject_records

DEFAULT_BATCH_SIZE = 4
DEFAULT_VAD_BATCH_SIZE = 8


def clean_re_reads_with_audio(
    rejects: Path, clips: Path
) -> tuple[list[RejectRecord], list[str]]:
    """The clean re-reads in ``rejects``, split into those with staged audio and those without.

    A missing WAV is not fatal — a resumed filter run stages audio only for the shards it
    actually walked — but it is always worth naming, since the corpus silently shrinks by
    exactly that many clips.
    """
    records = [r for r in read_reject_records(rejects) if r.is_clean_re_read]
    present = [r for r in records if (clips / r.audio_filename).exists()]
    missing = [r.audio_filename for r in records if not (clips / r.audio_filename).exists()]
    return present, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rejects", type=Path, required=True, help="Reject sink JSONL (the predicate)."
    )
    parser.add_argument(
        "--clips", type=Path, required=True, help="Directory of staged 16 kHz WAVs."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Staged decodes JSONL to write."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Phoneme model batch."
    )
    parser.add_argument(
        "--vad-batch-size", type=int, default=DEFAULT_VAD_BATCH_SIZE, help="VAD batch."
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    parser.add_argument(
        "--limit", type=int, help="Stage only the first N clips in filename order."
    )
    args = parser.parse_args()

    # Imported here, not at module scope: selecting the population
    # (:func:`clean_re_reads_with_audio`) is pure JSON over the sink and a directory
    # listing, and must not drag in torch or the audio stack to be testable.
    import soundfile as sf
    import torch

    from .audio import TARGET_SAMPLE_RATE
    from .inference import MuaalemPhonemeModel
    from .vad import compute_clip_intervals

    records, missing = clean_re_reads_with_audio(args.rejects, args.clips)
    records.sort(key=lambda r: r.audio_filename)
    if args.limit is not None:
        records = records[: args.limit]
    if missing:
        print(f"WARNING: {len(missing)} clean re-read(s) have no staged WAV, e.g. {missing[:3]}")
    if not records:
        raise SystemExit(f"No clean re-reads with staged audio under {args.clips}.")
    print(f"staging {len(records)} clean re-reads from {args.clips}", flush=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    intervals = compute_clip_intervals(
        [r.audio_filename for r in records],
        args.clips,
        device=device,
        dtype=dtype,
        batch_size=args.vad_batch_size,
    )
    print(f"vad done for {len(intervals)} clips", flush=True)

    model = MuaalemPhonemeModel.load(device=device, dtype=dtype)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    with args.out.open("w", encoding="utf-8") as f:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            waveforms = [
                sf.read(args.clips / r.audio_filename, dtype="float32")[0] for r in batch
            ]
            for record, waveform, decode in zip(
                batch, waveforms, model.decode_batch(waveforms, TARGET_SAMPLE_RATE)
            ):
                matched += int(decode.phonemes == record.predicted_phonemes)
                f.write(
                    json.dumps(
                        {
                            "audio_filename": record.audio_filename,
                            "surah_ayah": record.surah_ayah,
                            "reciter_id": record.reciter_id,
                            "duration_s": len(waveform) / TARGET_SAMPLE_RATE,
                            "decode": decode.phonemes,
                            "stored_decode": record.predicted_phonemes,
                            "num_logit_frames": decode.num_logit_frames,
                            "class_ids": list(decode.class_ids),
                            "speech_intervals": [
                                [float(a), float(b)]
                                for a, b in intervals.get(record.audio_filename, [])
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"decoded {start + len(batch)}/{len(records)}", flush=True)

    share = matched / len(records)
    print(
        f"wrote {args.out}: {len(records)} clips, restaged decode reproduces the "
        f"gate's for {matched} ({share:.1%})"
    )


if __name__ == "__main__":
    main()
