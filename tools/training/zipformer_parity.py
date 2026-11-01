"""The #245 zipformer screen's reference-runtime gate: sherpa-onnx parity.

The card-reproduction gate stopped twice (:mod:`training.zipformer_gate`): the card's PER table
is v3's, and v3 does not reproduce it. Its replacement, fixed before running
(``.orch/tasks/zip-run/parity-gate.md``), asks the question a decoder bug would answer: do our
decoder and sherpa-onnx -- the runtime the card says these exports are built for -- emit the
same units on the same 577 benchmark clips?

* **The pass rule is per source** (studio, phone, unseen): at least :data:`IDENTICAL_PERCENT` of
  a source's clips must decode identically, and its native-unit edits must stay within
  :data:`EDITS_PER_MILLE` per mille of our emitted units. Pooling would let the phone clips --
  the ones a front-end difference hurts most -- hide inside the studio and unseen allowance
  (principal r2 §2).
* **The padded final chunk counts as a difference** (parity-gate.md rule 4). Ours advances while
  it has feature frames left and zero-pads the last call; sherpa decodes only while
  ``processed + input_frames < num_frames``. Any non-blank token the padded call emits makes the
  clip differing even when the transcripts agree.
* **The phone table-hit bound** (astra r1 §4.2): on the phone clips the card's table holds, the
  character distance between the two runtimes over the reference's own characters must stay
  within :data:`CHARACTER_PER_MILLE` per mille. By the edit-distance triangle inequality that
  bounds the difference in their character PER on exactly the disputed cohort, without letting
  opposing errors cancel.
* **The verdict-adapter probe** (parity-gate.md rule 5, astra r1 §4.3): nine canaries -- the
  shortest, median and longest clip of each source -- are decoded again on the crops
  ``checks-w1`` actually feeds the model (first preview, first window, a later warmed window,
  the stop), and the two runtimes' cut decisions must agree. sherpa-onnx exposes a token's first
  frame only, so its run ends somewhere in ``[start, next_start - 1]``; a decision both bounds
  agree on is exact (:func:`cut_certainty`), and any other is reported as ambiguous rather than
  guessed. Frame shifts away from a cut are the allowance (astra r1 §3); a shift that changes a
  cut is a mismatch.
* **The front end is compared, not shared** (rule 6, report only): our torchaudio kaldi fbank and
  sherpa's kaldi-native-fbank are fed the same waveform and tail, and the max-abs difference is
  recorded per source.
* **Audio is loaded once, then proven** (principal r2 §1, gap 1): every benchmark clip is 16 kHz
  mono PCM, so our loader resamples nothing; a clip that were not is refused, because this wheel
  ships no reader that could test a resampler difference. Each clip's waveform is hashed, and the
  sherpa phase re-reads the file directly and compares, so a shared-loader mistake cannot hide.
* **Nothing here trains, labels or selects** (#245's constraints): the record is the decoder's
  evidence, v3 never appears, and no sealed clip is decoded -- the plan drops those before any
  decode.

The gate runs in three phases, because our decoder needs torch/torchaudio and sherpa-onnx must
stay out of the repo's environments (a throwaway venv, never a dependency)::

    python -m training.zipformer_parity ours --work-dir /root/scratch/zipformer/data \\
        --out-dir /root/scratch/zipformer/parity --workers 12
    /root/scratch/sherpa-venv/bin/python -m training.zipformer_parity sherpa \\
        --ours-dir /root/scratch/zipformer/parity --model /path/v3.2.onnx --tokens /path/tokens.txt
    python -m training.zipformer_parity verdict --ours-dir /root/scratch/zipformer/parity \\
        --work-dir /root/scratch/zipformer/data \\
        --out tools/tadabur/zipformer_screen/parity.json

``scripts/zipformer-screen.sh parity launch`` stages the branch and runs all three detached on
the run box. Only the pure rule lives here (the tests import this module without torch); the
phases import their own runtimes inside the functions that need them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from training.zipformer_gate import card_table, character_per, edit_counts, reference_units

#: The committed gate record, beside the identity and the card gate it replaces.
PARITY_DIR = Path(__file__).resolve().parent.parent / "tadabur" / "zipformer_screen"
PARITY_PATH = PARITY_DIR / "parity.json"
DECODES_NAME = "parity_decodes.jsonl.gz"

#: The share of a source's clips that must decode identically (parity-gate.md rule 2).
IDENTICAL_PERCENT = 99
#: Native-unit edits allowed per thousand units we emitted, per source (rule 2).
EDITS_PER_MILLE = 1
#: Character distance allowed per thousand reference characters on the phone table-hit clips
#: (astra r1 §4.2).
CHARACTER_PER_MILLE = 1

#: The checks the probe exercises on each canary: the two looks ``checks-w1`` makes early, one
#: later warmed window, and the session's last check (astra r1 §4.3).
PROBE_CHECKS = ("first-preview", "first-window", "later-window", "stop")

#: sherpa's front end, pinned explicitly rather than inherited (parity-gate.md rule 1): the
#: card's own options, which are also sherpa's defaults, handed to its recognizer by value.
FEATURE_CONFIG = {
    "sampling_rate": 16_000,
    "feature_dim": 80,
    "low_freq": 20.0,
    "high_freq": -400.0,
    "dither": 0.0,
    "normalize_samples": True,
    "snip_edges": False,
}
#: The same two constants :mod:`training.zipformer_decode` decodes with: 16 kHz audio and the
#: card's 1.05 s flush. Compared against ours in the verdict, never assumed.
SHERPA_SAMPLE_RATE = 16_000
SHERPA_TAIL_SAMPLES = 16_800
#: Output steps per second: the card's 40 ms encoder step, which sherpa's timestamps are on.
SHERPA_STEPS_PER_SECOND = 25


class ParityStop(RuntimeError):
    """The parity gate refused to pass: a source, a bound, a probe check or a protocol fact."""


# --- the pre-declared rule -------------------------------------------------------------------


def our_calls(num_frames: int, *, chunk_frames: int) -> int:
    """How many calls our streaming loop makes over ``num_frames`` feature frames.

    Ours advances ``chunk_frames`` frames per call while any frames remain, zero-padding the
    last one: ``ceil(num_frames / chunk_frames)``.
    """
    return -(-num_frames // chunk_frames)


def sherpa_calls(num_frames: int, *, input_frames: int, chunk_frames: int) -> int:
    """How many calls sherpa's readiness rule makes over the same features.

    sherpa decodes only while ``processed + input_frames < num_frames``, so it never runs a call
    it would have to pad. The calls ours makes beyond this count are the padded tail
    parity-gate.md rule 4 watches.
    """
    remaining = num_frames - input_frames
    return -(-remaining // chunk_frames) if remaining > 0 else 0


def identical_clip(ours: Sequence[str], sherpa: Sequence[str],
                   padded_tokens: Sequence[int]) -> bool:
    """Whether a clip decodes identically: the same units *and* no padded-call emission.

    A padded emission is a difference even when the transcripts agree (parity-gate.md rule 4);
    it means the two runtimes' streams diverged where one of them had run out of audio.
    """
    return not padded_tokens and list(ours) == list(sherpa)


def source_rule(rows: Iterable[Mapping], sources: Sequence[str]) -> dict:
    """Each source's parity against the rule: identical clips and native-unit edits.

    ``rows`` are one per decoded clip (``key``, ``source``, ``ours``, ``sherpa`` and
    ``padded_tokens``). Both thresholds are compared in integers, so an exact boundary passes
    (:func:`our_calls`' ``>=``, as the gate's other rules are inclusive): ``identical * 100 >=
    percent * clips`` and ``edits * 1000 <= per_mille * units_ours``.
    """
    totals = {source: {"clips": 0, "identical": 0, "differing": [], "unit_edits": 0,
                       "units_ours": 0, "units_sherpa": 0, "padded_clips": 0}
              for source in sources}
    for row in rows:
        entry = totals[row["source"]]
        ours, sherpa = list(row["ours"]), list(row["sherpa"])
        edits, units = edit_counts(ours, sherpa)
        entry["unit_edits"] += edits
        entry["units_ours"] += units
        entry["units_sherpa"] += len(sherpa)
        if row["padded_tokens"]:
            entry["padded_clips"] += 1
        if identical_clip(ours, sherpa, row["padded_tokens"]):
            entry["identical"] += 1
        else:
            entry["differing"].append(row["key"])
        entry["clips"] += 1
    for entry in totals.values():
        entry["identical_percent"] = (
            100.0 * entry["identical"] / entry["clips"] if entry["clips"] else None)
        entry["edits_per_mille"] = (
            1000.0 * entry["unit_edits"] / entry["units_ours"] if entry["units_ours"] else None)
        entry["passes"] = (
            entry["clips"] > 0
            and entry["identical"] * 100 >= IDENTICAL_PERCENT * entry["clips"]
            and entry["unit_edits"] * 1000 <= EDITS_PER_MILLE * entry["units_ours"]
        )
    return totals


def phone_rule(distance: int, reference_characters: int) -> dict:
    """The phone table-hit bound: character distance between the runtimes over reference chars.

    ``distance`` is the character edit distance between what the two runtimes emitted on the
    clips the card's table scores; the denominator is the reference's own characters, so the
    bound is on their character PER difference, not on the size of our output.
    """
    if reference_characters <= 0:
        raise ValueError("the phone table-hit cohort has no reference characters")
    return {
        "character_distance": distance,
        "reference_characters": reference_characters,
        "per_mille": 1000.0 * distance / reference_characters,
        "passes": distance * 1000 <= CHARACTER_PER_MILLE * reference_characters,
    }


def cut_decision(start: int, end: int, boundary: int) -> bool:
    """Whether an emission whose run spans ``[start, end]`` steps is kept at ``boundary``.

    ``checks-w1`` drops an emission whose midpoint is before the check's start (and splits a
    window at its hop): the midpoint is ``(start + end) / 2``, so the comparison is exact in
    integers and an emission exactly at the boundary is kept.
    """
    return start + end >= 2 * boundary


def cut_certainty(start: int, next_start: int | None, boundary: int) -> str:
    """What sherpa's start frames fix about that decision: ``kept``, ``dropped`` or ``ambiguous``.

    sherpa exposes a token's first frame only, so its run ends somewhere in
    ``[start, next_start - 1]`` (a single frame for the last token). A decision both ends agree
    on is determined; anything else is ambiguous and reported rather than guessed.
    """
    end_high = start if next_start is None else max(start, next_start - 1)
    low = cut_decision(start, start, boundary)
    high = cut_decision(start, end_high, boundary)
    if low and high:
        return "kept"
    if not low and not high:
        return "dropped"
    return "ambiguous"


def probe_check(entry: Mapping) -> dict:
    """One sampled check's parity: the same units, and the same side of every cut boundary.

    ``entry`` carries our emissions (``start``/``end`` steps, exact) beside sherpa's
    (``start`` only), the crop-relative ``boundaries`` of the check, and the check's name. A
    determined sherpa decision that differs from ours is a mismatch; an ambiguous one is
    counted, not excused.
    """
    ours, sherpa = list(entry["ours"]), list(entry["sherpa"])
    tokens_equal = [emission["text"] for emission in ours] == [
        emission["text"] for emission in sherpa]
    starts_equal = [emission["start"] for emission in ours] == [
        emission["start"] for emission in sherpa]
    ambiguous, mismatches = 0, []
    if tokens_equal:
        for boundary in entry["boundaries"]:
            for index, (mine, theirs) in enumerate(zip(ours, sherpa)):
                next_start = (sherpa[index + 1]["start"]
                              if index + 1 < len(sherpa) else None)
                certainty = cut_certainty(theirs["start"], next_start, boundary)
                kept = cut_decision(mine["start"], mine["end"], boundary)
                if certainty == "ambiguous":
                    ambiguous += 1
                elif (certainty == "kept") != kept:
                    mismatches.append({
                        "boundary": boundary, "index": index, "unit": mine["text"],
                        "sherpa": certainty, "ours": "kept" if kept else "dropped",
                    })
    return {
        **entry,
        "tokens_equal": tokens_equal,
        "starts_equal": starts_equal,
        "ambiguous": ambiguous,
        "mismatches": mismatches,
        "passes": tokens_equal and not mismatches,
    }


def require_parity(record: Mapping) -> None:
    """Stop unless every source, the phone bound and every probe check passed."""
    failures: list[str] = []
    for source, entry in sorted(record["sources"].items()):
        if entry["passes"]:
            continue
        if not entry["clips"]:
            failures.append(f"{source}: no clip was decoded")
            continue
        failures.append(
            f"{source}: {entry['identical']}/{entry['clips']} identical "
            f"({entry['identical_percent']:.2f}%, need {IDENTICAL_PERCENT}%), "
            f"{entry['unit_edits']}/{entry['units_ours']} unit edits "
            f"({entry['edits_per_mille']:.3f} per mille, limit {EDITS_PER_MILLE})")
    phone = record["phone_table_hits"]
    if not phone["passes"]:
        failures.append(
            f"phone table hits: {phone['character_distance']} characters of "
            f"{phone['reference_characters']} ({phone['per_mille']:.3f} per mille, "
            f"limit {CHARACTER_PER_MILLE})")
    failed = [check for check in record["probe"]["checks"] if not check["passes"]]
    if not record["probe"]["checks"]:
        failures.append("probe: no check was exercised")
    for check in failed:
        failures.append(
            f"probe {check['key']} {check['check']}: "
            + ("the runtimes emitted different units"
               if not check["tokens_equal"]
               else f"{len(check['mismatches'])} cut decisions differ"))
    if failures:
        raise ParityStop("; ".join(failures))


# --- the shared record plumbing --------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fbank_name(key: str) -> str:
    """One clip's feature dump's file name; a benchmark key carries a ``/``."""
    return key.replace("/", "__") + ".npy"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(record: Mapping, path: Path) -> None:
    """Write one phase's scratch record, indent 1 and sorted keys.

    ``tadabur.tlog_index.write_json`` is the canonical writer for a committed record, but it
    imports the corpus readers and therefore librosa; the sherpa phase must run in the throwaway
    venv, which holds only numpy, soundfile, sherpa-onnx and kaldi-native-fbank. The committed
    ``parity.json`` still goes through the canonical writer in the verdict.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(record), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping]) -> None:
    """The committed per-clip record: gzipped with a zero mtime, so a re-run is identical."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
                          + "\n").encode("utf-8"))


# --- phase 1: our decoder's side -------------------------------------------------------------


def _probe_spans(num_samples: int) -> list[tuple[str, object]]:
    """The checks the canaries exercise: first preview, first window, a later window, the stop.

    "A later warmed window" is the last window of the schedule, so it is the one whose warm-up
    start is furthest from zero and whose cache path is the same, decoded again.
    """
    from training.decoding import PREVIEW, WINDOW, check_schedule

    spans = check_schedule(num_samples)
    previews = [span for span in spans if span.kind == PREVIEW]
    windows = [span for span in spans if span.kind == WINDOW]
    chosen: list[tuple[str, object]] = []
    if previews:
        chosen.append(("first-preview", previews[0]))
    if windows:
        chosen.append(("first-window", windows[0]))
        later = [span for span in windows if span.start > windows[0].start]
        if later:
            chosen.append(("later-window", later[-1]))
    chosen.append(("stop", spans[-1]))
    return chosen


def _probe(decoder, samples, name: str, span) -> dict:
    """One sampled check's our-side emissions, on the crop ``checks-w1`` would decode.

    The crop is ``[max(0, start - one hop), end)``: the fresh warm-started inference the
    verdict arm runs. Its emissions stay in the crop's own frame space, and ``boundaries`` are
    that space's cut positions, so the verdict can compare them with sherpa's unchanged.
    """
    import numpy as np

    from training.decoding import PREVIEW, STOP
    from training.zipformer_decode import HOP_STEPS, STEP_SAMPLES, scan_units, warmup_start

    begin = warmup_start(span)
    emissions = scan_units(np.concatenate(decoder.stream_calls(samples[begin:span.end])),
                           decoder.tokens)
    offset = begin // STEP_SAMPLES
    start = span.start // STEP_SAMPLES - offset
    boundaries = ([start] if span.kind in (PREVIEW, STOP)
                  else [start, start + HOP_STEPS])
    return {
        "check": name,
        "kind": span.kind,
        "span": [begin, span.end],
        "boundaries": boundaries,
        "ours": [{"text": emission.text, "start": emission.start_step,
                  "end": emission.end_step} for emission in emissions],
    }


def _ours_clip(item: tuple) -> dict:
    """One clip's our-side record: units, frame counts, the padded tail, and any probes."""
    import numpy as np
    import soundfile

    from tadabur.audio import decode_to_mono_16k_with_rate_and_decoder, read_audio_bytes
    from training.distill_data import SAMPLE_RATE
    from training.zipformer_decode import (BLANK_ID, CHUNK_FRAMES, INPUT_FRAMES, featurize,
                                           scan_units)
    from training.zipformer_run import worker_decoder

    key, source, audio, fbank_path, probes = item
    decoder = worker_decoder()
    raw = read_audio_bytes(Path(audio))
    samples, rate, reader = decode_to_mono_16k_with_rate_and_decoder(raw)
    if rate != SAMPLE_RATE:
        raise ValueError(
            f"{key}: {rate} Hz. Only 16 kHz clips can be held to this gate: sherpa-onnx's "
            "wheel ships no reader that could test a resampler difference (principal r2 §1)")
    calls = decoder.stream_calls(samples)
    features = featurize(samples, decoder.fbank, scale=decoder.scale)
    predicted = sherpa_calls(len(features), input_frames=INPUT_FRAMES,
                             chunk_frames=CHUNK_FRAMES)
    emissions = scan_units(np.concatenate(calls), decoder.tokens)
    record = {
        "key": key,
        "source": source,
        "audio": audio,
        "audio_sha256": _sha256(raw),
        "sample_rate": rate,
        "channels": soundfile.info(str(audio)).channels,
        "reader": reader,
        "num_samples": int(samples.size),
        "num_frames": int(len(features)),
        "calls": len(calls),
        "sherpa_calls": predicted,
        "padded_tokens": [int(frame) for call in calls[predicted:] for frame in call
                          if int(frame) != BLANK_ID],
        "waveform_sha256": _sha256(samples.tobytes()),
        "ours": [emission.text for emission in emissions],
    }
    np.save(fbank_path, features)
    if probes:
        record["probes"] = [_probe(decoder, samples, name, span)
                            for name, span in _probe_spans(len(samples))]
    return record


def _canaries(keys: Sequence[str], source_of: Mapping[str, str],
              durations: Mapping[str, int]) -> list[str]:
    """The shortest, median and longest clip of each source, the clip id breaking ties.

    Fixed before any result is looked at (astra r1 §4.3): no outcome selects a canary, and no
    sealed clip is in ``keys`` to select.
    """
    chosen: list[str] = []
    for source in sorted(set(source_of.values())):
        ordered = sorted((key for key in keys if source_of[key] == source),
                         key=lambda key: (durations[key], key))
        chosen.extend({ordered[0], ordered[len(ordered) // 2], ordered[-1]})
    return sorted(chosen)


def _duration(path: Path) -> int:
    """A clip's duration in 16 kHz samples, from its header (no decode)."""
    import soundfile

    from training.distill_data import SAMPLE_RATE

    info = soundfile.info(str(path))
    return int(round(info.frames * SAMPLE_RATE / info.samplerate))


def _ours_protocol(scale: float) -> dict:
    """The constants our side decoded with, read from the decoder that owns them."""
    from training.distill_data import SAMPLE_RATE
    from training.zipformer_decode import (BLANK_ID, CHUNK_FRAMES, CHUNK_SIZE, FBANK_OPTIONS,
                                           IDENTITY_PATH, INPUT_FRAMES, STEP_SAMPLES,
                                           TAIL_SAMPLES)

    return {
        "ours": {
            "identity": str(IDENTITY_PATH.relative_to(PARITY_DIR.parents[2])),
            "identity_sha256": _sha256_file(IDENTITY_PATH),
            "sample_scale": scale,
            "sample_rate": SAMPLE_RATE,
            "step_samples": STEP_SAMPLES,
            "chunk": CHUNK_SIZE,
            "chunk_frames": CHUNK_FRAMES,
            "input_frames": INPUT_FRAMES,
            "tail_samples": TAIL_SAMPLES,
            "blank_id": BLANK_ID,
            "fbank": dict(FBANK_OPTIONS),
            "tokens_sha256": _sha256_file(PARITY_DIR / "tokens.txt"),
        }
    }


def ours(work_dir: Path, out_dir: Path, workers: int, limit: int | None = None) -> None:
    """Decode the plan's clips with our decoder and record what the rule reads of our side.

    The plan is the committed disjointness plan, bound to the fetched manifest's SHA-256 (no
    sealed clip is decoded); the identity is verified before any decode, so the arm this gate
    checks is the one every comparison decode rests on.
    """
    from concurrent.futures import ProcessPoolExecutor

    from training.zipformer_decode import IDENTITY_PATH, ONNX_FILE, load_zipformer
    from training.zipformer_run import (CARD_SOURCES, decode_plan, files_from_record, load_fetch,
                                        manifest_rows, start_worker)

    out_dir = Path(out_dir)
    fetch = load_fetch(work_dir)
    files = files_from_record(work_dir, fetch, onnx=ONNX_FILE)
    scale = float(_read_json(IDENTITY_PATH)["identity"]["sample_scale"])
    load_zipformer(files, scale=scale, identity_path=IDENTITY_PATH)
    plan = decode_plan(files.manifest)
    rows = {row["id"]: row for row in manifest_rows(files.manifest)}
    keys = list(plan["decode"])
    if limit is not None:
        keys = keys[:limit]
    base = files.manifest.parent
    selected = _canaries(keys, {key: CARD_SOURCES[rows[key]["source"]] for key in keys},
                         {key: _duration(base / rows[key]["audio"]) for key in keys})

    shutil.rmtree(out_dir / "fbank", ignore_errors=True)
    (out_dir / "fbank").mkdir(parents=True, exist_ok=True)
    items = [(key, CARD_SOURCES[rows[key]["source"]], str(base / rows[key]["audio"]),
              str(out_dir / "fbank" / _fbank_name(key)), key in selected) for key in keys]
    print(f"parity ours: {len(items)} clips, {len(selected)} canaries, scale {scale}, "
          f"{workers} workers", flush=True)
    with open(out_dir / "ours.jsonl", "w", encoding="utf-8") as handle, ProcessPoolExecutor(
            max_workers=workers, initializer=start_worker,
            initargs=(files, scale)) as pool:
        for done, record in enumerate(pool.map(_ours_clip, items, chunksize=2), 1):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if done % 100 == 0:
                print(f"  ours: {done}/{len(items)} clips", flush=True)
    _write_json(_ours_protocol(scale), out_dir / "ours-protocol.json")
    print(f"parity ours: {len(items)} clips decoded -> {out_dir}", flush=True)


# --- phase 2: sherpa-onnx's side -------------------------------------------------------------


def _fbank_options():
    """kaldi-native-fbank's options set explicitly to the card's front end.

    The library is the one inside sherpa-onnx, so this is the reference front end itself; the
    options are set rather than defaulted, and recorded with ``as_dict()``.
    """
    import kaldi_native_fbank

    options = kaldi_native_fbank.FbankOptions()
    options.frame_opts.samp_freq = float(SHERPA_SAMPLE_RATE)
    options.frame_opts.frame_shift_ms = 10.0
    options.frame_opts.frame_length_ms = 25.0
    options.frame_opts.dither = 0.0
    options.frame_opts.window_type = "povey"
    options.frame_opts.snip_edges = False
    options.mel_opts.num_bins = 80
    options.mel_opts.low_freq = 20.0
    options.mel_opts.high_freq = -400.0
    options.use_energy = False
    return options


def _recognizer(model: Path, tokens: Path):
    """sherpa's zipformer2 CTC recognizer with the front end pinned by hand.

    ``from_zipformer2_ctc`` builds ``FeatureExtractorConfig`` from the library's defaults; the
    gate must pin what it compares against, so the config is built here and the pybind
    recognizer is constructed from it directly: CPU, one thread, greedy, endpointing and text
    rewrites off (parity-gate.md rule 1).
    """
    from sherpa_onnx.lib._sherpa_onnx import (OnlineModelConfig, OnlineRecognizer,
                                              OnlineRecognizerConfig,
                                              OnlineZipformer2CtcModelConfig, ProviderConfig)

    config = OnlineRecognizerConfig(
        feat_config=_feature_config(),
        model_config=OnlineModelConfig(
            zipformer2_ctc=OnlineZipformer2CtcModelConfig(model=str(model)),
            tokens=str(tokens),
            num_threads=1,
            provider_config=ProviderConfig(provider="cpu", device=0),
            debug=False,
        ),
        decoding_method="greedy_search",
        enable_endpoint=False,
    )
    return OnlineRecognizer(config)


def _feature_config():
    """sherpa's feature extractor, every value of :data:`FEATURE_CONFIG` given explicitly."""
    from sherpa_onnx.lib._sherpa_onnx import FeatureExtractorConfig

    return FeatureExtractorConfig(**FEATURE_CONFIG)


def _sherpa_decode(recognizer, samples) -> dict:
    """One waveform's units, start frames and call count through sherpa's streaming loop.

    The audio is the 16 kHz mono waveform plus exactly :data:`SHERPA_TAIL_SAMPLES` zeros, then
    ``input_finished``: the card's flush, fed as waveform rather than as padded features.
    sherpa's timestamps are seconds on the 40 ms encoder step; they are turned back into the
    frame index our emissions carry.
    """
    import numpy as np

    stream = recognizer.create_stream()
    stream.accept_waveform(SHERPA_SAMPLE_RATE, samples)
    stream.accept_waveform(SHERPA_SAMPLE_RATE,
                           np.zeros(SHERPA_TAIL_SAMPLES, dtype=np.float32))
    stream.input_finished()
    calls = 0
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
        calls += 1
    result = recognizer.get_result(stream)
    return {
        "tokens": list(result.tokens),
        "start": [int(round(stamp * SHERPA_STEPS_PER_SECOND)) for stamp in result.timestamps],
        "calls": calls,
    }


def _knf_features(waveform, options):
    """kaldi-native-fbank's features over the waveform plus the flush's tail."""
    import numpy as np
    import kaldi_native_fbank

    padded = np.concatenate([waveform, np.zeros(SHERPA_TAIL_SAMPLES, dtype=np.float32)])
    fbank = kaldi_native_fbank.OnlineFbank(options)
    fbank.accept_waveform(SHERPA_SAMPLE_RATE, padded)
    fbank.input_finished()
    return np.stack([fbank.get_frame(index) for index in range(fbank.num_frames_ready)])


_SHARED: dict[str, object] = {}


def _start_sherpa(model: str, tokens: str) -> None:
    _SHARED["recognizer"] = _recognizer(Path(model), Path(tokens))
    _SHARED["fbank_options"] = _fbank_options()


def _sherpa_clip(item: tuple) -> dict:
    """One clip's sherpa-side record: units, starts, call count, probe crops, fbank difference."""
    import numpy as np
    import soundfile

    key, audio, ours_sha, ours_raw_sha, fbank_path, probes = item
    if (found := _sha256(Path(audio).read_bytes())) != ours_raw_sha:
        raise ValueError(f"{key}: the file is {found}, our loader read {ours_raw_sha}")
    waveform, rate = soundfile.read(audio, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if rate != SHERPA_SAMPLE_RATE:
        raise ValueError(f"{key}: {rate} Hz; this gate only holds 16 kHz clips")
    if (found := _sha256(waveform.tobytes())) != ours_sha:
        raise ValueError(f"{key}: the direct read is {found}, our loader's waveform {ours_sha}")
    decoded = _sherpa_decode(_SHARED["recognizer"], waveform)
    reference = np.load(fbank_path)
    theirs = _knf_features(waveform, _SHARED["fbank_options"])
    same_shape = theirs.shape == reference.shape
    record = {
        "key": key,
        "sample_rate": rate,
        "reader": "soundfile",
        "waveform_sha256": _sha256(waveform.tobytes()),
        "tokens": decoded["tokens"],
        "start": decoded["start"],
        "calls": decoded["calls"],
        "fbank_frames": list(theirs.shape),
        "fbank_max_abs": float(np.abs(theirs - reference).max()) if same_shape else None,
        "probes": [],
    }
    for probe in probes:
        begin, end = probe["span"]
        got = _sherpa_decode(_SHARED["recognizer"], waveform[begin:end])
        record["probes"].append({
            "span": [begin, end],
            "sherpa": [{"text": text, "start": start}
                       for text, start in zip(got["tokens"], got["start"])],
        })
    return record


def _sherpa_protocol(model: Path, tokens: Path) -> dict:
    """What sherpa's side ran: the wheel, its ONNX Runtime, and the front end by value."""
    import importlib.metadata as metadata

    import sherpa_onnx
    import soundfile
    from sherpa_onnx.lib import _sherpa_onnx

    options = _fbank_options()
    extension = Path(_sherpa_onnx.__file__)
    return {
        "version": sherpa_onnx.__version__,
        "onnxruntime": sherpa_onnx.onnxruntime_version,
        "extension_sha256": _sha256_file(extension),
        "wheel": f"sherpa-onnx=={metadata.version('sherpa-onnx')}",
        "model": {"path": str(model), "sha256": _sha256_file(model)},
        "tokens": {"path": str(tokens), "sha256": _sha256_file(tokens)},
        "front_end": {
            "library": "kaldi-native-fbank",
            "version": metadata.version("kaldi-native-fbank"),
            "feature_config": dict(FEATURE_CONFIG),
            "frame_options": options.frame_opts.as_dict(),
            "mel_options": options.mel_opts.as_dict(),
            "fbank_options": options.as_dict(),
        },
        "audio_reader": {"soundfile": soundfile.__version__,
                         "libsndfile": soundfile.__libsndfile_version__},
        "protocol": {
            "sample_rate": SHERPA_SAMPLE_RATE,
            "tail_samples": SHERPA_TAIL_SAMPLES,
            "steps_per_second": SHERPA_STEPS_PER_SECOND,
            "decoding": "greedy_search",
            "num_threads": 1,
            "provider": "cpu",
            "endpoint_detection": False,
            "text_rewrites": [],
        },
    }


def sherpa(ours_dir: Path, model: Path, tokens: Path, workers: int) -> None:
    """Decode our side's clips with sherpa-onnx and record what the rule reads of its side."""
    from concurrent.futures import ProcessPoolExecutor

    ours_dir = Path(ours_dir)
    rows = _read_jsonl(ours_dir / "ours.jsonl")
    items = [(row["key"], row["audio"], row["waveform_sha256"], row["audio_sha256"],
              str(ours_dir / "fbank" / _fbank_name(row["key"])), row.get("probes", []))
             for row in rows]
    print(f"parity sherpa: {len(items)} clips, {workers} workers, sherpa-onnx with the card's "
          "front end pinned", flush=True)
    with open(ours_dir / "sherpa.jsonl", "w", encoding="utf-8") as handle, ProcessPoolExecutor(
            max_workers=workers, initializer=_start_sherpa,
            initargs=(str(model), str(tokens))) as pool:
        for done, record in enumerate(pool.map(_sherpa_clip, items, chunksize=2), 1):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if done % 100 == 0:
                print(f"  sherpa: {done}/{len(items)} clips", flush=True)
    _write_json(_sherpa_protocol(Path(model), Path(tokens)),
                ours_dir / "sherpa-protocol.json")
    print(f"parity sherpa: {len(items)} clips decoded -> {ours_dir}", flush=True)


# --- phase 3: the verdict --------------------------------------------------------------------


def _require_same_front_end(ours_protocol: Mapping, sherpa_protocol: Mapping) -> None:
    """Stop unless both phases were fed the same waveform, tail and front end.

    The two phases compute their features in different libraries; if their inputs or options
    differ, the comparison is between front ends, not between runtimes.
    """
    mine = ours_protocol["ours"]
    theirs = sherpa_protocol
    config = theirs["front_end"]["feature_config"]
    frames = theirs["front_end"]["frame_options"]
    mismatches: list[str] = []

    def check(name, ours_value, sherpa_value):
        if ours_value != sherpa_value:
            mismatches.append(f"{name}: ours {ours_value!r}, sherpa {sherpa_value!r}")

    check("sample rate", mine["sample_rate"], theirs["protocol"]["sample_rate"])
    check("tail samples", mine["tail_samples"], theirs["protocol"]["tail_samples"])
    check("steps per second", mine["sample_rate"] // mine["step_samples"],
          theirs["protocol"]["steps_per_second"])
    check("fbank bins", mine["fbank"]["num_mel_bins"], config["feature_dim"])
    check("fbank rate", mine["fbank"]["sample_frequency"], config["sampling_rate"])
    check("fbank shift (ms)", mine["fbank"]["frame_shift"], frames["frame_shift_ms"])
    check("fbank length (ms)", 25.0, frames["frame_length_ms"])
    check("fbank window", mine["fbank"]["window_type"], frames["window_type"])
    check("fbank dither", mine["fbank"]["dither"], config["dither"])
    check("fbank snip_edges", mine["fbank"]["snip_edges"], config["snip_edges"])
    check("fbank high_freq", mine["fbank"]["high_freq"], config["high_freq"])
    check("tokens", mine["tokens_sha256"], theirs["tokens"]["sha256"])
    if mismatches:
        raise ParityStop("the phases did not share a front end: " + "; ".join(mismatches))


def _phone_table_hits(rows: Sequence[Mapping], sherpa_rows: Mapping[str, Mapping],
                      work_dir: Path) -> dict:
    """The phone clips the card's table scores, and the two runtimes' distance over them."""
    from training.zipformer_run import (card_inventory, files_from_record, load_fetch,
                                        manifest_rows)

    files = files_from_record(work_dir, load_fetch(work_dir))
    table = card_table(json.loads(files.text2phoneme.read_text(encoding="utf-8")))
    units = card_inventory()
    manifest = {row["id"]: row for row in manifest_rows(files.manifest)}
    hits: list[str] = []
    distance = reference_characters = 0
    ours_rows, sherpa_per_rows = [], []
    for row in rows:
        if row["source"] != "phone":
            continue
        gold = reference_units(table, manifest[row["key"]]["reference_text"], units)
        if gold is None:
            continue
        hits.append(row["key"])
        reference = "".join(gold)
        theirs = "".join(sherpa_rows[row["key"]]["tokens"])
        distance += edit_counts("".join(row["ours"]), theirs)[0]
        reference_characters += len(reference)
        ours_rows.append({"source": "phone", "gold": gold, "pred": row["ours"]})
        sherpa_per_rows.append({"source": "phone", "gold": gold,
                                "pred": sherpa_rows[row["key"]]["tokens"]})
    if not hits:
        raise ParityStop("no phone clip has a card-table reference: the bound has no cohort")
    return {
        "clips": len(hits),
        **phone_rule(distance, reference_characters),
        "ours": character_per(ours_rows)["phone"],
        "sherpa": character_per(sherpa_per_rows)["phone"],
    }


def verdict(ours_dir: Path, work_dir: Path, out: Path, decodes: Path) -> dict:
    """Apply the pre-declared rule to both phases' records and write the gate's evidence."""
    from tadabur.tlog_index import write_json

    ours_dir = Path(ours_dir)
    rows = _read_jsonl(ours_dir / "ours.jsonl")
    sherpa_rows = {row["key"]: row for row in _read_jsonl(ours_dir / "sherpa.jsonl")}
    _require_same_front_end(_read_json(ours_dir / "ours-protocol.json"),
                            _read_json(ours_dir / "sherpa-protocol.json"))
    unpaired = sorted({row["key"] for row in rows} ^ set(sherpa_rows))
    if unpaired:
        raise ParityStop(
            f"{len(unpaired)} clips were decoded by one runtime only, e.g. {unpaired[:3]}")

    rule_rows = [{"key": row["key"], "source": row["source"], "ours": row["ours"],
                  "sherpa": sherpa_rows[row["key"]]["tokens"],
                  "padded_tokens": row["padded_tokens"]} for row in rows]
    sources = source_rule(rule_rows, sorted({row["source"] for row in rule_rows}))
    pooled = {field: sum(entry[field] for entry in sources.values())
              for field in ("clips", "identical", "unit_edits", "units_ours", "units_sherpa",
                            "padded_clips")}
    pooled["reported_only"] = "the pass rule is per source (principal r2 §2)"

    call_mismatches = [row["key"] for row in rows
                       if row["sherpa_calls"] != sherpa_rows[row["key"]]["calls"]]
    if call_mismatches:
        raise ParityStop(
            f"{len(call_mismatches)} clips: sherpa's call count is not the readiness rule's, "
            f"e.g. {call_mismatches[:3]}")
    waveform_mismatches = [row["key"] for row in rows
                           if sherpa_rows[row["key"]]["waveform_sha256"]
                           != row["waveform_sha256"]]
    if waveform_mismatches:
        raise ParityStop(
            f"{len(waveform_mismatches)} clips read differently by the two loaders, "
            f"e.g. {waveform_mismatches[:3]}")

    checks: list[dict] = []
    for row in rows:
        if not row.get("probes"):
            continue
        theirs = {tuple(probe["span"]): probe
                  for probe in sherpa_rows[row["key"]]["probes"]}
        for probe in row["probes"]:
            if tuple(probe["span"]) not in theirs:
                raise ParityStop(
                    f"{row['key']}: sherpa did not decode the {probe['check']} probe")
            checks.append(probe_check({**probe, "key": row["key"], "source": row["source"],
                                       "sherpa": theirs[tuple(probe["span"])]["sherpa"]}))
    probe = {
        "canaries": sorted({row["key"] for row in rows if row.get("probes")}),
        "checks": checks,
        "ambiguous_decisions": sum(check["ambiguous"] for check in checks),
        "passes": bool(checks) and all(check["passes"] for check in checks),
    }

    fbank: dict[str, dict] = {}
    for row in rows:
        entry = fbank.setdefault(row["source"], {"clips": 0, "max_abs": 0.0,
                                                 "shape_mismatches": 0})
        entry["clips"] += 1
        value = sherpa_rows[row["key"]]["fbank_max_abs"]
        if value is None:
            entry["shape_mismatches"] += 1
        else:
            entry["max_abs"] = max(entry["max_abs"], value)

    record = {
        "schema_version": 1,
        "issue": 245,
        "gate": "sherpa-parity",
        "rule": {
            "identical_percent": IDENTICAL_PERCENT,
            "unit_edits_per_mille": EDITS_PER_MILLE,
            "phone_character_per_mille": CHARACTER_PER_MILLE,
            "per_source": True,
            "padded_final_chunk": "a non-blank token the padded call emits makes the clip "
                                  "differing (rule 4)",
            "probe_checks": list(PROBE_CHECKS),
            "frame_allowance": "a start frame may shift only where no sampled cut changes "
                               "(astra r1 §3)",
        },
        "ours": _read_json(ours_dir / "ours-protocol.json")["ours"],
        "sherpa": _read_json(ours_dir / "sherpa-protocol.json"),
        "audio": {
            "clips": len(rows),
            "sample_rates": dict(Counter(row["sample_rate"] for row in rows)),
            "channels": dict(Counter(row["channels"] for row in rows)),
            "readers": dict(Counter(row["reader"] for row in rows)),
            "direct_read_matches": len(rows) - len(waveform_mismatches),
            "per_clip": "audio_sha256, waveform_sha256, channels, num_samples and num_frames "
                        "are in parity_decodes.jsonl.gz",
        },
        "sources": sources,
        "pooled": pooled,
        "padded_calls": {
            "clips": sum(1 for row in rows if row["padded_tokens"]),
            "tokens": sum(len(row["padded_tokens"]) for row in rows),
            "call_counts_agree": not call_mismatches,
        },
        "fbank": fbank,
        "phone_table_hits": _phone_table_hits(rows, sherpa_rows, work_dir),
        "probe": probe,
        "explanatory": {
            "v1_0_reference_check": "not run here and not gated: the card-explanation check is "
                                    "explanatory only (parity-gate.md rule 8) and is recorded "
                                    "by the readout",
        },
    }
    _write_jsonl_gz(decodes, ({
        "key": row["key"],
        "source": row["source"],
        "audio_sha256": row["audio_sha256"],
        "waveform_sha256": row["waveform_sha256"],
        "channels": row["channels"],
        "num_samples": row["num_samples"],
        "num_frames": row["num_frames"],
        "identical": identical_clip(row["ours"], sherpa_rows[row["key"]]["tokens"],
                                    row["padded_tokens"]),
        "ours": row["ours"],
        "sherpa": sherpa_rows[row["key"]]["tokens"],
        "calls": row["calls"],
        "sherpa_calls": sherpa_rows[row["key"]]["calls"],
        "padded_tokens": row["padded_tokens"],
        "fbank_max_abs": sherpa_rows[row["key"]]["fbank_max_abs"],
    } for row in rows))
    try:
        require_parity(record)
    except ParityStop as stop:
        write_json({**record, "passed": False, "stop": str(stop)}, out)
        raise
    write_json({**record, "passed": True}, out)
    return record


# --- the command line ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run one phase; a gate that does not pass writes its record and exits 3."""
    parser = argparse.ArgumentParser(description="The #245 sherpa-onnx parity gate.")
    phases = parser.add_subparsers(dest="phase", required=True)
    default_workers = min(12, os.cpu_count() or 1)

    own = phases.add_parser("ours", help="our decoder's side of the comparison")
    own.add_argument("--work-dir", type=Path, required=True)
    own.add_argument("--out-dir", type=Path, required=True)
    own.add_argument("--workers", type=int, default=default_workers)
    own.add_argument("--limit", type=int, default=None,
                     help="decode only the plan's first N clips (a smoke run)")

    theirs = phases.add_parser("sherpa", help="sherpa-onnx's side of the comparison")
    theirs.add_argument("--ours-dir", type=Path, required=True)
    theirs.add_argument("--model", type=Path, required=True)
    theirs.add_argument("--tokens", type=Path, required=True)
    theirs.add_argument("--workers", type=int, default=default_workers)

    judged = phases.add_parser("verdict", help="apply the rule and write the record")
    judged.add_argument("--ours-dir", type=Path, required=True)
    judged.add_argument("--work-dir", type=Path, required=True)
    judged.add_argument("--out", type=Path, default=PARITY_PATH)
    judged.add_argument("--decodes", type=Path, default=PARITY_DIR / DECODES_NAME)

    args = parser.parse_args(argv)
    if args.phase == "ours":
        ours(args.work_dir, args.out_dir, args.workers, limit=args.limit)
        return 0
    if args.phase == "sherpa":
        sherpa(args.ours_dir, args.model, args.tokens, args.workers)
        return 0
    try:
        record = verdict(args.ours_dir, args.work_dir, args.out, args.decodes)
    except ParityStop as stop:
        print(f"parity: STOP: {stop}", flush=True)
        return 3
    print(f"parity: PASSED, {record['pooled']['identical']}/{record['pooled']['clips']} clips "
          f"identical -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
