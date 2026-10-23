"""Score the base teacher and ``h448`` on every committed truth site, in one command (#84, #90).

This is the runner around :mod:`training.truth_scorer`:

1. **Sites.** Every truth-site file ``tadabur/truth_sites/<name>.jsonl`` (a name without a
   dot, so ``p35_fixtures.relocation.jsonl`` is not one), loaded and validated by
   :func:`tadabur.truth_sites.load_truth_sites`, with the listening session's verdicts applied
   by site id (:func:`tadabur.listening_session.adjudicated`, which resolves the P3.5
   re-listens), each site's canonical reciter id from the staged-clip registry, and the
   verified sites' design weights (:mod:`tadabur.ship_a_verified`). Ship A's word sample with
   its whole-word verdicts (:func:`tadabur.word_sample.verified_words`) is decoded and graded
   alongside.
2. **Decodes.** Each model decodes every item (a site's or sampled word's clip and sample
   span) two ways,
   through :class:`training.decoding.Decoder`: the span **whole**, and the deployed
   **stream at b = 0** (``confirmed-stream-v2-flush``: 5 s windows from the item's first
   sample, each normalized on its own, 1 s hop, block 0 committed, the first window's startup
   rule and the last window's flush). Both models are decoded with bf16 weights at batch size
   1, so the two arms share one fingerprint but the model, and batch size 1 keeps every item's
   decode independent of what it was batched with. The item's audio is read from the staged
   WAV only after its checksum and length match the registry. Decodes are cached in
   :data:`DECODES_DIR`, one file per model with its fingerprints and the **identity** of its
   weights (a checkpoint's SHA-256, or the hub commit the teacher resolved to). Every run
   checks the requested model and protocols against the cache; on a re-run only items missing
   from the cache are decoded, and only once the loaded weights prove to be the cache's, so
   new truth sites cost only their own audio and a run with nothing new needs no GPU.
3. **Check sequences** (#125). Each model also records every item's **check sequence**: the
   updates Muraja's transcriber would hand its engine over the item (each full window's hop and
   overlap, each 200 ms preview, and the stop's decode), one inference per check, featurised and
   decoded as the app does (:meth:`training.decoding.Decoder.app_checks`, whose docstring cites
   the transcriber and states what it bypasses). They are cached in :data:`CHECKS_DIR` under
   the same identity and fingerprint rules (:data:`CHECK_STORE`), and make a third arm per
   model, ``checks``, whose decode is the transcript the app commits from them.
4. **App outcomes.** Every item, per arm, replayed through Muraja's own engine under every
   scoring of :data:`training.muraja_policy.SCORINGS` (:func:`app_outcomes`): a ``checks`` arm
   on its recorded check sequence, a ``spans`` or ``stream_b0`` arm on the checks its one decode
   stands for (the single-decode approximation, kept as the fallback). This needs the harness
   ``tools/muraja_harness`` built on this machine (a Mac: ``bash tools/muraja_harness/build.sh``);
   the decodes come from the GPU box.
5. **Report.** :func:`training.truth_scorer.score` over every site and arm, less the items
   the shared exclusion takes out (:func:`tadabur.exclusions.current`: waqf realized, #134;
   onset trimmed, #138; not Hafs, #90), written as
   :data:`REPORT_PATH`; the §8 power simulation's inputs as :data:`POWER_INPUTS_PATH`
   (base and h448 only); and the human-readable report :data:`DOC_PATH`. With it (#90): the
   word sample's false flags and their causes (:mod:`training.word_flags`); per frame and half
   of the verified sites, their app outcome and per-allowance view, and the decode-level
   direction table per direction, stratum and base-teacher outcome with :data:`ARM_GAPS`
   (:func:`verified_views`, :mod:`training.direction_table`); and the synthetic-edit blind check
   per operation with #90's call (:mod:`tadabur.edit_check_summary`).

Every site appears under every arm, so a comparison between protocols or models is always
paired on the same sites (§9 streaming pairing): an item one arm could not decode is an
error, never a dropped site.

Usage (from ``tools/``)::

    # On the GPU box, with the staged clips (decodes only what the caches lack); Ship A's
    # mining extension stages its clips apart, so give one directory holding both:
    python -m training.truth_baseline --audio-dir <clips> --decode-only
    # On a Mac with the Muraja harness built, torch-free, from the committed decodes:
    python -m training.truth_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tadabur.edit_check_summary import summarize as summarize_edit_check
from tadabur.edit_check_summary import type_verdicts
from tadabur.exclusions import Exclusions, current
from tadabur.listening_session import adjudicated, read_verdicts
from tadabur.ship_a import HALVES
from tadabur.ship_a_verified import FRAMES, Design, load_designs
from tadabur.staged_audio import REGISTRY_PATH, StagedClip, load_staged_clips
from tadabur.synthetic_edits import EDITS_PATH, read_items
from tadabur.synthetic_edits import WORKLIST_PATH as BLIND_CHECK_PATH
from tadabur.truth_sites import TRUTH_SITES_DIR, TruthSite, load_truth_sites
from tadabur.word_sample import SampledWord, VerifiedWord, load_words, verified_words
from training import direction_table
from training.decoding import (
    APP_CHECKS,
    INFERENCE_POLICY,
    SPANS,
    DecodeFingerprint,
    Decoder,
    committed_text,
    stream_protocol,
)
from training.muraja_policy import (
    RECORDED_CHECKS,
    SCORINGS,
    SINGLE_DECODE,
    Harness,
    Item,
    MurajaText,
    SiteGrade,
    grade_items,
    require_harness,
    single_decode_checks,
)
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline_doc import render
from training.truth_scorer import (
    MurajaGrades,
    SiteRow,
    item_key,
    power_inputs,
    score,
    scored_sites,
)
from training.word_flags import word_report

BASELINE_DIR = Path(__file__).parent.parent / "tadabur" / "truth_baseline"
DECODES_DIR = BASELINE_DIR / "decodes"
CHECKS_DIR = BASELINE_DIR / "checks"
REPORT_PATH = BASELINE_DIR / "report.json"
POWER_INPUTS_PATH = BASELINE_DIR / "power_inputs.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "truth-baseline.md"

#: The two models this baseline is about (§8: never a candidate), by the references their
#: committed decodes record; extending the cache requires the same reference.
DEFAULT_MODELS = {
    "base": "obadx/muaalem-model-v3_2",
    "h448": "/root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt",
}
WEIGHTS_DTYPE = "bf16"
BATCH_SIZE = 1
#: bf16 weights decode only on CUDA, under its bf16 autocast (``training.decoding``).
DEVICE_TYPE = "cuda"
#: Protocol name in an arm -> the decode mode its fingerprint records, for the decodes cache.
PROTOCOLS = {"spans": SPANS, "stream_b0": stream_protocol(0, flush_tail=True)}
#: The arm graded on each item's recorded check sequence; its decode is the transcript the app
#: commits from those checks (``training.decoding.committed_text``).
CHECKS = "checks"
CHECK_PROTOCOLS = {CHECKS: APP_CHECKS}
#: Every protocol an arm can be under.
ARM_PROTOCOLS = {**PROTOCOLS, **CHECK_PROTOCOLS}


#: What an item is decoded for: a truth site, or a sampled word (:mod:`tadabur.word_sample`).
#: Either names its clip, its span and the clip's checksum.
Located = TruthSite | SampledWord


def arm(model: str, protocol: str) -> str:
    return f"{model}/{protocol}"


def comparisons(models: Sequence[str]) -> list[tuple[str, str]]:
    """h448 − base under each protocol; per model, stream − spans, and the recorded check
    sequence − the stream (the move from the single-decode approximation)."""
    pairs = []
    if {"h448", "base"} <= set(models):
        pairs += [(arm("h448", p), arm("base", p)) for p in ARM_PROTOCOLS]
    pairs += [(arm(m, "stream_b0"), arm(m, "spans")) for m in models]
    pairs += [(arm(m, CHECKS), arm(m, "stream_b0")) for m in models]
    return pairs


def truth_site_files(directory: Path = TRUTH_SITES_DIR) -> list[Path]:
    """Every truth-site file: ``<name>.jsonl`` with no dot in the name."""
    return sorted(p for p in directory.glob("*.jsonl") if "." not in p.stem)


def load_sites(paths: Sequence[Path]) -> list[TruthSite]:
    sites = [site for path in paths for site in load_truth_sites(path)]
    ids = [site.site_id for site in sites]
    if len(set(ids)) != len(ids):
        raise ValueError("a site id appears in more than one truth-site file")
    return sites


def items_of(
    sites: Sequence[Located], registry: Mapping[str, StagedClip]
) -> dict[str, Located]:
    """One representative site (or sampled word) per item; every item's clip must be staged
    as recorded."""
    items: dict[str, Located] = {}
    for site in sites:
        clip = registry.get(site.audio_filename)
        if clip is None or clip.audio_sha256 != site.audio_sha256:
            raise ValueError(f"{site.site_id}: its clip is not in the registry with its checksum")
        items.setdefault(item_key(site), site)
    return items


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: The weight selector a checkpoint cache records: one checkpoint holds the live weights and
#: the averaged ones, so its SHA-256 names neither.
LIVE, EMA = "live", "ema"


def model_identity(model_ref: str, decoder=None, use_ema: bool = False) -> dict:
    """The immutable identity of the weights behind ``model_ref``.

    A checkpoint file is its SHA-256; the averaged selector is recorded when the caller loaded
    it, because the same file holds live and averaged weights and the hash cannot say which.
    Live stays **unrecorded**: every historical cache and external identity carries no
    selector, and a selector-free record means live, so extending or resuming them still
    matches. A hub model is the commit its loaded config resolved to
    (``config._commit_hash``), which needs the loaded ``decoder``.
    """
    path = Path(model_ref)
    if path.is_file():
        identity = {"model_ref": model_ref, "checkpoint_sha256": file_sha256(path)}
        if decoder is not None and use_ema:
            identity["weights"] = EMA
        return identity
    if decoder is None:
        raise ValueError(f"{model_ref}: a hub model's revision is read from its loaded config")
    revision = getattr(decoder.model.config, "_commit_hash", None)
    if not revision:
        raise ValueError(f"{model_ref} resolved to no hub revision; refusing an unbound cache")
    return {"model_ref": model_ref, "hub_revision": revision}


def expected_fingerprint(model_ref: str, mode: str) -> dict:
    """The complete fingerprint every decode of ``model_ref`` in decode ``mode`` must carry:
    the frozen dtype, batch size, device, autocast and inference policy of this baseline."""
    return DecodeFingerprint(
        model=model_ref,
        mode=mode,
        weights_dtype=WEIGHTS_DTYPE,
        batch_size=BATCH_SIZE,
        device_type=DEVICE_TYPE,
        autocast=DEVICE_TYPE == "cuda",
        policy=INFERENCE_POLICY,
    ).as_dict()


@dataclass(frozen=True)
class Store:
    """One kind of per-model cache: its directory, the protocols it holds (name -> decode
    mode), how one item's audio is decoded into them, and whether each item is written on one
    line (a check sequence runs to hundreds of checks)."""

    directory: Path
    protocols: Mapping[str, str]
    decode: Callable[[Decoder, np.ndarray], dict]
    one_item_per_line: bool = False

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.json"


def _spans_and_stream(decoder: Decoder, span: np.ndarray) -> dict:
    return {
        "spans": decoder.decode_spans([span])[0],
        "stream_b0": decoder.decode_stream(span, block=0, flush_tail=True),
    }


def _check_sequence(decoder: Decoder, span: np.ndarray) -> dict:
    return {CHECKS: decoder.app_checks(span)}


#: Each item decoded whole and streamed at b = 0.
DECODE_STORE = Store(DECODES_DIR, PROTOCOLS, _spans_and_stream)
#: Each item's recorded check sequence (``training.decoding.Decoder.app_checks``).
CHECK_STORE = Store(CHECKS_DIR, CHECK_PROTOCOLS, _check_sequence, one_item_per_line=True)


def read_cache(path: Path) -> dict:
    if not path.exists():
        return {"identity": None, "fingerprints": None, "items": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def cache_text(cache: dict, one_item_per_line: bool) -> str:
    """The cache as JSON: indented, or with each item on one line of its own."""
    if not one_item_per_line:
        return json.dumps(cache, ensure_ascii=False, indent=1) + "\n"
    head = json.dumps({k: v for k, v in cache.items() if k != "items"}, ensure_ascii=False, indent=1)
    items = ",\n".join(
        f"  {json.dumps(key, ensure_ascii=False)}: "
        f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for key, value in cache["items"].items()
    )
    return head.removesuffix("\n}") + ',\n "items": {\n' + items + "\n }\n}\n"


def check_cache(path: Path, cache: dict, model_ref: str, protocols: Mapping[str, str]) -> None:
    """Every run: the cache must be this model, under these protocols, from these weights.

    The model reference and every protocol's complete fingerprint are checked without
    loading anything; a checkpoint file present on this machine is re-hashed against the
    cache's identity too.
    """
    if cache["identity"] is None:
        if cache["items"]:
            raise ValueError(f"{path} records no model identity; regenerate it")
        return
    if cache["identity"]["model_ref"] != model_ref:
        raise ValueError(
            f"{path} holds decodes of {cache['identity']['model_ref']!r}, not {model_ref!r}"
        )
    for protocol, mode in protocols.items():
        found, expected = cache["fingerprints"][protocol], expected_fingerprint(model_ref, mode)
        if found != expected:
            raise ValueError(f"{path}: {protocol} was decoded as {found}, expected {expected}")
    # The file hash only: the cache's own identity records which selector it decoded, and both
    # selectors live in the same bytes.
    if Path(model_ref).is_file() and model_identity(model_ref)["checkpoint_sha256"] \
            != cache["identity"].get("checkpoint_sha256"):
        raise ValueError(f"{model_ref} no longer has the bytes {path} was decoded from")


def decode_items(
    decoder: Decoder,
    items: Mapping[str, Located],
    registry: Mapping[str, StagedClip],
    audio_dir: Path,
    decode: Callable[[Decoder, np.ndarray], dict],
) -> dict[str, dict]:
    """Each item's span of its staged clip, decoded by ``decode``."""
    from tadabur.staged_audio import verify_staged
    from training.decode_evalset import read_clip_audio

    by_clip: dict[str, list[tuple[str, Located]]] = {}
    for key, site in sorted(items.items()):
        by_clip.setdefault(site.audio_filename, []).append((key, site))
    decoded: dict[str, dict] = {}
    for clip_name, clip_items in sorted(by_clip.items()):
        verify_staged(registry[clip_name], audio_dir)  # decode only the recorded bytes
        samples = read_clip_audio(audio_dir, clip_name)
        for key, site in clip_items:
            span = samples[site.start_sample : site.end_sample]
            decoded[key] = {"audio_sha256": site.audio_sha256, **decode(decoder, span)}
    return decoded


def update_cache(
    store: Store,
    name: str,
    model_ref: str,
    items: Mapping[str, Located],
    registry: Mapping[str, StagedClip],
    audio_dir: Path | None,
    use_ema: bool = False,
) -> dict:
    """The model's cache in ``store``, checked, and extended with any item it lacks.

    New decodes are merged only when the loaded weights' identity (:func:`model_identity`)
    and every fingerprint match the cache's, so one cache never mixes two checkpoints.
    ``use_ema`` decodes a checkpoint's averaged weights and records that selector.
    """
    path = store.path(name)
    cache = read_cache(path)
    check_cache(path, cache, model_ref, store.protocols)
    stale = [k for k, v in cache["items"].items() if k in items and v["audio_sha256"] != items[k].audio_sha256]
    if stale:
        raise ValueError(f"{path}: {len(stale)} cached item(s) were decoded from other audio")
    missing = {k: site for k, site in items.items() if k not in cache["items"]}
    if not missing:
        return cache
    if audio_dir is None:
        raise SystemExit(f"{path}: {len(missing)} item(s) have no decode; pass --audio-dir")
    import torch

    print(f"{path}: decoding {len(missing)} item(s) with {model_ref}", flush=True)
    decoder = Decoder.load(
        model_ref, DEVICE_TYPE, weights_dtype=WEIGHTS_DTYPE, batch_size=BATCH_SIZE,
        use_ema=use_ema,
    )
    identity = model_identity(model_ref, decoder, use_ema)
    fingerprints = {p: decoder.fingerprint(mode).as_dict() for p, mode in store.protocols.items()}
    for protocol, record in fingerprints.items():
        if record != expected_fingerprint(model_ref, store.protocols[protocol]):
            raise ValueError(f"{model_ref} decodes as {record}, not the frozen protocol")
    if cache["identity"] is not None and identity != cache["identity"]:
        raise ValueError(f"{model_ref} is {identity}, but {path} was decoded from {cache['identity']}")
    decoded = decode_items(decoder, missing, registry, audio_dir, store.decode)
    del decoder
    torch.cuda.empty_cache()
    cache = {
        "model": name,
        "identity": identity,
        "fingerprints": cache["fingerprints"] or fingerprints,
        "items": dict(sorted({**cache["items"], **decoded}.items())),
    }
    write_text_atomically(path, cache_text(cache, store.one_item_per_line))
    return cache


def update_decodes(
    name: str,
    model_ref: str,
    items: Mapping[str, Located],
    registry: Mapping[str, StagedClip],
    audio_dir: Path | None,
    use_ema: bool = False,
) -> dict:
    """The model's decodes cache (:data:`DECODE_STORE`), extended as :func:`update_cache` says."""
    return update_cache(DECODE_STORE, name, model_ref, items, registry, audio_dir, use_ema)


def model_caches(
    name: str,
    model_ref: str,
    items: Mapping[str, Located],
    registry: Mapping[str, StagedClip],
    audio_dir: Path | None,
    use_ema: bool = False,
) -> tuple[dict, dict]:
    """The model's decodes and check sequences, each checked and extended as
    :func:`update_cache` says, and refused when the two come from different weights. With no
    ``audio_dir`` it only reads: every item must already be decoded, from its own audio."""
    cache = update_decodes(name, model_ref, items, registry, audio_dir, use_ema)
    recorded = update_cache(CHECK_STORE, name, model_ref, items, registry, audio_dir, use_ema)
    if recorded["identity"] != cache["identity"]:
        raise ValueError(f"{name}: its decodes and check sequences come from different weights")
    return cache, recorded


@dataclass(frozen=True)
class KeptGrades:
    """Every site's and sampled word's kept grade as ``[scoring][arm][site or word id]``, and
    the phonemes of each placed one's word on Muraja's reference (what
    :mod:`training.word_flags` attributes a flag with)."""

    grades: Mapping[str, Mapping[str, Mapping[str, SiteGrade]]]
    texts: Mapping[str, str]


def app_outcomes(
    sites: Sequence[TruthSite],
    words: Sequence[VerifiedWord],
    decodes: Mapping[str, Mapping[str, str]],
    sequences: Mapping[str, Mapping[str, Sequence[dict]]],
    harness: Harness,
) -> tuple[MurajaGrades, KeptGrades]:
    """Every site's app outcome, and every site's and sampled word's kept grade, per scoring and
    arm, replayed through Muraja's engine (``training.muraja_policy``): an arm in ``sequences``
    on each item's recorded check sequence, any other on the checks its one decode stands for
    (single-decode approximation). A word is graded in the same session as the sites of its
    item."""
    by_item: dict[str, list[Located]] = {}
    for site in [*sites, *(word.site for word in words)]:
        by_item.setdefault(item_key(site), []).append(site)

    def checks(arm: str, key: str, item_sites: list[Located]) -> tuple[dict, ...]:
        if arm in sequences:
            return tuple(sequences[arm][key])
        n_samples = item_sites[0].end_sample - item_sites[0].start_sample
        return tuple(single_decode_checks(decodes[arm][key], n_samples))

    items = [
        Item(f"{arm}#{key}", tuple(item_sites), checks(arm, key, item_sites))
        for arm in sorted(decodes)
        for key, item_sites in sorted(by_item.items())
    ]
    graded = grade_items(items, SCORINGS, harness)
    word_ids = {word.site.site_id for word in words}
    outcomes: dict[str, dict[str, dict[str, str]]] = {
        scoring: {arm: {} for arm in decodes} for scoring in graded.grades}
    kept: dict[str, dict[str, dict[str, SiteGrade]]] = {
        scoring: {arm: {} for arm in decodes} for scoring in graded.grades}
    for scoring, by_site in graded.grades.items():
        for (key, site_id), grade in by_site.items():
            arm_ = key.split("#", 1)[0]
            kept[scoring][arm_][site_id] = grade
            if site_id not in word_ids:
                outcomes[scoring][arm_][site_id] = grade.outcome
    approximations = {arm: RECORDED_CHECKS if arm in sequences else SINGLE_DECODE for arm in decodes}
    muraja = MurajaGrades(
        outcomes, SCORINGS, graded.build, approximations,
        tuple(i for i in graded.unplaced if i not in word_ids),
        tuple(i for i in graded.unattributed if i not in word_ids),
    )
    text = MurajaText(harness.quran_db)
    placed = {site.site_id: text.locate(site) for site in [*sites, *(word.site for word in words)]}
    texts = {site_id: text.word_phonemes(where) for site_id, where in placed.items() if where}
    return muraja, KeptGrades(kept, texts)


def p35_reproduction(sites: Sequence[TruthSite], base_spans: Mapping[str, str]) -> dict:
    """How many P3.5 items the base whole-span decode reproduces from the re-location (#83).

    The re-location rule kept a site only where this decode showed the contrast, so an
    identical decode means the base teacher's outcome at those sites was fixed by selection.
    """
    relocation = TRUTH_SITES_DIR / "p35_fixtures.relocation.jsonl"
    decode_of_site: dict[str, str] = {}
    for line in relocation.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for site_id in row["site_ids"]:
            decode_of_site[site_id] = row["decode"]
    keys = {item_key(s): decode_of_site[s.site_id] for s in sites if s.site_id in decode_of_site}
    same = sum(base_spans[key] == decode for key, decode in keys.items())
    return {"items": len(keys), "identical": same}


def scored(
    sites: Sequence[TruthSite],
    reciter_of: Mapping[str, int],
    decodes: Mapping[str, Mapping[str, str]],
    models: Sequence[str],
    muraja: MurajaGrades,
    excluded_items: Exclusions,
    design_weights: Mapping[str, float] | None = None,
) -> tuple[dict, dict]:
    """The report over every arm and the §8 power inputs (base and ``h448`` only), both less
    the items the shared exclusion takes out, with weights computed over every site (the
    verified sites' design weights, :mod:`tadabur.ship_a_verified`)."""
    report = score(sites, reciter_of, decodes, comparisons(models), muraja,
                   design_weights=design_weights, excluded_items=excluded_items)
    baseline_arms = {a: d for a, d in decodes.items() if a.split("/")[0] in ("base", "h448")}
    inputs = power_inputs(sites, reciter_of, baseline_arms, muraja, excluded_items, design_weights)
    inputs["today_system_arm"] = arm("h448", "stream_b0")
    return report, inputs


#: The paired differences each verified-site row reports (#90): ``h448`` streamed against the
#: base whole (the sizing's "base already right"), split into the weights (both whole) and
#: ``h448``'s windowing (its stream against its whole decode); the base's own windowing; and the
#: two models on the app's checks.
ARM_GAPS = (
    (arm("h448", "stream_b0"), arm("base", "spans")),
    (arm("h448", "spans"), arm("base", "spans")),
    (arm("h448", "stream_b0"), arm("h448", "spans")),
    (arm("base", "stream_b0"), arm("base", "spans")),
    (arm("h448", CHECKS), arm("base", CHECKS)),
)


def _with_gaps(entry: dict, rows: Sequence[SiteRow], gaps: Sequence[tuple[str, str]]) -> dict:
    """An entry of the direction table with each :data:`ARM_GAPS` difference over its rows."""
    if rows:
        entry["gaps"] = {f"{a} - {b}": direction_table.difference(rows, a, b) for a, b in gaps}
    return entry


def verified_views(
    sites: Sequence[TruthSite],
    reciter_of: Mapping[str, int],
    decodes: Mapping[str, Mapping[str, str]],
    muraja: MurajaGrades,
    designs: Mapping[str, Design],
    excluded_items: Exclusions,
) -> dict:
    """The verified sites per frame and half (:mod:`tadabur.ship_a_verified`), which were
    selected by different rules and are never pooled here:

    * ``app``: :func:`training.truth_scorer.score` over that subset alone, its pooled cells
      (the app outcome per side, arm and scoring) and its per-allowance view;
    * ``directions``: the direction table (:mod:`training.direction_table`);
    * ``strata``: the same rates per recorded stratum and side;
    * ``teacher``: per direction, base-teacher outcome of the stratum and side.

    Every direction entry carries :data:`ARM_GAPS` (when both models are scored)."""
    weights = {site_id: d.weight for site_id, d in designs.items()}
    gaps = [g for g in ARM_GAPS if set(g) <= set(decodes)]
    arms = sorted(decodes)
    views: dict[str, dict[str, dict]] = {}
    for frame in FRAMES:
        views[frame] = {}
        for half in HALVES:
            subset = [s for s in sites if s.site_id in designs
                      and (designs[s.site_id].frame, designs[s.site_id].half) == (frame, half)]
            report = score(subset, reciter_of, decodes, (), muraja, design_weights=weights,
                           excluded_items=excluded_items)
            rows = scored_sites(subset, reciter_of, decodes, design_weights=weights,
                                excluded_items=excluded_items).rows
            views[frame][half] = {
                "app": {"cells": [c for c in report["cells"] if c["family"] == "all" and c["sites"]],
                        "allowances": report["allowances"]},
                "directions": [
                    _with_gaps(entry, direction_table.members(rows, key), gaps)
                    for entry, key in zip(direction_table.table(rows, arms), direction_table.KEYS)],
                "strata": [
                    _with_gaps({"stratum": label, "side": side_,
                                **direction_table.summary(held, arms)}, held, gaps)
                    for (label, side_), held in direction_table.groups(
                        rows, lambda r: r.site.stratum).items()],
                "teacher": [
                    _with_gaps({"direction": label.split(" ")[0], "teacher": label.split(" ")[1],
                                "side": side_, **direction_table.summary(held, arms)}, held, gaps)
                    for (label, side_), held in direction_table.groups(
                        rows, lambda r: f"{direction_table.direction(r.site)} "
                                        f"{direction_table.teacher_outcome(r.site)}").items()],
            }
    return views


def edit_check() -> dict:
    """The synthetic-edit blind check per operation (``tadabur.edit_check_summary``), with
    #90's call on each type."""
    summary = summarize_edit_check(load_truth_sites(BLIND_CHECK_PATH), read_items(EDITS_PATH),
                                   read_verdicts(), current())
    summary["type_verdicts"] = type_verdicts(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, help="staged 16 kHz clips; needed only to decode")
    parser.add_argument(
        "--model", action="append", default=[], metavar="NAME=REF",
        help="a model to score (default: base and h448); its decodes cache under NAME",
    )
    parser.add_argument(
        "--decode-only", action="store_true",
        help="fill the caches and stop: the GPU box has no Muraja harness to grade with",
    )
    args = parser.parse_args()
    models = dict(m.split("=", 1) for m in args.model) or DEFAULT_MODELS

    files = truth_site_files()
    verdicts = read_verdicts()
    sites = adjudicated(load_sites(files), verdicts)
    words = verified_words(load_words(), verdicts)
    registry = load_staged_clips(REGISTRY_PATH)
    items = items_of([*sites, *(word.site for word in words)], registry)
    print(f"{len(sites)} sites and {len(words)} sampled words on {len(items)} items from "
          f"{[p.name for p in files]}", flush=True)

    decodes: dict[str, dict[str, str]] = {}
    sequences: dict[str, dict[str, list[dict]]] = {}
    fingerprints: dict[str, dict] = {}
    for name, ref in models.items():
        cache, recorded = model_caches(name, ref, items, registry, args.audio_dir)
        fingerprints[name] = {**cache["fingerprints"], **recorded["fingerprints"]}
        for protocol in PROTOCOLS:
            decodes[arm(name, protocol)] = {k: v[protocol] for k, v in cache["items"].items() if k in items}
        sequences[arm(name, CHECKS)] = {k: v[CHECKS] for k, v in recorded["items"].items() if k in items}
        decodes[arm(name, CHECKS)] = {k: committed_text(c) for k, c in sequences[arm(name, CHECKS)].items()}
    if args.decode_only:
        print(f"Caches complete for {sorted(models)}; grade them where the Muraja harness is built")
        return

    reciter_of = {name: clip.reciter_id for name, clip in registry.items()}
    muraja, kept = app_outcomes(sites, words, decodes, sequences, require_harness())
    excluded, designs = current(), load_designs()
    weights = {site_id: d.weight for site_id, d in designs.items()}
    report, inputs = scored(sites, reciter_of, decodes, list(models), muraja, excluded, weights)
    report["word_sample"] = word_report(words, kept.grades, kept.texts, excluded)
    report["verified"] = verified_views(sites, reciter_of, decodes, muraja, designs, excluded)
    report["edit_check"] = edit_check()
    report["truth_site_files"] = [p.name for p in files]
    report["fingerprints"] = fingerprints
    report["protocols"] = ARM_PROTOCOLS
    if "base" in models:
        report["p35_base_reproduction"] = p35_reproduction(sites, decodes[arm("base", "spans")])
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(POWER_INPUTS_PATH, json.dumps(inputs, ensure_ascii=False, indent=1) + "\n")

    write_text_atomically(DOC_PATH, render(report))
    print(f"Wrote {REPORT_PATH}, {POWER_INPUTS_PATH} and {DOC_PATH}")


if __name__ == "__main__":
    main()
