"""The published mini's false flags on round 3's graded confirm population (reporting only).

The owner asked (2026-11-18) what the **published** ``sysofwan/hifzguide-muaalem-mini`` scores
on round 3's pooled confirm read, so the v2 card can state v2's real false-flag change against
the version the app ships. Round 3 decoded and graded the teacher, C95 and
``r3_candidate_ema`` on confirm.3 + confirm.4 (``docs/consonant-round3-readout.md``); this
module decodes the published mini on exactly those gating rows and nothing else.

It is a **reporting** decode. The round's verdict, readout and pre-registration are inputs to
the report, never re-decided here, and no gate reads this decode. The model is not a confirm
producer, so the decode does not pass
:func:`tadabur.tlog_eval_decode.require_confirm_models`; it reuses that module's committed
manifest selection (:func:`~tadabur.tlog_eval_decode.read_confirm_manifest`, which still
refuses an unclaimed tranche and reads the committed choice), its clip streaming, per-shard
writer, checks store and pooled confirm merge, so the protocol is the round's own.

The mini's identity is its immutable Hub artifact, which the v2 card records
(``docs/release-v1.0.7.md``): the repository, the pinned revision and the
``model.safetensors`` SHA-256. The decoder is loaded the way the repository documents
(``from_pretrained`` at that revision, through the vendored modeling code), and ``verify``
proves that loader and the repository's own ``trust_remote_code`` forward agree -- every weight
tensor and a fixed ``(1, 250, 160)`` window -- before anything is decoded.

Usage, from ``tools/``::

    python -m training.mini_flags plan
    python -m training.mini_flags verify --record <run>/verify.json
    python -m training.mini_flags decode --role confirm.3 --parts-dir <run>/parts \\
        --cache-dir <cache> --shard-index 0 --shard-count 4
    python -m training.mini_flags merge --parts-dir <run>/parts --out-dir <run>/out
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from tadabur.tlog_eval_decode import (RUN_STORES, append_timing, ayah_word_counts,
                                      gating_rows, merge_confirm_decodes, provenance,
                                      read_confirm_manifest, run_shard, shard_stream)
from tadabur.tlog_index import head_sha, write_json
from tadabur.tlog_tranches import claim_record
from training.aug_readout_spec import CONSONANT3
from training.decoding import DecodeFingerprint, Decoder
from training.external_decode import MODES, record_identity
from training.truth_baseline import (BATCH_SIZE, DEVICE_TYPE, WEIGHTS_DTYPE, file_sha256)

#: The published model this measurement is of, at the identity the v2 card pins: the Hub
#: repository, the revision it was measured at, and the weights file's own SHA-256.
MINI = "mini"
MINI_REF = "sysofwan/hifzguide-muaalem-mini"
MINI_REVISION = "a2bebf7c2e9a74d9ff5c3269594132d7caea7696"
MINI_WEIGHTS = "model.safetensors"
MINI_WEIGHTS_SHA256 = "2570fd3780a8435a11eb20e336bac29ac6f4dedd7c367d39abc20627e6784bb9"
#: The fixed feature window ``verify`` runs both loaders on: 5 s of deployed features, the
#: shape the app's forward takes. The seed makes the window reproducible.
WINDOW_FRAMES, WINDOW_DIM, WINDOW_SEED = 250, 160, 0


def weights_path() -> Path:
    """The pinned weights file, downloaded at the pinned revision into the Hub cache."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MINI_REF, MINI_WEIGHTS, revision=MINI_REVISION))


def require_weights(path: Path) -> None:
    """Refuse anything but the weights file the card pins: a moved ``main`` cannot slip in."""
    found = file_sha256(Path(path))
    if found != MINI_WEIGHTS_SHA256:
        raise SystemExit(f"{MINI_REF}@{MINI_REVISION} is {found}, "
                         f"not the pinned {MINI_WEIGHTS_SHA256}")


def expected_fingerprint(mode: str) -> dict:
    """The fingerprint a decode of the pinned mini must carry: the round's own numerics."""
    return DecodeFingerprint(
        model=f"{MINI_REF}@{MINI_REVISION}", mode=mode, weights_dtype=WEIGHTS_DTYPE,
        batch_size=BATCH_SIZE, device_type=DEVICE_TYPE, autocast=DEVICE_TYPE == "cuda",
    ).as_dict()


def mini_decoder() -> Decoder:
    """The published mini at its pinned revision, over the weights the card pins.

    ``Decoder.load`` resolves a Hub id through the vendored modeling code and its published
    feature extractor, at the revision given, so the loaded config must resolve to that same
    revision. The weights file is hashed before any forward.
    """
    require_weights(weights_path())
    decoder = Decoder.load(MINI_REF, DEVICE_TYPE, weights_dtype=WEIGHTS_DTYPE,
                           batch_size=BATCH_SIZE, revision=MINI_REVISION)
    revision = getattr(decoder.model.config, "_commit_hash", None)
    if revision != MINI_REVISION:
        raise SystemExit(f"{MINI_REF} resolved to {revision}, not the pinned {MINI_REVISION}")
    return decoder


def require_protocol(decoder) -> None:
    """Refuse a decoder whose decode would not be the round's: every mode's fingerprint must
    be the pinned model at the round's dtype, batch size, device and inference policy."""
    for arm, mode in MODES.items():
        found = decoder.fingerprint(mode).as_dict()
        if found != expected_fingerprint(mode):
            raise SystemExit(f"{MINI} decodes {arm} as {found}, "
                             f"not {expected_fingerprint(mode)}")


def decoder_identity(decoder) -> dict:
    """The identity record written beside the parts, in ``load_decoders``' shape, so the
    pooled merge cites the weights and numerics that actually decoded the tranches."""
    return {"model": MINI,
            "identity": {"model_ref": MINI_REF, "hub_revision": MINI_REVISION,
                         "safetensors_sha256": MINI_WEIGHTS_SHA256},
            "fingerprints": {arm: decoder.fingerprint(mode).as_dict()
                             for arm, mode in MODES.items()}}


def verify() -> dict:
    """Prove the pinned artifact and the vendored loader equal the published model.

    Loads the repository twice over the same revision -- through its own ``trust_remote_code``
    class, and through the vendored class every decode tool uses -- and refuses unless the two
    state dicts are equal and both forwards agree on a fixed deployed-shape window. Then loads
    the decode's own decoder and checks its protocol fingerprints.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    from tadabur.muaalem import Wav2Vec2BertForMultilevelCTC
    from training.distill_student import PHONEME_LEVEL

    weights = Path(hf_hub_download(MINI_REF, MINI_WEIGHTS, revision=MINI_REVISION))
    require_weights(weights)
    published = AutoModel.from_pretrained(MINI_REF, revision=MINI_REVISION,
                                          trust_remote_code=True).eval()
    vendored = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        MINI_REF, revision=MINI_REVISION).eval()
    published_state, vendored_state = published.state_dict(), vendored.state_dict()
    if set(published_state) != set(vendored_state):
        differing = sorted(set(published_state) ^ set(vendored_state))
        raise SystemExit(f"the two loaders disagree on {len(differing)} tensors, "
                         f"first {differing[0]}")
    weights_diff = max((float((published_state[name].float()
                               - vendored_state[name].float()).abs().max())
                        for name in published_state), default=0.0)
    torch.manual_seed(WINDOW_SEED)
    window = torch.rand(1, WINDOW_FRAMES, WINDOW_DIM)
    with torch.no_grad():
        published_logits = published(window, return_dict=True)["logits"][PHONEME_LEVEL]
        vendored_logits = vendored(window, return_dict=True)["logits"][PHONEME_LEVEL]
    window_diff = float((published_logits - vendored_logits).abs().max())
    del published, vendored
    if weights_diff or window_diff:
        raise SystemExit(f"the vendored loader differs from the published model: weights "
                         f"{weights_diff}, window {window_diff}")
    decoder = mini_decoder()
    require_protocol(decoder)
    record = {"model": MINI, "model_ref": MINI_REF, "hub_revision": MINI_REVISION,
              "weights": MINI_WEIGHTS, "weights_sha256": MINI_WEIGHTS_SHA256,
              "weights_bytes": weights.stat().st_size,
              "tensors": len(published_state),
              "loader_weights_max_abs_diff": weights_diff,
              "window": {"seed": WINDOW_SEED,
                         "shape": [1, WINDOW_FRAMES, WINDOW_DIM],
                         "max_abs_diff": window_diff},
              "fingerprints": decoder_identity(decoder)["fingerprints"]}
    del decoder
    return record


def plan() -> dict:
    """The committed job: the round's two confirm tranches, their gating shards, the model."""
    roles = CONSONANT3.confirm_tranches
    shards = {role: len({row["shard_file"]
                         for row in gating_rows(read_confirm_manifest(role, spec=CONSONANT3))})
              for role in roles}
    return {"model": MINI, "model_ref": MINI_REF, "hub_revision": MINI_REVISION,
            "weights_sha256": MINI_WEIGHTS_SHA256, "roles": list(roles), "shards": shards}


def decode_role(role: str, *, parts_dir: Path, cache_dir: Path, shard_index: int,
                shard_count: int, clip_cache: Path | None = None) -> dict:
    """Decode one process's share of ``role``'s gating rows with the published mini.

    The rows are the round's committed confirm manifest's gating stratum, read through the
    same function the round's own decode used; the parts land beside them in that decoder's
    names (``tlog.<role>.<shard>.<model>.jsonl``), so a killed run resumes by row key.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"--shard-index must be in [0, {shard_count})")
    rows = gating_rows(read_confirm_manifest(role, spec=CONSONANT3))
    by_shard: dict[str, list] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    shards = sorted(by_shard)[shard_index::shard_count]
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    decoder = mini_decoder()
    require_protocol(decoder)
    record_identity(parts_dir / f"{MINI}.identity.json", decoder_identity(decoder), label=MINI)
    stream = shard_stream(cache_dir)
    word_counts = ayah_word_counts()
    timings = []
    for shard_file in shards:
        timing = {"shard_file": shard_file, **run_shard(
            shard_file, by_shard[shard_file], parts_dir, {MINI: decoder}, stream=stream,
            word_counts=word_counts, stores=RUN_STORES, clip_cache=clip_cache,
            share_features=False, role=role)}
        append_timing(parts_dir / "timing.jsonl", timing)
        timings.append(timing)
    return {"role": role, "shards": len(shards),
            "items": sum(timing["items"] for timing in timings),
            "audio_seconds": round(sum(timing["audio_seconds"] for timing in timings), 1)}


def merge(parts_dir: Path, out_dir: Path, *, pull_request: int | None = None) -> dict:
    """Pool the two tranches' mini parts into one file per the round's merge, with the claims
    the round spent them under. A missing or repeated row is refused by that merge."""
    tranches = CONSONANT3.confirm_tranches
    return merge_confirm_decodes(
        Path(parts_dir), Path(out_dir),
        manifests={role: read_confirm_manifest(role, spec=CONSONANT3) for role in tranches},
        claims={role: claim_record(role) for role in tranches},
        provenance=provenance(head_sha(), pull_request), models=[MINI], spec=CONSONANT3)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan_ = commands.add_parser("plan", help="print the committed job")
    verify_ = commands.add_parser("verify", help="prove the pinned artifact and loader")
    verify_.add_argument("--record", type=Path, required=True,
                         help="where to write the verification record")
    decode_ = commands.add_parser("decode", help="decode one process's share of a tranche")
    decode_.add_argument("--role", required=True, choices=list(CONSONANT3.confirm_tranches))
    decode_.add_argument("--parts-dir", type=Path, required=True)
    decode_.add_argument("--cache-dir", type=Path, required=True)
    decode_.add_argument("--shard-index", type=int, default=0)
    decode_.add_argument("--shard-count", type=int, default=1)
    decode_.add_argument("--clip-cache", type=Path, default=None)
    merge_ = commands.add_parser("merge", help="pool the tranches into one file")
    merge_.add_argument("--parts-dir", type=Path, required=True)
    merge_.add_argument("--out-dir", type=Path, required=True)
    merge_.add_argument("--pr", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "plan":
        print(json.dumps(plan(), sort_keys=True))
    elif args.command == "verify":
        write_json(verify(), args.record)
        print(json.dumps(json.loads(args.record.read_text(encoding="utf-8")), sort_keys=True))
    elif args.command == "decode":
        print(json.dumps(decode_role(
            args.role, parts_dir=args.parts_dir, cache_dir=args.cache_dir,
            shard_index=args.shard_index, shard_count=args.shard_count,
            clip_cache=args.clip_cache), sort_keys=True))
    else:
        print(json.dumps(merge(args.parts_dir, args.out_dir,
                               pull_request=args.pr), sort_keys=True))


if __name__ == "__main__":
    main()
