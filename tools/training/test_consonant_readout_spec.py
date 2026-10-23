"""#95 and the consonant round as two specs over one readout, and the round's pins (#175).

The bars and sets are #174's pre-registration (its own document pin lives in
``test_aug_readout_prereg.py``); this file pins the facts the readout must not invent: which model
is the baseline, what fixes Muraja and the harness, and that C95 is the checkpoint it says it is.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from training import aug_readout_gate as ag
from training import aug_readout_spec as asp

MURAJA_COMMIT = "99c326f3fb7c6c53ff525568976c7e2442ccde3c"
HARNESS_SHA256 = "cfcb99ba583fd2324f7ede0e5ab896379c54fa52783460d583e44d103eb68d8b"
C95_CHECKPOINT = "8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6"
C95_REF = "/root/scratch/issue-94/repo-fix/tools/runs/aug94_candidate_v2/checkpoint.pt"


def test_95_and_the_consonant_round_are_two_specs_over_one_pipeline():
    aug, con = asp.spec("aug95"), asp.spec("consonant1")
    assert (aug.issue, con.issue) == (95, 175)
    assert (aug.baseline, con.baseline) == ("h448", "c95")
    assert {"tarteel_v1.look", "tarteel_v1.confirm", "qdat", "retasy", "qud_half",
            "tadabur_dev_unseen"} <= set(aug.sets)
    assert {"tarteel_v1.look", "tarteel_v1.confirm", "qdat", "qud_other",
            "tadabur_dev_unseen", "ship_a_verified"} <= set(con.sets)
    assert "retasy" not in con.sets  # descriptive only for this round (#171)
    with pytest.raises(KeyError):
        asp.spec("consonant2")


def test_the_consonant_spec_pins_muraja_and_the_harness():
    pins = asp.CONSONANT1.pins
    assert pins.muraja_commit == MURAJA_COMMIT
    assert pins.harness_sha256 == HARNESS_SHA256


def test_confirm_refuses_a_pin_that_differs():
    pins = asp.Pins(muraja_commit=MURAJA_COMMIT, harness_sha256=HARNESS_SHA256,
                    prereg_sha256="a" * 64)
    spec = dataclasses.replace(asp.CONSONANT1, pins=pins)
    harness = {"muraja_commit": MURAJA_COMMIT, "binary_sha256": HARNESS_SHA256}
    asp.check_pins(spec, harness, "a" * 64)  # every pin matches: no refusal
    for observed, prereg in (({**harness, "muraja_commit": "0" * 40}, "a" * 64),
                             ({**harness, "binary_sha256": "0" * 64}, "a" * 64),
                             (harness, "b" * 64)):
        with pytest.raises(SystemExit, match="pin"):
            asp.check_pins(spec, observed, prereg)


def test_the_round_s_baseline_is_the_metrics_constant():
    from training import aug_readout_metrics as am

    assert asp.C95 is am.C95
    assert am.comparison_name(am.CANDIDATE, am.C95) == f"candidate−{asp.C95}"
    assert asp.CONSONANT1.baseline is am.C95


def test_c95_is_the_pinned_checkpoint_the_readout_validates():
    spec = asp.CONSONANT1
    assert asp.C95_SHA256 == C95_CHECKPOINT
    model = ag.model_spec("c95", spec=spec)
    assert (model.ref, model.use_ema, model.step, model.recipe) == (
        C95_REF, True, 10_000, "aug94-mixture-v2")
    record = {"model": "c95", "fingerprints": ag.expected_fingerprints(model.ref),
              "identity": {"model_ref": model.ref, "weights": "ema", "step": 10_000,
                           "recipe": "aug94-mixture-v2",
                           "checkpoint_sha256": C95_CHECKPOINT}}
    assert ag.validate_identity("qdat", "c95", record, spec=spec) == record["identity"]
    for edit, match in (
            (lambda r: r["fingerprints"]["checks"].update(batch_size=8), "fingerprint"),
            (lambda r: r["identity"].update(checkpoint_sha256="b" * 64), "checkpoint")):
        altered = json.loads(json.dumps(record))
        edit(altered)
        with pytest.raises(ValueError, match=match):
            ag.validate_identity("qdat", "c95", altered, spec=spec)


# --- the round's decodes are its own; C95's are #95's immutable inputs (review R1.1) ----------


def test_the_round_reads_c95_s_committed_decodes_and_never_overwrites_them():
    from training import aug_readout as ar

    round_sources = ar.Sources(asp.CONSONANT1)
    for set_name in asp.REUSED_SETS:
        c95 = round_sources.decode_path(set_name, asp.C95)
        assert c95 == ag.decodes_file(set_name, "candidate_ema")  # #95's file, byte for byte
        assert c95.is_file()
        candidate = round_sources.decode_path(set_name, "candidate_ema")
        assert candidate == asp.CONSONANT_ROOT / "decodes" / f"{set_name}.candidate_ema.jsonl.gz"
        assert candidate != c95  # merging the candidate cannot touch #95's input
    # qud_other is the round's own set: C95 is decoded for it and stored in the round's root.
    qud_other = round_sources.decode_path("qud_other", asp.C95)
    assert qud_other == asp.CONSONANT_ROOT / "decodes" / "qud_other.c95.jsonl.gz"
    assert qud_other != round_sources.decode_path("qud_other", "candidate_ema")
    # The baselines are #95's everywhere: no round decodes them.
    for model in ("h448", "base"):
        assert round_sources.decode_path("tadabur_dev_unseen", model) == ag.decodes_file(
            "tadabur_dev_unseen", model)


def test_a_round_s_merge_and_reads_target_that_round_s_root(tmp_path):
    import dataclasses

    from training import aug_readout as ar

    spec = dataclasses.replace(asp.CONSONANT1, root=tmp_path)
    assert ar.Sources(spec).decode_path("qdat", "candidate_ema") == (
        tmp_path / "decodes" / "qdat.candidate_ema.jsonl.gz")
    assert ar.Sources(spec).decode_path("qdat", asp.C95) == (
        asp.AUG95_ROOT / "decodes" / "qdat.candidate_ema.jsonl.gz")


def test_only_the_tracer_shadows_the_candidate_with_c95_s_decodes():
    from training import aug_readout as ar
    from training import consonant_readout as cr

    role = "candidate"  # the tracer reads the candidate by its role, C95 by its name
    production = ar.Sources(asp.CONSONANT1)
    tracer = ar.Sources(cr.tracer_spec())
    for set_name in ("tarteel_v1.look", "qdat", "tadabur_dev_unseen"):
        assert production.decode_path(set_name, role) != production.decode_path(set_name, asp.C95)
        assert tracer.decode_path(set_name, role) == tracer.decode_path(set_name, asp.C95)
    assert not any(role == "candidate" for _, role in asp.CONSONANT1.decode_sources)


def test_the_reproduction_check_refuses_a_later_round_s_sources():
    from training import aug_readout as ar

    with pytest.raises(ValueError, match="#95"):
        ar.reproduction(ar.Sources(asp.CONSONANT1))


def test_merge_never_writes_another_round_s_committed_decodes(tmp_path):
    """The read resolver may point at #95's files; the merge's write may not (review R2.2)."""
    from training import aug_readout as ar
    from training import aug_readout_gate as ag

    pinned = ag.identity_record(asp.CONSONANT1.models[asp.C95])
    gpu = tmp_path / "gpu"
    gpu.mkdir()
    for set_name, model in (("qdat", asp.C95), ("tadabur_dev_unseen", "h448")):
        (gpu / f"{set_name}.{model}.identity.json").write_text(
            json.dumps({"model": model, "fingerprints": {}, "identity": pinned}),
            encoding="utf-8")
    historical = ag.decodes_file("qdat", "candidate_ema")
    before = historical.read_bytes()
    with pytest.raises(SystemExit, match="another round"):
        ar.merge(gpu, ag.Layout(tmp_path / "round"), spec=asp.CONSONANT1)
    assert historical.read_bytes() == before


# --- #179: the #178 candidate is the round's, pinned before any decode --------------------------

#: #178's checkpoint on cuda-dev (``docs/consonant-round-run.md``): one file, live and EMA.
CANDIDATE_REF = ("/root/scratch/issue-178/repo/tools/runs/consonant1_candidate/checkpoint.pt")
CANDIDATE_SHA256 = "057678950541569739d3bb40bed8e8bcf490d97096ea814d14df97627dc214a2"
CANDIDATE_RECIPE = "aug94-mixture-v2"


def test_the_round_pins_the_178_candidate_for_both_weight_selectors():
    """The spec says which checkpoint the round decodes, at which weights, step and recipe: the
    decode run must refuse anything else, so the pin cannot live only in a run note."""
    pins = asp.CONSONANT1.models
    assert set(pins) == {"c95", "candidate_live", "candidate_ema"}
    for model, weights in (("candidate_live", "live"), ("candidate_ema", "ema")):
        assert pins[model] == asp.ModelIdentity(CANDIDATE_REF, weights, 10_000,
                                                CANDIDATE_RECIPE, CANDIDATE_SHA256)
    assert pins["candidate_live"].ref == pins["candidate_ema"].ref


def test_the_candidate_resolves_to_the_178_checkpoint_live_and_ema():
    for weights, use_ema in (("live", False), ("ema", True)):
        model = ag.model_spec(f"candidate_{weights}", spec=asp.CONSONANT1)
        assert (model.ref, model.use_ema, model.step, model.recipe) == (
            CANDIDATE_REF, use_ema, 10_000, CANDIDATE_RECIPE)
    # #95's arm is untouched: the pin is this round's, not the pipeline's.
    assert ag.model_spec("candidate_ema", spec=asp.AUG95).ref == ag.ARMS["candidate"]
    assert ag.model_spec("candidate_ema").ref == ag.ARMS["candidate"]


def _candidate_record(model: str) -> dict:
    pin = asp.CONSONANT1.models[model]
    return {"model": model, "fingerprints": ag.expected_fingerprints(pin.ref),
            "identity": ag.identity_record(pin)}


def test_the_candidate_s_identity_is_the_round_s_pin_and_nothing_else():
    for model in ("candidate_live", "candidate_ema"):
        record = _candidate_record(model)
        assert ag.validate_identity("tarteel_v1.look", model, record,
                                    spec=asp.CONSONANT1) == record["identity"]
    for edit in ({"checkpoint_sha256": "b" * 64},
                 {"model_ref": "/root/scratch/issue-94/repo-fix/tools/runs/"
                               "aug94_candidate_v2/checkpoint.pt"},
                 {"step": 8_000}, {"recipe": "aug94-mixture-v1"}, {"weights": "live"}):
        record = _candidate_record("candidate_ema")
        record["identity"].update(edit)
        with pytest.raises(ValueError, match="pinned checkpoint"):
            ag.validate_identity("tarteel_v1.look", "candidate_ema", record,
                                 spec=asp.CONSONANT1)
    record = _candidate_record("candidate_ema")
    record["fingerprints"]["checks"]["batch_size"] = 8
    with pytest.raises(ValueError, match="fingerprint"):
        ag.validate_identity("tarteel_v1.look", "candidate_ema", record, spec=asp.CONSONANT1)
