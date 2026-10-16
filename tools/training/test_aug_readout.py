"""Tests for the augmentation readout (#95): metrics, bars, the verdict and the discipline.

Torch-free, GPU-free and harness-free: items, decodes and word grades are synthetic, and a fake
:class:`training.aug_readout.Sources` stands in for the committed files and the harness.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from tadabur.qud_text import ItemReference
from tadabur.tarteel_split import CONFIRM, LOOK, half
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_decode as ard
from training import aug_readout_metrics as am
from training import aug_readout_verdict as av
from training.acceptance_stats import CANNOT_CERTIFY, FAIL, NO_BOUND, PASS
from training.external_corpus import WordGrade
from training.external_decode import DEFAULT_SPECS, ModelSpec
from training.muraja_policy import SHIPPED_DEFAULT, SiteGrade
from training.nonpro_eval import ClipItem
from training.nonpro_gap import DevClip
from training.word_flags import Flag

SHIPPED = SHIPPED_DEFAULT.name
# Two words, ``قَالَ سَططَ``; a decode carries no space.
REFERENCE = "قَالَ سَططَ"
CLEAN = REFERENCE.replace(" ", "")
SWAPPED = "قَالَشَططَ"  # the second word's س heard as ش
OFFSETS = (0, 6)
FLAG = Flag(("other_consonant",), (), "minor", (), consonants=(("other_consonant", "س→ش"),))


def _item(n: int, clusters: int = 12, labels=None, clip: str | None = None) -> ClipItem:
    name = clip or f"c{n:03d}.wav"
    return ClipItem(f"{name}#0", "qdat", name, 5, 109, 0, 16000, "e" * 64,
                    ItemReference(("قَالَ", "سَططَ"), (7, 8), REFERENCE, OFFSETS), f"r{n % clusters}",
                    n % clusters, labels or {"final_label": "in_correct"})


def _grade(model: str, item: ClipItem, position: int, flagged: bool, graded: bool = True
           ) -> WordGrade:
    outcome = "flagged" if flagged else "not_flagged" if graded else "not_graded"
    return WordGrade(model, SHIPPED, item, position, 2, position + 1, 7 + position,
                     SiteGrade(outcome), FLAG if flagged else None)


def _line(item: ClipItem, text: str) -> dict:
    return {"key": item.key, "audio_filename": item.audio_filename, "audio_sha256": "e" * 64,
            "start_sample": 0, "end_sample": 16000,
            "checks": [{"hop": text, "overlap": "", "flush": True}], "stream_b0": text}


# --- word segments and agreement ----------------------------------------------------------------


def test_a_decode_is_cut_at_the_reference_s_word_boundaries():
    assert am.word_segments(CLEAN, REFERENCE) == ("قَالَ", "سَططَ")
    assert am.word_segments(SWAPPED, REFERENCE) == ("قَالَ", "شَططَ")
    # An inserted letter belongs to the word consumed before it (the first, at the start).
    assert am.word_segments("x" + CLEAN + "y", REFERENCE) == ("xقَالَ", "سَططَy")
    assert am.word_segments("", REFERENCE) == ("", "")


def test_word_agreement_counts_the_words_whose_letters_are_the_teacher_s():
    items = [_item(0), _item(1)]
    decodes = {"base": {i.key: CLEAN for i in items},
               "h448": {items[0].key: SWAPPED, items[1].key: CLEAN}}
    terms = am.word_agreement_terms(items, decodes, ("h448", "base"))
    assert terms.by_model["h448"][0].tolist() == [1, 2]
    assert terms.by_model["h448"][1].tolist() == [2, 2]
    assert terms.by_model["base"][0].tolist() == [2, 2]  # the teacher agrees with itself


# --- false flags and the paired comparison ------------------------------------------------------


def _flag_grades(pattern: dict[str, callable], n: int = 24) -> tuple[list, list[WordGrade]]:
    items = [_item(i) for i in range(n)]
    grades = [_grade(model, item, p, flagged(i, p))
              for model, flagged in pattern.items() for i, item in enumerate(items)
              for p in (0, 1)]
    return items, grades


def test_false_flags_are_flagged_over_graded_words_compared_in_pairs():
    _, grades = _flag_grades({"candidate": lambda i, p: p == 1 and i % 6 == 0,
                              "h448": lambda i, p: p == 1 and i % 2 == 0,
                              "control": lambda i, p: p == 1 and i % 2 == 0})
    view = am.compare(am.false_flag_terms(grades, ("candidate", "control", "h448"), SHIPPED),
                      am.PER_1000)
    assert view["models"]["h448"]["point"] == pytest.approx(1000 * 12 / 48)
    assert view["models"]["candidate"]["point"] == pytest.approx(1000 * 4 / 48)
    difference = view["comparisons"]["candidate−h448"]
    assert difference["point"] == pytest.approx(1000 * -8 / 48)
    assert difference["interval"]["method"] == "bootstrap"
    assert difference["interval"]["upper"] < 0
    assert set(view["comparisons"]) == {"candidate−h448", "candidate−control"}
    assert (view["observations"], view["clusters"]) == (48, 12)


def test_an_ungraded_word_is_out_of_the_rate_and_counted_as_ungraded():
    item = _item(0)
    grades = [_grade("h448", item, 0, False, graded=False), _grade("h448", item, 1, True)]
    terms = am.false_flag_terms(grades, ("h448",), SHIPPED)
    assert terms.by_model["h448"][0].tolist() == [0, 1]
    assert terms.by_model["h448"][1].tolist() == [0, 1]
    assert am.ungraded_terms(grades, ("h448",), SHIPPED).by_model["h448"][0].tolist() == [1, 0]


def test_models_must_place_the_same_words():
    item = _item(0)
    grades = [_grade("h448", item, 0, False), _grade("candidate", item, 1, False)]
    with pytest.raises(ValueError, match="different words"):
        am.false_flag_terms(grades, ("h448", "candidate"), SHIPPED)


def test_a_clip_is_flagged_when_any_of_its_words_is():
    items = [_item(0), _item(1)]
    grades = [_grade("h448", items[0], 0, True), _grade("h448", items[0], 1, False),
              _grade("h448", items[1], 0, False), _grade("h448", items[1], 1, False)]
    terms = am.clips_flagged_terms(grades, ("h448",), SHIPPED)
    assert terms.by_model["h448"][0].tolist() == [1, 0]


# --- the other observations --------------------------------------------------------------------


def test_tadabur_agreement_is_section_2_s_against_the_cached_teacher():
    clips = [DevClip("a.wav", 3, (0, 0), 1.0, "abcd"), DevClip("b.wav", 4, (0, 1), 1.0, "ab")]
    terms = am.teacher_agreement_terms(clips, {"h448": {"a.wav": "abxd", "b.wav": "ab"}},
                                       ("h448",))
    assert terms.clusters == (3, 4)
    assert terms.by_model["h448"][0].tolist() == [3, 2]
    assert terms.by_model["h448"][1].tolist() == [4, 2]


def test_the_bench_counts_agreeing_slots_of_the_experts_transcript_per_kind():
    item = _item(0)
    terms = am.bench_terms([item], {"h448": {item.key: SWAPPED}}, {item.audio_filename: REFERENCE},
                           ("h448",))
    assert terms["consonant"].by_model["h448"][0].tolist() == [3]  # ق ل agree, س does not, ط does
    assert terms["consonant"].by_model["h448"][1].tolist() == [4]
    assert terms["shaddah"].by_model["h448"][0].tolist() == [1]


def test_leniency_counts_blank_frames_and_carriers_left_empty():
    clips = [DevClip("a.wav", 3, (0, 0), 1.0, "")]
    terms = am.blank_terms(clips, {"h448": {"a.wav": [90, 125]}}, ("h448",))
    assert terms.by_model["h448"][0].tolist() == [90]
    item = _item(0)
    empty = am.abstention_terms([item], {"h448": {item.key: "قَالَ"}}, ("h448",))
    assert empty.by_model["h448"][1].tolist() == [4]  # ق ل س ط
    assert empty.by_model["h448"][0].tolist()[0] >= 1  # the second word is not committed


def test_terms_refuse_misaligned_arrays():
    with pytest.raises(ValueError):
        am.Terms((1, 2), {"h448": (np.zeros(1), np.zeros(2))})


# --- the bars and the verdict ------------------------------------------------------------------


def _interval(lower, upper, method="bootstrap", reason=""):
    return {"lower": lower, "upper": upper, "method": method, "reason": reason}


def _view(points: dict, diffs: dict, scale=am.PER_1000) -> dict:
    return {"scale": scale, "observations": 100, "clusters": 20,
            "models": {m: {"point": p, "num": 0, "den": 1, "interval": _interval(p, p)}
                       for m, p in points.items()},
            "comparisons": {name: {"point": d[0], "interval": _interval(d[1], d[2])}
                            for name, d in diffs.items()}}


def _passing_report() -> dict:
    vs_h, vs_c = "candidate−h448", "candidate−control"
    agreement = _view({"candidate": 95.0, "control": 94.0, "h448": 93.0},
                      {vs_h: (2.0, 1.0, 3.0), vs_c: (1.0, 0.5, 1.5)}, am.POINTS)
    bench = _view({"candidate": 90.0, "h448": 90.0}, {vs_h: (0.0, -1.5, 1.5)}, am.POINTS)
    return {
        "main": {"tarteel_v1_confirm": _view(
            {"candidate": 40.0, "control": 52.0, "h448": 57.0, "base": 21.0},
            {vs_h: (-17.0, -20.0, -14.0), vs_c: (-12.0, -15.0, -9.0)})},
        "mechanism": {"tarteel_v1_confirm": agreement, "qdat": agreement},
        "consistency": {"qdat": _view({"candidate": 60.0, "h448": 65.0},
                                      {vs_h: (-5.0, -9.0, -1.0)})},
        "guards": {"qud_half": _view({"candidate": 4.0, "h448": 4.3}, {vs_h: (-0.3, -1.0, 0.5)}),
                   "tadabur_dev_unseen": _view({"candidate": 94.5, "h448": 94.6},
                                               {vs_h: (-0.1, -0.6, 0.4)}, am.POINTS),
                   "qdat_bench": {k: bench for k in ("consonant", "haraka", "shaddah")}},
    }


def _condition(result: dict, name: str) -> dict:
    return next(c for c in result["conditions"] if c["name"] == name)


def test_the_bars_are_128_s_as_the_owner_set_them():
    assert av.BARS == {"main_upper_below": 0.0, "main_gap_share": 0.25,
                       "mechanism_lower_above": 0.0, "qud_upper_at_most": 1.0,
                       "tadabur_lower_at_least": -1.0, "bench_lower_at_least": -2.0}


def test_a_candidate_clearing_every_bar_passes_and_beats_the_control():
    result = av.verdict(_passing_report())
    assert result["verdict"] == PASS
    assert [c["verdict"] for c in result["conditions"]] == [PASS] * 10
    gap = _condition(result, "main.quarter_gap")
    assert (gap["gap"], gap["required_reduction"], gap["reduction"]) == (36.0, 9.0, 17.0)
    assert result["attribution"]["reading"] == av.BEATS_CONTROL


@pytest.mark.parametrize("edit, name", [
    (lambda r: r["main"]["tarteel_v1_confirm"]["comparisons"]["candidate−h448"]["interval"]
     .update(upper=0.0), "main.app_outcome"),  # < 0 is strict
    (lambda r: r["main"]["tarteel_v1_confirm"]["models"]["candidate"].update(point=50.0),
     "main.quarter_gap"),  # a reduction of 7 of a 36 gap
    (lambda r: r["mechanism"]["qdat"]["comparisons"]["candidate−h448"]["interval"]
     .update(lower=0.0), "mechanism.qdat"),  # > 0 is strict
    (lambda r: r["consistency"]["qdat"]["models"]["candidate"].update(point=65.0),
     "consistency.qdat"),  # not lower
    (lambda r: r["guards"]["qud_half"]["comparisons"]["candidate−h448"]["interval"]
     .update(upper=1.01), "guard.qud"),
    (lambda r: r["guards"]["tadabur_dev_unseen"]["comparisons"]["candidate−h448"]["interval"]
     .update(lower=-1.01), "guard.tadabur"),
    (lambda r: r["guards"]["qdat_bench"].update(shaddah=_view(
        {"candidate": 1.0, "h448": 1.0}, {"candidate−h448": (0.0, -2.5, 1.0)}, am.POINTS)),
     "guard.qdat_bench.shaddah"),
    (lambda r: r["main"]["tarteel_v1_confirm"]["comparisons"]["candidate−h448"]["interval"]
     .update(upper="inf"), "main.app_outcome"),  # an adverse infinite end is compared
])
def test_any_failed_bar_fails_the_verdict(edit, name):
    report = _passing_report()
    edit(report)
    result = av.verdict(report)
    assert _condition(result, name)["verdict"] == FAIL
    assert result["verdict"] == FAIL


def test_the_inclusive_guards_hold_at_their_bar():
    report = _passing_report()
    report["guards"]["qud_half"]["comparisons"]["candidate−h448"]["interval"]["upper"] = 1.0
    report["guards"]["tadabur_dev_unseen"]["comparisons"]["candidate−h448"]["interval"][
        "lower"] = -1.0
    assert av.verdict(report)["verdict"] == PASS


def test_no_bound_cannot_certify_and_never_passes():
    report = _passing_report()
    report["guards"]["qud_half"]["comparisons"]["candidate−h448"]["interval"] = _interval(
        None, None, NO_BOUND, "sparse or degenerate")
    result = av.verdict(report)
    assert _condition(result, "guard.qud")["verdict"] == CANNOT_CERTIFY
    assert result["verdict"] == CANNOT_CERTIFY
    report["main"]["tarteel_v1_confirm"]["models"]["candidate"]["point"] = 56.0
    assert av.verdict(report)["verdict"] == FAIL  # a fail outranks cannot_certify


def test_a_gap_that_is_not_positive_cannot_certify():
    report = _passing_report()
    report["main"]["tarteel_v1_confirm"]["models"]["base"]["point"] = 60.0
    assert _condition(av.verdict(report), "main.quarter_gap")["verdict"] == CANNOT_CERTIFY


def test_beating_h448_but_not_the_control_is_a_null_for_augmentation():
    report = _passing_report()
    report["main"]["tarteel_v1_confirm"]["comparisons"]["candidate−control"]["interval"][
        "upper"] = 2.0
    result = av.verdict(report)
    assert result["verdict"] == PASS  # the reading never changes the verdict
    assert result["attribution"]["reading"] == av.NULL_FOR_AUGMENTATION
    report["main"]["tarteel_v1_confirm"]["comparisons"]["candidate−h448"]["interval"][
        "upper"] = 1.0
    assert av.verdict(report)["attribution"]["reading"] == av.NOT_APPLICABLE


# --- models, sets and the order ------------------------------------------------------------------


def test_readout_models_name_an_arm_s_weights_at_the_final_step():
    assert ag.model_spec("candidate_ema") == ModelSpec(ag.ARMS["candidate"], True, 10_000)
    assert ag.model_spec("control_live") == ModelSpec(ag.ARMS["control"], False, 10_000)
    assert ag.model_spec("h448") is DEFAULT_SPECS["h448"]
    with pytest.raises(ValueError):
        ag.model_spec("candidate_best")
    # The truth baseline's identity records stay exactly what they were.
    assert DEFAULT_SPECS["h448"].identity_extra() == {}
    assert ag.model_spec("candidate_ema").identity_extra() == {"weights": "ema", "step": 10_000}


def test_before_the_choice_only_the_candidate_s_look_half_may_be_read():
    assert ag.allowed_models(ag.TARTEEL_LOOK, None) == ("candidate_live", "candidate_ema")
    for name in set(ag.SETS) - {ag.TARTEEL_LOOK}:
        with pytest.raises(SystemExit, match="only after the look half"):
            ag.allowed_models(name, None)


def test_after_the_choice_only_the_chosen_weights_are_read():
    choice = {"weights": "ema"}
    assert ag.allowed_models(ag.TARTEEL_CONFIRM, choice) == ("candidate_ema", "control_ema")
    assert ag.allowed_models(ag.TADABUR_SET, choice) == ("candidate_ema", "control_ema", "h448",
                                                         "base")


def test_the_choice_is_fewer_look_half_flags_then_agreement_then_live():
    def inputs(live, ema):
        return {f"candidate_{w}": {"false_flags_per_1000": f, "word_agreement_points": a}
                for w, (f, a) in (("live", live), ("ema", ema))}

    assert ar.choose(inputs((50.0, 90.0), (49.0, 80.0))) == "ema"
    assert ar.choose(inputs((49.0, 90.0), (50.0, 99.0))) == "live"
    assert ar.choose(inputs((50.0, 90.0), (50.0, 91.0))) == "ema"
    assert ar.choose(inputs((50.0, 90.0), (50.0, 90.0))) == "live"


# --- QUD's half ------------------------------------------------------------------------------------


def test_qud_s_half_is_a_salted_hash_of_the_segment_and_re_checked_on_load(tmp_path):
    keys = [f"qud_m_{i:03d}_001.mp3#{k}" for i in range(400) for k in range(2)]
    record = ar.qud_selection_record({"m": keys})
    assert record["kept"]["m"] == sorted(k for k in keys if ar.qud_kept(k))
    assert 0.4 < record["kept_counts"]["m"] / len(keys) < 0.6
    path = tmp_path / "s.json"
    path.write_text(json.dumps(record))
    assert ar.load_qud_selection(path) == set(record["kept"]["m"])
    record["kept"]["m"].append(next(k for k in keys if not ar.qud_kept(k)))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not keep"):
        ar.load_qud_selection(path)


def test_the_committed_qud_half_is_the_rule_over_both_mushafs():
    record = json.loads(ar.QUD_SELECTION_PATH.read_text(encoding="utf-8"))
    assert record["salt"] == ar.QUD_SALT and set(record["kept"]) == {
        "abdul_hamid_ghraio_2026_yt", "anas_almiman_yt"}
    assert sum(record["kept_counts"].values()) == len(ar.load_qud_selection())


# --- the decoder's pieces -------------------------------------------------------------------------


def test_the_blank_count_rides_on_exactly_the_deployed_stream():
    from training.decoding import BLANK_ID, Decoder, window_audio

    class Fake:
        batch_size = 1
        emissions = Decoder.emissions
        decode_stream = Decoder.decode_stream

        def window_rows(self, windows):
            rng = np.random.default_rng(len(windows))
            return [np.where(rng.random(125) < 0.7, BLANK_ID, rng.integers(1, 40, 125))
                    for _ in windows]

    span = np.zeros(16000 * 12, dtype=np.float32)
    out = ard.stream_with_blanks(Fake(), span)
    assert out[ag.STREAM] == Fake().decode_stream(span, block=0, flush_tail=True)
    rows = Fake().window_rows(window_audio(span))
    assert out[ag.BLANKS] == [sum(int((r == BLANK_ID).sum()) for r in rows), 125 * len(rows)]


def test_a_set_keeps_only_its_items_and_never_loads_a_recording_without_any():
    from training.external_decode import Recording

    loaded = []

    def load(name):
        def inner():
            loaded.append(name)
            return np.zeros(4), [{"key": f"{name}#0"}, {"key": f"{name}#1"}]
        return inner

    recordings = [Recording(("a#0", "a#1"), load("a")), Recording(("b#0", "b#1"), load("b"))]
    (kept,) = ard.kept(iter(recordings), lambda key: key == "a#1")
    assert kept.keys == ("a#1",)
    assert [r["key"] for r in kept.load()[1]] == ["a#1"]
    assert loaded == ["a"]


def test_tarteel_s_halves_split_by_clip_hash():
    clips = [f"1_{i}_x.wav" for i in range(200)]
    look = [c for c in clips if half(c) == LOOK]
    assert 0 < len(look) < 200 and all(half(c) == CONFIRM for c in set(clips) - set(look))


def test_a_checkpoint_at_another_step_is_refused_before_it_is_loaded(monkeypatch, tmp_path):
    from training import external_decode as ed

    monkeypatch.setattr(ed, "checkpoint_step", lambda path: 8_000)
    with pytest.raises(ValueError, match="holds step 8000, not 10000"):
        ed.load_decoders(lambda model: tmp_path / f"{model}.json",
                         {"candidate_ema": ag.model_spec("candidate_ema")})


# --- identity -------------------------------------------------------------------------------------

SHA = {"candidate": "c" * 64, "control": "d" * 64}


def _record(model: str) -> dict:
    spec = ag.model_spec(model)
    identity = ag.baseline_identities()[model] if model in ag.BASELINES else {
        "model_ref": spec.ref, "checkpoint_sha256": SHA[model.rsplit("_", 1)[0]],
        **spec.identity_extra()}
    return {"model": model, "identity": identity, "fingerprints": ag.expected_fingerprints(spec.ref)}


def _manifest(phase: str, weights: str | None) -> dict:
    return {f"{s}/{m}": _record(m) for s, models in ag.needs(phase, weights).items()
            for m in models}


def test_a_complete_identity_names_the_checkpoint_s_sha_its_weights_and_the_final_step():
    assert ag.validate_identity(ag.QUD_SET, "candidate_ema", _record("candidate_ema"))["step"] \
        == 10_000
    for edit in ({"step": 8_000}, {"weights": "live"}, {"checkpoint_sha256": "x"},
                 {"model_ref": "/elsewhere.pt"}):
        record = _record("candidate_ema")
        record["identity"].update(edit)
        with pytest.raises(ValueError):
            ag.validate_identity(ag.QUD_SET, "candidate_ema", record)
    record = _record("candidate_ema")
    del record["identity"]["checkpoint_sha256"]  # Sol's finding 2: no SHA, no identity
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        ag.validate_identity(ag.QUD_SET, "candidate_ema", record)
    record = _record("h448")
    record["fingerprints"]["checks"]["batch_size"] = 32
    with pytest.raises(ValueError, match="fingerprints"):
        ag.validate_identity(ag.TADABUR_SET, "h448", record)


def test_a_sitting_needs_every_identity_it_reads():
    with pytest.raises(ValueError, match="no identity record"):  # Sol's finding 2: empty models
        ag.require_manifest({}, ag.needs(CONFIRM, "ema"), {"weights": "ema", "identities": {}})
    records = _manifest(CONFIRM, "ema")
    del records[f"{ag.QUD_SET}/control_ema"]
    with pytest.raises(ValueError, match="no identity record for qud_half/control_ema"):
        ag.require_manifest(records, ag.needs(CONFIRM, "ema"), {"weights": "ema",
                                                                 "identities": {}})
    records = _manifest(CONFIRM, "ema")
    records[f"{ag.QDAT_SET}/control_ema"]["identity"]["checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="one checkpoint"):
        ag.require_manifest(records, ag.needs(CONFIRM, "ema"), {"weights": "ema",
                                                                 "identities": {}})
    choice = {"weights": "ema", "identities": {"candidate_ema": {
        **_record("candidate_ema")["identity"], "checkpoint_sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="the committed choice read"):
        ag.require_manifest(_manifest(CONFIRM, "ema"), ag.needs(CONFIRM, "ema"), choice)


def test_every_reused_baseline_decode_is_one_model_under_the_frozen_protocol():
    found = ag.baseline_identities()
    assert set(found) == {"base", "h448"}
    assert found["h448"]["model_ref"].endswith("h448_stream/checkpoint.pt")


def test_a_readout_decode_must_carry_and_match_its_set_s_whole_provenance(tmp_path, monkeypatch):
    import gzip

    item = _item(0)
    line = _line(item, CLEAN)
    path = tmp_path / "x.jsonl.gz"
    monkeypatch.setattr(ar, "decodes_file", lambda set_name, model: path)

    def write(record):
        path.write_bytes(gzip.compress((json.dumps(record) + "\n").encode()))

    write({**line, "audio_sha256": "0" * 64})
    with pytest.raises(ValueError, match="not the audio"):
        ar.read_readout(ag.QDAT_SET, "candidate_ema", {item.key: line}, {item.key})
    # Sol's finding 3: a field missing on either side is refused, never skipped.
    write({k: v for k, v in line.items() if k != "audio_sha256"})
    with pytest.raises(ValueError, match="no \\['audio_sha256'\\]"):
        ar.read_readout(ag.QDAT_SET, "candidate_ema", {item.key: line}, {item.key})
    write(line)
    with pytest.raises(ValueError, match="no \\['end_sample'\\]"):
        ar.read_readout(ag.QDAT_SET, "candidate_ema",
                        {item.key: {k: v for k, v in line.items() if k != "end_sample"}},
                        {item.key})
    with pytest.raises(ValueError, match="mp3_sha256"):
        ar.read_readout(ag.QUD_SET, "candidate_ema", {item.key: line}, {item.key})
    assert ar.read_readout(ag.QDAT_SET, "candidate_ema", {item.key: line}, {item.key})
    with pytest.raises(ValueError, match="not decoded"):
        ar.read_readout(ag.QDAT_SET, "candidate_ema", {item.key: line}, {item.key, "y#0"})


# --- the committed choice and the claims ----------------------------------------------------------


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def _repo(tmp_path) -> ag.Layout:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("x")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "-qm", "init")
    return ag.Layout(tmp_path)


def _commit(layout: ag.Layout, *paths) -> None:
    _git(layout.root, "add", *(p.name for p in paths))
    _git(layout.root, "commit", "-qm", "c")


def test_a_choice_counts_as_committed_only_when_tracked_and_unchanged(tmp_path):
    layout = _repo(tmp_path)
    path = layout.choice
    path.write_text("{}")
    assert not ag.is_committed(path)
    _commit(layout, path)
    assert ag.is_committed(path)
    path.write_text('{"weights": "live"}')
    assert not ag.is_committed(path)


def test_a_claim_is_held_once_and_recovered_only_by_the_same_sitting(tmp_path):
    path = tmp_path / "x.claim.json"
    with ag.claim(path, {"a": 1}):
        with pytest.raises(SystemExit, match="being read now"):  # a concurrent invocation
            with ag.claim(path, {"a": 1}):
                pass
    with ag.claim(path, {"a": 1}):  # after an interruption, the same sitting may run again
        pass
    with pytest.raises(SystemExit, match="another sitting"):
        with ag.claim(path, {"a": 2}):
            pass


N = 48


#: h448's and the control's mistake: a kasra for the ل's fatha, the س as ش, the geminate let go.
H448_WRONG = "قَالِشَطَ"


def _decode_text(model: str, i: int) -> str:
    """h448 and the control go wrong in two items of three; the candidate (only the س as ش) in
    one of four; the teacher never."""
    if model in ("h448", "control"):
        return H448_WRONG if i % 3 != 0 else CLEAN
    return SWAPPED if model == "candidate" and i % 4 == 0 else CLEAN


def _flags(model: str, i: int, position: int) -> bool:
    return position == 1 and {"base": i % 8 == 0, "h448": i % 2 == 0, "control": i % 2 == 0,
                              "candidate": i % 4 == 0}[model]


class FakeSources(ar.Sources):
    """Every set synthetic, its input files in ``files``; names are bound to roles."""

    def __init__(self, files):
        self.files = files
        files.mkdir(exist_ok=True)
        for name in ("look", "confirm"):
            (files / f"{name}.input").write_text(name)
        self.read: list[str] = []
        # Eleven clusters, so no pattern below lines up with them.
        self.items = [_item(i, clusters=11, clip=f"{i:03d}_{i}_c.wav") for i in range(4 * N)]

    def input_files(self, phase, weights):
        return [self.files / f"{phase}.input"]

    def _role(self, model: str) -> str:
        return model if model in ag.BASELINES else model.rsplit("_", 1)[0]

    def _set(self, name: str, items, models) -> ar.SetData:
        self.read.append(name)
        return ar.SetData(items, {m: {i.key: _line(i, _decode_text(self._role(m), n))
                                      for n, i in enumerate(items)} for m in models})

    def tarteel(self, which_half, models):
        return self._set(f"tarteel_v1.{which_half}",
                         [i for i in self.items if half(i.audio_filename) == which_half], models)

    def qdat(self, models):
        return self._set("qdat", self.items[:N], models)

    def qud(self, models):
        return self._set("qud_half", self.items[N:2 * N], models)

    def retasy(self, models):
        return self._set("retasy", self.items[2 * N:3 * N], models)

    def tadabur(self, models):
        self.read.append("tadabur")
        clips = [DevClip(f"d{i}.wav", i % 12, (0, i), 1.0, CLEAN) for i in range(N)]
        roles = (*models, *ag.BASELINES)
        decodes = {m: {c.filename: CLEAN if self._role(m) == "candidate" or i % 5 else SWAPPED
                       for i, c in enumerate(clips)} for m in roles}
        blanks = {m: {c.filename: [80, 125] for c in clips} for m in roles}
        return ar.TadaburData(clips, decodes, blanks, {"h448": N, "base": N})

    def bench(self):
        return {i.audio_filename: REFERENCE for i in self.items[:N]}

    def grade(self, items, decodes, models):
        index = {i.key: n for n, i in enumerate(items)}
        return [_grade(m, i, p, _flags(self._role(m), index[i.key], p))
                for m in models for i in items for p in (0, 1)]

    harness_record = {"build": {"muraja_commit": "test"}, "binary_sha256": "a" * 64}

    def harness_identity(self):
        return dict(self.harness_record)


def _looked(tmp_path) -> tuple[ag.Layout, FakeSources, dict]:
    """A readout whose look sitting ran, with its manifest; nothing committed yet."""
    layout = _repo(tmp_path / "readout")
    sources = FakeSources(tmp_path / "files")
    layout.models.write_text(json.dumps(_manifest(LOOK, None)))
    choice = ar.run_look(sources, layout)
    return layout, sources, choice


def _committed(tmp_path) -> tuple[ag.Layout, FakeSources, dict]:
    layout, sources, choice = _looked(tmp_path)
    _commit(layout, layout.look, layout.choice, layout.claim(LOOK))
    layout.models.write_text(json.dumps({**_manifest(LOOK, None),
                                         **_manifest(CONFIRM, choice["weights"])}))
    return layout, sources, choice


def test_look_needs_the_candidate_s_identities_before_it_reads(tmp_path):
    layout = _repo(tmp_path / "readout")
    sources = FakeSources(tmp_path / "files")
    with pytest.raises(ValueError, match="no identity record"):
        ar.run_look(sources, layout)
    assert sources.read == [] and not layout.claim(LOOK).exists()


def test_look_reads_the_look_half_once_and_binds_the_choice(tmp_path):
    layout, sources, choice = _looked(tmp_path)
    assert sources.read == ["tarteel_v1.look"]
    assert choice["weights"] == "live"  # live and EMA tie here; a tie goes to live
    look = json.loads(layout.look.read_text())
    assert set(look["inputs"]) == {"candidate_live", "candidate_ema"}
    assert look["control"] == ar.CONTROL_ON_LOOK
    assert choice["models"] == {"candidate": "candidate_live", "control": "control_live"}
    assert choice["identities"]["candidate_live"]["checkpoint_sha256"] == SHA["candidate"]
    assert choice["look_sha256"] == ag.sha256_file(layout.look)
    with pytest.raises(SystemExit, match="read once"):
        ar.run_look(sources, layout)


def test_an_untracked_or_edited_choice_unlocks_nothing(tmp_path, monkeypatch):
    layout, sources, choice = _looked(tmp_path)
    layout.models.write_text(json.dumps({**_manifest(LOOK, None),
                                         **_manifest(CONFIRM, "live")}))
    # Sol's finding 1: an untracked choice opens neither the decoder, nor merge, nor confirm.
    with pytest.raises(SystemExit, match="not committed"):
        ard.check_models(ag.QUD_SET, ["candidate_live", "control_live"], layout)
    with pytest.raises(SystemExit, match="not committed"):
        ar.run_confirm(sources, layout)
    decodes = tmp_path / "box"
    decodes.mkdir()
    (decodes / f"{ag.QUD_SET}.candidate_live.identity.json").write_text(
        json.dumps(_record("candidate_live")))
    published = []
    monkeypatch.setattr("training.external_corpus.merge_decode_files",
                        lambda *a: published.append(a))
    with pytest.raises(SystemExit, match="not committed"):
        ar.merge(decodes, layout)
    assert published == []
    _commit(layout, layout.look, layout.choice, layout.claim(LOOK))
    ard.check_models(ag.QUD_SET, ["candidate_live", "control_live"], layout,
                     checkpoint_sha256=lambda ref: SHA["candidate"])
    with pytest.raises(SystemExit, match="not the one the choice read"):
        ard.check_models(ag.QUD_SET, ["candidate_live"], layout,
                         checkpoint_sha256=lambda ref: "0" * 64)
    # Editing the committed choice to the other weights unlocks nothing either.
    layout.choice.write_text(json.dumps({**choice, "weights": "ema"}))
    with pytest.raises(SystemExit, match="not committed"):
        ard.check_models(ag.QUD_SET, ["candidate_ema", "control_ema"], layout)


def test_a_committed_choice_must_be_bound_to_the_committed_look(tmp_path):
    layout, _, choice = _looked(tmp_path)
    layout.choice.write_text(json.dumps({**choice, "look_sha256": "0" * 64}))
    _commit(layout, layout.look, layout.choice, layout.claim(LOOK))
    with pytest.raises(SystemExit, match="not bound"):
        ag.committed_choice(layout)


def test_the_decoder_reads_the_look_half_only_for_the_candidate(tmp_path):
    layout = _repo(tmp_path)
    ard.check_models(ag.TARTEEL_LOOK, ["candidate_live", "candidate_ema"], layout)
    with pytest.raises(SystemExit):
        ard.check_models(ag.TARTEEL_LOOK, ["control_live"], layout)
    with pytest.raises(SystemExit, match="not committed"):
        ard.check_models(ag.QUD_SET, ["candidate_live"], layout)


def test_confirm_needs_every_identity_before_it_reads(tmp_path):
    layout, sources, _ = _committed(tmp_path)
    layout.models.write_text(json.dumps(_manifest(LOOK, None)))  # no confirm identities
    with pytest.raises(ValueError, match="no identity record"):
        ar.run_confirm(sources, layout)
    assert "qdat" not in sources.read and not layout.claim(CONFIRM).exists()


def test_confirm_reads_every_set_once_and_writes_an_audited_verdict(tmp_path):
    layout, sources, choice = _committed(tmp_path)
    report = ar.run_confirm(sources, layout)
    assert sorted(sources.read) == sorted(["tarteel_v1.look", "tarteel_v1.confirm", "qdat",
                                           "qud_half", "retasy", "tadabur"])
    main = report["main"]["tarteel_v1_confirm"]
    assert set(main["models"]) == {"candidate", "control", "h448", "base"}
    assert main["models"]["h448"]["point"] == pytest.approx(250.0)
    assert main["models"]["candidate"]["point"] == pytest.approx(125.0)
    retasy = report["catch"]["retasy_in_correct"]["words"]["models"]
    assert set(retasy) == {"base", "h448", "candidate", "control"}  # side by side
    verdict = json.loads(layout.verdict.read_text())
    assert verdict["verdict"] == PASS, [c for c in verdict["conditions"] if c["verdict"] != PASS]
    assert verdict["attribution"]["reading"] == av.BEATS_CONTROL
    assert verdict["attribution"]["control_vs_h448"]["point"] == pytest.approx(0.0)
    audit = report["audit"]
    assert audit["choice_commit"] and audit["choice_sha256"] == ag.sha256_file(layout.choice)
    assert audit["identities"]["control_live"]["checkpoint_sha256"] == SHA["control"]
    assert audit["inputs"] == {str((sources.files / "confirm.input").resolve()):
                               ag.sha256_file(sources.files / "confirm.input")}
    assert report["leniency"]["blank_ratio"]["models"]["candidate"]["point"] == pytest.approx(64.0)
    with pytest.raises(SystemExit, match="read once"):
        ar.run_confirm(sources, layout)
    from training.aug_readout_doc import render

    text = render(json.loads(layout.look.read_text()), json.loads(layout.report.read_text()))
    assert "## Verdict: pass" in text and "candidate−control" in text
    assert "never by a re-run" in text and "control was not decoded" in text


def test_an_interrupted_confirm_resumes_only_as_the_same_sitting(tmp_path, monkeypatch):
    layout, sources, _ = _committed(tmp_path)
    original = ar.write_text_atomically

    def crash_on_verdict(path, text):
        if path == layout.verdict:
            raise KeyboardInterrupt
        original(path, text)

    monkeypatch.setattr(ar, "write_text_atomically", crash_on_verdict)
    with pytest.raises(KeyboardInterrupt):
        ar.run_confirm(sources, layout)
    assert layout.report.exists() and not layout.verdict.exists()
    monkeypatch.setattr(ar, "write_text_atomically", original)
    # Sol's finding 4: the inputs moved, so this is another sitting, and it is refused.
    (sources.files / "confirm.input").write_text("another decode")
    with pytest.raises(SystemExit, match="another sitting"):
        ar.run_confirm(sources, layout)
    (sources.files / "confirm.input").write_text("confirm")
    assert ar.run_confirm(sources, layout)["verdict"]["verdict"] == PASS



def _amend(layout: ag.Layout, edit) -> dict:
    """Rewrite the committed choice with ``edit`` and commit it again (Sol's round 2)."""
    choice = json.loads(layout.choice.read_text())
    edit(choice)
    layout.choice.write_text(ag.canonical(choice))
    _commit(layout, layout.choice)
    return choice


def test_an_amended_choice_cannot_pick_weights_the_look_did_not(tmp_path):
    layout, sources, choice = _committed(tmp_path)
    assert choice["weights"] == "live"

    def to_ema(c):
        c["weights"] = "ema"
        c["models"] = {"candidate": "candidate_ema", "control": "control_ema"}

    _amend(layout, to_ema)
    with pytest.raises(SystemExit, match="other weights than look.json chose"):
        ag.committed_choice(layout)
    with pytest.raises(SystemExit, match="other weights"):
        ard.check_models(ag.QUD_SET, ["candidate_ema", "control_ema"], layout,
                         checkpoint_sha256=lambda ref: SHA["candidate"])
    layout.models.write_text(json.dumps({**_manifest(LOOK, None), **_manifest(CONFIRM, "ema")}))
    with pytest.raises(SystemExit, match="other weights"):
        ar.run_confirm(sources, layout)
    assert not layout.verdict.exists()


def test_an_amended_choice_cannot_substitute_a_checkpoint_the_look_did_not_read(tmp_path):
    layout, sources, _ = _committed(tmp_path)
    other = "e" * 64

    def substitute(c):
        for identity in c["identities"].values():
            identity["checkpoint_sha256"] = other

    _amend(layout, substitute)
    with pytest.raises(SystemExit, match="other checkpoints than the look read"):
        ag.committed_choice(layout)
    with pytest.raises(SystemExit, match="other checkpoints"):
        ard.check_models(ag.QUD_SET, ["candidate_live", "control_live"], layout,
                         checkpoint_sha256=lambda ref: other)
    # A confirm manifest naming the substitute agrees with the amended choice, and is refused.
    records = {**_manifest(LOOK, None), **_manifest(CONFIRM, "live")}
    for name, record in records.items():
        if name.endswith("/candidate_live") and not name.startswith(ag.TARTEEL_LOOK):
            record["identity"]["checkpoint_sha256"] = other
    layout.models.write_text(json.dumps(records))
    with pytest.raises(SystemExit, match="other checkpoints"):
        ar.run_confirm(sources, layout)
    assert not layout.verdict.exists()


def test_a_confirm_candidate_must_be_the_checkpoint_the_choice_read(tmp_path):
    layout, sources, _ = _committed(tmp_path)
    records = json.loads(layout.models.read_text())
    records[f"{ag.QUD_SET}/candidate_live"]["identity"]["checkpoint_sha256"] = "e" * 64
    layout.models.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        ar.run_confirm(sources, layout)


def test_a_sitting_resumes_only_under_the_harness_it_claimed(tmp_path, monkeypatch):
    layout, sources, _ = _committed(tmp_path)
    original = ar.write_text_atomically

    def crash_on_verdict(path, text):
        if path == layout.verdict:
            raise KeyboardInterrupt
        original(path, text)

    monkeypatch.setattr(ar, "write_text_atomically", crash_on_verdict)
    with pytest.raises(KeyboardInterrupt):
        ar.run_confirm(sources, layout)
    monkeypatch.setattr(ar, "write_text_atomically", original)
    claimed = json.loads(layout.claim(CONFIRM).read_text())["harness"]
    assert claimed == FakeSources.harness_record
    sources.harness_record = {**FakeSources.harness_record, "quran_db_sha256": "b" * 64}
    with pytest.raises(SystemExit, match="another sitting"):
        ar.run_confirm(sources, layout)
    assert not layout.verdict.exists()


def test_a_harness_that_changes_during_a_sitting_writes_nothing(tmp_path):
    layout, sources, _ = _committed(tmp_path)
    grade = sources.grade

    def grade_and_swap(*args):
        sources.harness_record = {**FakeSources.harness_record, "binary_sha256": "f" * 64}
        return grade(*args)

    sources.grade = grade_and_swap
    with pytest.raises(SystemExit, match="harness changed"):
        ar.run_confirm(sources, layout)
    assert not layout.report.exists() and not layout.verdict.exists()


def test_the_harness_identity_hashes_the_binary_and_both_databases(tmp_path):
    from training.muraja_policy import Harness

    files = {}
    for name in ("binary", "quran.db", "layout.db"):
        files[name] = tmp_path / name
        files[name].write_text(name)
    harness = Harness(files["binary"], files["quran.db"], files["layout.db"], "c" * 40, "swift",
                      {"protocol_version": 3})
    identity = ag.harness_identity(harness)
    assert identity["quran_db_sha256"] == ag.sha256_file(files["quran.db"])
    assert identity["binary_sha256"] == ag.sha256_file(files["binary"])
    files["layout.db"].write_text("changed")
    assert ag.harness_identity(harness) != identity
