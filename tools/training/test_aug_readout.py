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
    assert ar.model_spec("candidate_ema") == ModelSpec(ar.ARMS["candidate"], True, 10_000)
    assert ar.model_spec("control_live") == ModelSpec(ar.ARMS["control"], False, 10_000)
    assert ar.model_spec("h448") is DEFAULT_SPECS["h448"]
    with pytest.raises(ValueError):
        ar.model_spec("candidate_best")
    # The truth baseline's identity records stay exactly what they were.
    assert DEFAULT_SPECS["h448"].identity_extra() == {}
    assert ar.model_spec("candidate_ema").identity_extra() == {"weights": "ema", "step": 10_000}


def test_before_the_choice_only_the_candidate_s_look_half_may_be_read():
    assert ar.allowed_models(ar.TARTEEL_LOOK, None) == ("candidate_live", "candidate_ema")
    for name in set(ar.SETS) - {ar.TARTEEL_LOOK}:
        with pytest.raises(SystemExit, match="only after the look half"):
            ar.allowed_models(name, None)


def test_after_the_choice_only_the_chosen_weights_are_read():
    choice = {"weights": "ema"}
    assert ar.allowed_models(ar.TARTEEL_CONFIRM, choice) == ("candidate_ema", "control_ema")
    assert ar.allowed_models(ar.TADABUR_SET, choice) == ("candidate_ema", "control_ema", "h448",
                                                         "base")


def test_the_decoder_refuses_what_the_order_does_not_allow(monkeypatch):
    monkeypatch.setattr(ard, "read_choice", lambda: None)
    ard.check_models(ar.TARTEEL_LOOK, ["candidate_live", "candidate_ema"])
    with pytest.raises(SystemExit):
        ard.check_models(ar.TARTEEL_LOOK, ["control_live"])
    with pytest.raises(SystemExit):
        ard.check_models(ar.QUD_SET, ["candidate_live"])
    monkeypatch.setattr(ard, "read_choice", lambda: {"weights": "live"})
    ard.check_models(ar.QUD_SET, ["candidate_live", "control_live"])
    with pytest.raises(SystemExit):
        ard.check_models(ar.QUD_SET, ["candidate_ema"])
    with pytest.raises(SystemExit):
        ard.check_models(ar.QUD_SET, ["h448"])  # its QUD decodes are committed


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
    assert out[ar.STREAM] == Fake().decode_stream(span, block=0, flush_tail=True)
    rows = Fake().window_rows(window_audio(span))
    assert out[ar.BLANKS] == [sum(int((r == BLANK_ID).sum()) for r in rows), 125 * len(rows)]


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


# --- identity -------------------------------------------------------------------------------------


def _record(model: str, set_name: str) -> dict:
    spec = ar.model_spec(model)
    committed = ar.baseline_identities()
    identity = committed[model] if model in ar.BASELINES else {
        "model_ref": spec.ref, "checkpoint_sha256": "f" * 64, **spec.identity_extra()}
    return {"model": model, "identity": identity,
            "fingerprints": ar._expected_fingerprints(spec.ref)}


def test_identity_records_are_checked_against_the_protocol_the_step_and_the_choice():
    records = {f"{ar.TARTEEL_LOOK}/candidate_live": _record("candidate_live", ar.TARTEEL_LOOK),
               f"{ar.TARTEEL_LOOK}/candidate_ema": _record("candidate_ema", ar.TARTEEL_LOOK),
               f"{ar.QUD_SET}/candidate_ema": _record("candidate_ema", ar.QUD_SET),
               f"{ar.TADABUR_SET}/h448": _record("h448", ar.TADABUR_SET)}
    out = ar.verify_models(records, {"weights": "ema"})
    assert out["candidate_ema"]["step"] == 10_000
    with pytest.raises(ValueError, match="did not pick"):
        ar.verify_models(records, {"weights": "live"})
    wrong = json.loads(json.dumps(records))
    wrong[f"{ar.QUD_SET}/candidate_ema"]["identity"]["step"] = 8_000
    with pytest.raises(ValueError, match="step"):
        ar.verify_models(wrong, {"weights": "ema"})
    wrong = json.loads(json.dumps(records))
    wrong[f"{ar.QUD_SET}/candidate_ema"]["identity"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="one checkpoint"):
        ar.verify_models(wrong, {"weights": "ema"})
    wrong = json.loads(json.dumps(records))
    wrong[f"{ar.TADABUR_SET}/h448"]["fingerprints"]["checks"]["batch_size"] = 32
    with pytest.raises(ValueError, match="fingerprints"):
        ar.verify_models(wrong, {"weights": "ema"})


def test_every_reused_baseline_decode_is_one_model_under_the_frozen_protocol():
    found = ar.baseline_identities()
    assert set(found) == {"base", "h448"}
    assert found["h448"]["model_ref"].endswith("h448_stream/checkpoint.pt")


def test_a_readout_decode_must_read_the_audio_the_base_decode_read(tmp_path, monkeypatch):
    import gzip

    item = _item(0)
    line = _line(item, CLEAN)
    path = tmp_path / "x.jsonl.gz"
    path.write_bytes(gzip.compress((json.dumps({**line, "audio_sha256": "0" * 64}) + "\n")
                                   .encode()))
    monkeypatch.setattr(ar, "decodes_file", lambda set_name, model: path)
    with pytest.raises(ValueError, match="not the audio"):
        ar.read_readout(ar.QDAT_SET, "candidate_ema", {item.key: line}, {item.key})
    path.write_bytes(gzip.compress((json.dumps(line) + "\n").encode()))
    assert ar.read_readout(ar.QDAT_SET, "candidate_ema", {item.key: line}, {item.key})
    with pytest.raises(ValueError, match="not decoded"):
        ar.read_readout(ar.QDAT_SET, "candidate_ema", {item.key: line}, {item.key, "y#0"})


# --- look and confirm, end to end on synthetic sets --------------------------------------------


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
    """Every set synthetic: names are bound to roles by the weights in :attr:`weights`."""

    def __init__(self, weights: str = "ema"):
        self.weights = weights
        self.read: list[str] = []
        # Eleven clusters, so no pattern below lines up with them.
        self.items = [_item(i, clusters=11, clip=f"{i:03d}_{i}_c.wav") for i in range(4 * N)]

    def _role(self, model: str) -> str:
        return model if model in ar.BASELINES else model.rsplit("_", 1)[0]

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
        roles = (*models, *ar.BASELINES)
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

    def harness_build(self):
        return {"muraja_commit": "test"}


def test_look_reads_the_look_half_once_and_writes_the_choice(tmp_path):
    sources = FakeSources()
    look, choice = tmp_path / "look.json", tmp_path / "choice.json"
    written = ar.run_look(sources, look_path=look, choice_path=choice)
    assert sources.read == ["tarteel_v1.look"]
    assert written["weights"] == "live"  # live and EMA tie here; a tie goes to live
    inputs = json.loads(look.read_text())["inputs"]
    assert set(inputs) == {"candidate_live", "candidate_ema"}
    assert json.loads(choice.read_text())["models"] == {"candidate": "candidate_live",
                                                        "control": "control_live"}
    with pytest.raises(SystemExit, match="read once"):
        ar.run_look(sources, look_path=look, choice_path=choice)


def test_confirm_refuses_without_a_committed_choice_and_after_a_verdict(tmp_path):
    paths = {"choice_path": tmp_path / "choice.json", "report_path": tmp_path / "report.json",
             "verdict_path": tmp_path / "verdict.json", "models_path": tmp_path / "models.json"}
    with pytest.raises(SystemExit, match="committed"):
        ar.run_confirm(FakeSources(), committed=lambda path: True, **paths)
    paths["choice_path"].write_text(json.dumps({"weights": "ema"}))
    with pytest.raises(SystemExit, match="committed"):
        ar.run_confirm(FakeSources(), committed=lambda path: False, **paths)
    paths["verdict_path"].write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        ar.run_confirm(FakeSources(), committed=lambda path: True, **paths)


def test_confirm_reads_every_set_once_and_writes_the_verdict(tmp_path):
    sources = FakeSources("ema")
    paths = {"choice_path": tmp_path / "choice.json", "report_path": tmp_path / "report.json",
             "verdict_path": tmp_path / "verdict.json", "models_path": tmp_path / "models.json"}
    paths["choice_path"].write_text(json.dumps({"weights": "ema"}))
    paths["models_path"].write_text(json.dumps({
        f"{ar.QDAT_SET}/candidate_ema": _record("candidate_ema", ar.QDAT_SET)}))
    report = ar.run_confirm(sources, committed=lambda path: True, **paths)
    assert sorted(sources.read) == sorted(["tarteel_v1.confirm", "qdat", "qud_half", "retasy",
                                           "tadabur"])
    main = report["main"]["tarteel_v1_confirm"]
    assert set(main["models"]) == {"candidate", "control", "h448", "base"}
    assert main["models"]["h448"]["point"] == pytest.approx(250.0)
    assert main["models"]["candidate"]["point"] == pytest.approx(125.0)
    verdict = json.loads(paths["verdict_path"].read_text())
    assert verdict["verdict"] == PASS, [c for c in verdict["conditions"] if c["verdict"] != PASS]
    assert verdict["attribution"]["reading"] == av.BEATS_CONTROL
    assert report["leniency"]["blank_ratio"]["models"]["candidate"]["point"] == pytest.approx(64.0)
    from training.aug_readout_doc import render

    text = render({"items": 1, "rule": ar.CHOICE_RULE, "choice": "ema", "inputs": {
        "candidate_ema": {"graded_words": 2, "flagged_words": 1, "false_flags_per_1000": 500.0,
                          "word_agreement_points": 50.0}}}, json.loads(paths["report_path"]
                                                                      .read_text()))
    assert "## Verdict: pass" in text and "candidate−control" in text


def test_a_choice_counts_as_committed_only_when_tracked_and_unchanged(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    path = tmp_path / "choice.json"
    path.write_text("{}")
    assert not ar.is_committed(path)
    git("add", "choice.json")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    assert ar.is_committed(path)
    path.write_text('{"weights": "live"}')
    assert not ar.is_committed(path)


def test_a_checkpoint_at_another_step_is_refused_before_it_is_loaded(monkeypatch, tmp_path):
    from training import external_decode as ed

    monkeypatch.setattr(ed, "checkpoint_step", lambda path: 8_000)
    with pytest.raises(ValueError, match="holds step 8000, not 10000"):
        ed.load_decoders(lambda model: tmp_path / f"{model}.json",
                         {"candidate_ema": ar.model_spec("candidate_ema")})
