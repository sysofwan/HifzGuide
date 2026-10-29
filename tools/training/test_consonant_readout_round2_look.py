"""The round-2 look phase (#227): each arm's live/EMA choice on tlog look gating, read once.

``aug_readout.run_look(sources, layout, spec=<round-2 spec>)`` grades the candidate's and the
control's live and EMA decodes on ``Sources.tlog("look", models)``, applies #95's rule to each
arm on its own, writes ``look.json`` and ``choice.json`` and exports the per-word kept grades of
C95 and the chosen candidate. The interface these tests pin:

* ``choice["weights"]`` is ``{"candidate": <live|ema>, "control": <live|ema>}`` and equals
  ``look["choice"]``; ``choice["models"]`` names the chosen ``r2_*`` models and
  ``choice["identities"]`` holds all four arm identities (:func:`training.aug_readout_gate.
  identity_record` of the spec's models);
* the grades land in ``<root>/look_grades.jsonl.gz``, one line per word and model, with at least
  ``key``, ``position``, ``word_index``, ``model``, ``outcome``, ``flagged`` and ``graded``;
* the sitting refuses unless the spec's pre-registration is committed in HEAD.

Every source is synthetic.
"""

from __future__ import annotations

import dataclasses
import gzip
import json

import pytest

from tadabur.tarteel_split import LOOK
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import round2_fixtures as fx

ITEMS = 12
FLAG = fx.consonant_flag("س→ش")
REFUSED = (SystemExit, ValueError)


class Sitting:
    """A look sitting in a repository holding the committed pre-registration.

    ``flagged[model]`` is how many of the 12 items carry a flag on their second word;
    ``texts[model]`` the decode every item gets (the teacher hears the clean reading).
    """

    def __init__(self, tmp_path, flagged: dict[str, int], texts: dict[str, str] | None = None,
                 *, committed: bool = True):
        self.repo = fx.init_repo(tmp_path / "repo")
        prereg = (fx.commit_prereg if committed else fx.write_prereg)(self.repo)
        self.spec = fx.filled_spec(root=self.repo / "readout", prereg=prereg)
        self.layout = ag.Layout(self.spec.root)
        self.layout.root.mkdir()
        self.layout.models.write_text(json.dumps(fx.look_manifest(self.spec)), encoding="utf-8")
        items = [fx.tlog_item(n, 1 + n % 5) for n in range(ITEMS)]
        self.sources = fx.TlogSources(
            tmp_path / "files", items,
            lambda model, i, p: FLAG if p == 1 and i < flagged.get(model, 0) else None,
            texts, spec=self.spec)
        self.grades_path = self.layout.root / "look_grades.jsonl.gz"

    def run(self, spec: asp.RoundSpec | None = None) -> dict:
        return ar.run_look(self.sources, self.layout, spec=spec or self.spec)

    def commit(self) -> None:
        names = [self.layout.look.name, self.layout.choice.name,
                 self.layout.claim(LOOK).name, self.grades_path.name]
        fx.git(self.layout.root, "add", *names)
        fx.git(self.layout.root, "commit", "-qm", "the look")


def _weights(sitting: Sitting) -> dict:
    return sitting.run()["weights"]


# --- each arm chooses on its own -------------------------------------------------------------------


def test_each_arm_takes_the_weights_with_fewer_flags_per_1000_on_its_own(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 6, "r2_candidate_ema": 2,
                                 "r2_control_live": 1, "r2_control_ema": 5})
    assert _weights(sitting) == {"candidate": "ema", "control": "live"}


def test_the_other_way_round_is_chosen_as_independently(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 1, "r2_candidate_ema": 4,
                                 "r2_control_live": 5, "r2_control_ema": 2})
    assert _weights(sitting) == {"candidate": "live", "control": "ema"}


def test_a_tie_in_flags_goes_to_higher_agreement_with_the_teacher(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 3, "r2_candidate_ema": 3,
                                 "r2_control_live": 3, "r2_control_ema": 3},
                      texts={"r2_candidate_live": fx.SWAPPED, "r2_control_ema": fx.SWAPPED})
    # Candidate: ema hears what the teacher hears, live does not. Control: the reverse.
    assert _weights(sitting) == {"candidate": "ema", "control": "live"}
    other = Sitting(tmp_path / "other", {"r2_candidate_live": 3, "r2_candidate_ema": 3,
                                         "r2_control_live": 3, "r2_control_ema": 3},
                    texts={"r2_candidate_ema": fx.SWAPPED, "r2_control_live": fx.SWAPPED})
    assert _weights(other) == {"candidate": "live", "control": "ema"}


def test_a_tie_in_flags_and_agreement_goes_to_live(tmp_path):
    sitting = Sitting(tmp_path, {m: 3 for m in fx.R2_MODELS})
    assert _weights(sitting) == {"candidate": "live", "control": "live"}


# --- what is written -----------------------------------------------------------------------------


def test_the_look_writes_the_choice_with_both_arms_identities(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 6, "r2_candidate_ema": 2,
                                 "r2_control_live": 1, "r2_control_ema": 5})
    choice = sitting.run()
    look = json.loads(sitting.layout.look.read_text(encoding="utf-8"))
    assert json.loads(sitting.layout.choice.read_text(encoding="utf-8")) == choice
    assert choice["issue"] == sitting.spec.issue
    assert choice["weights"] == look["choice"] == {"candidate": "ema", "control": "live"}
    assert choice["models"] == {"candidate": "r2_candidate_ema", "control": "r2_control_live"}
    assert choice["identities"] == look["identities"] == {
        model: ag.identity_record(sitting.spec.models[model]) for model in fx.R2_MODELS}
    assert choice["look_sha256"] == ag.sha256_file(sitting.layout.look)
    assert sitting.sources.read == ["look"]  # tlog look gating only, never a confirm row


def test_the_look_exports_the_per_word_grades_of_c95_and_the_chosen_candidate(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 6, "r2_candidate_ema": 2,
                                 "r2_control_live": 1, "r2_control_ema": 5, "c95": 4})
    sitting.run()
    with gzip.open(sitting.grades_path, "rt", encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    assert {line["model"] for line in lines} == {"c95", "r2_candidate_ema"}
    for model, flagged in (("c95", 4), ("r2_candidate_ema", 2)):
        mine = [line for line in lines if line["model"] == model]
        assert len(mine) == 2 * ITEMS  # every word of every look gating clip
        assert sum(line["flagged"] for line in mine) == flagged
        assert all(line["graded"] for line in mine)
    first = lines[0]
    assert {"key", "position", "word_index", "model", "outcome", "flagged", "graded"} \
        <= set(first)
    keys = {i.key for i in sitting.sources.items}
    assert {line["key"] for line in lines} == keys


def test_a_second_look_is_refused_once_the_choice_exists(tmp_path):
    sitting = Sitting(tmp_path, {m: 1 for m in fx.R2_MODELS})
    sitting.run()
    with pytest.raises(SystemExit, match="read"):
        sitting.run()


# --- gated sets read only after the choice is committed ------------------------------------------


def test_every_gated_set_refuses_until_the_choice_is_committed(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 6, "r2_candidate_ema": 2,
                                 "r2_control_live": 1, "r2_control_ema": 5})
    sitting.run()
    with pytest.raises(SystemExit, match="not committed"):
        ag.committed_choice(sitting.layout, spec=sitting.spec)
    sitting.commit()
    committed = ag.committed_choice(sitting.layout, spec=sitting.spec)
    assert committed.choice["weights"] == {"candidate": "ema", "control": "live"}
    allowed = set(ag.allowed_models("tlog.confirm", committed.choice, sitting.spec))
    assert {"r2_candidate_ema", "r2_control_live", "c95"} <= allowed
    assert not allowed & {"r2_candidate_live", "r2_control_ema"}


def test_an_edited_choice_cannot_pick_weights_the_look_did_not(tmp_path):
    sitting = Sitting(tmp_path, {"r2_candidate_live": 6, "r2_candidate_ema": 2,
                                 "r2_control_live": 1, "r2_control_ema": 5})
    sitting.run()
    choice = json.loads(sitting.layout.choice.read_text(encoding="utf-8"))
    choice["weights"]["control"] = "ema"
    choice["models"]["control"] = "r2_control_ema"
    sitting.layout.choice.write_text(ag.canonical(choice), encoding="utf-8")
    sitting.commit()
    with pytest.raises(SystemExit, match="weights"):
        ag.committed_choice(sitting.layout, spec=sitting.spec)


# --- what it refuses -------------------------------------------------------------------------------


def test_the_look_refuses_without_a_committed_pre_registration_and_grades_nothing(tmp_path):
    sitting = Sitting(tmp_path, {m: 1 for m in fx.R2_MODELS}, committed=False)
    with pytest.raises(SystemExit, match="committed"):
        sitting.run()
    assert sitting.sources.read == [] and sitting.sources.grade_calls == 0
    for path in (sitting.layout.look, sitting.layout.choice, sitting.layout.claim(LOOK),
                 sitting.grades_path):
        assert not path.exists()


def test_the_look_refuses_a_round_whose_arms_are_not_filled_in(tmp_path):
    sitting = Sitting(tmp_path, {m: 1 for m in fx.R2_MODELS})
    unfilled = dataclasses.replace(asp.CONSONANT2, root=sitting.spec.root,
                                   prereg_path=sitting.spec.prereg_path, pins=sitting.spec.pins)
    with pytest.raises(REFUSED, match="r2_"):
        sitting.run(unfilled)
    assert sitting.sources.grade_calls == 0 and not sitting.layout.choice.exists()


def test_the_look_needs_an_identity_for_every_arm_weight_before_it_reads(tmp_path):
    sitting = Sitting(tmp_path, {m: 1 for m in fx.R2_MODELS})
    records = json.loads(sitting.layout.models.read_text(encoding="utf-8"))
    del records["tlog.look/r2_control_live"]
    sitting.layout.models.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="r2_control_live"):
        sitting.run()
    assert sitting.sources.read == [] and sitting.sources.grade_calls == 0
