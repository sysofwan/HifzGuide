"""A shaddah-probe work directory is bound to the run that made it (torch-free)."""

from __future__ import annotations

import pytest

from training import shaddah_probe_run as run


def _segments(n: int) -> list[run.PoolSegmentRef]:
    return [
        run.PoolSegmentRef(f"c{i}.wav#0", f"c{i}.wav", 0, 16000, "رَببِ", i, 0.5, (0,))
        for i in range(n)
    ]


@pytest.fixture
def checkpoints(tmp_path):
    base, student = tmp_path / "base.pt", tmp_path / "h448.pt"
    base.write_bytes(b"teacher weights")
    student.write_bytes(b"student weights v1")
    return {run.BASE: str(base), run.H448: str(student)}


def test_the_same_run_resumes(tmp_path, checkpoints):
    work = tmp_path / "work"
    work.mkdir()
    identity = run.run_identity(checkpoints, _segments(3), "A")
    run.bind_work_dir(work, identity)
    (work / "observations_base.json").write_text("{}")
    run.bind_work_dir(work, run.run_identity(checkpoints, _segments(3), "A"))  # resume: accepted


def test_a_changed_checkpoint_is_refused(tmp_path, checkpoints):
    work = tmp_path / "work"
    work.mkdir()
    run.bind_work_dir(work, run.run_identity(checkpoints, _segments(3), "A"))
    (work / "posteriors_h448.npz").write_bytes(b"cached")

    with open(checkpoints[run.H448], "wb") as f:  # same path, other weights
        f.write(b"student weights v2")
    with pytest.raises(SystemExit, match="models"):
        run.bind_work_dir(work, run.run_identity(checkpoints, _segments(3), "A"))


def test_a_smoke_work_dir_is_not_resumed_as_the_full_run(tmp_path, checkpoints):
    work = tmp_path / "work"
    work.mkdir()
    run.bind_work_dir(work, run.run_identity(checkpoints, _segments(2), "A"))  # --limit 2
    (work / "posteriors_base.npz").write_bytes(b"cached")
    with pytest.raises(SystemExit, match="segments"):
        run.bind_work_dir(work, run.run_identity(checkpoints, _segments(5), "A"))


def test_outputs_of_unknown_provenance_are_refused(tmp_path, checkpoints):
    work = tmp_path / "work"
    work.mkdir()
    (work / "stretch_base.json").write_text("{}")
    with pytest.raises(SystemExit, match="no run_identity.json"):
        run.bind_work_dir(work, run.run_identity(checkpoints, _segments(1), "A"))


def test_a_model_is_identified_by_its_content(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert run.model_content(str(a)) == run.model_content(str(b))
    b.write_bytes(b"other")
    assert run.model_content(str(a)) != run.model_content(str(b))


def test_a_moved_hub_branch_cannot_mix_revisions(tmp_path, checkpoints, monkeypatch):
    """Bound at hub commit A, a resume after upstream moved to B is refused, and every
    stage's decoder is loaded at A whatever the default branch now says."""
    from training import decoding

    work = tmp_path / "work"
    work.mkdir()
    run.bind_work_dir(work, run.run_identity(checkpoints, _segments(3), "A"))
    (work / "posteriors_base.npz").write_bytes(b"made at A")

    with pytest.raises(SystemExit, match="hub_revision"):
        run.bind_work_dir(work, run.run_identity(checkpoints, _segments(3), "B"))

    loads = []
    monkeypatch.setattr(
        decoding.Decoder, "load", classmethod(lambda cls, ref, device, **kw: loads.append((ref, kw)))
    )
    run.load_decoder(run.BASE, checkpoints[run.H448], work)
    run.load_decoder(run.H448, checkpoints[run.H448], work)
    assert [kw["revision"] for _, kw in loads] == ["A", "A"]
    assert loads[1][0] == checkpoints[run.H448]
