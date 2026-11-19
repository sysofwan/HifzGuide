"""Idle-time mining (#263): phase A's word times, phase B's pool extension and phase C's decode.

Phase A. One clip, whole, through the base teacher; its filename ayah's spaceless phoneme
reference as the transcript; :func:`training.ctc_paths.ctc_best_path` for where each phoneme
sits; the phonetizer's per-word offsets for where one Uthmani word ends and the next begins. The
times depend only on the reference and the teacher's posteriors, never on a decoded mark.

Phase B. The labelled ``reported:*`` rows of confirm.1/2, decoded by the teacher, C95 and round
2's EMA candidate into ``checks`` and ``spans``.

Phase C. Round 3's chosen weights alone (``r3_candidate_ema``, under the round-3 spec), over the
rows round 2's gating decodes and phase B's mined decode hold, so the two line up row by row.

Modules:

``word_times``
    :func:`~tadabur.tlog_mining.word_times.align_words` (pure numpy) and
    :func:`~tadabur.tlog_mining.word_times.build_record`, the committed record of one clip.
``selection``
    Which rows each phase may touch: phase A's gating stratum of ``look``, ``confirm.1`` and
    ``confirm.2``; phase B's labelled reported rows of the two confirm tranches, with the
    un-phonetizable ones listed (#248); phase C's gating rows and phase B's. Every mined phase
    requires slice 1's amendment at its pinned SHA-256, and phase C also round 3's committed
    choice at its pinned checkpoint and round 3's committed ``pass`` verdict.
``batch``
    Phase A's batch: disjoint whole shards per worker, appending to one part per shard so a
    killed run resumes by key.
``mined_decode``
    A mined phase's batch: the rows through the phase's models — phase B's three round-2
    producers, or phase C's round-3 weights alone — into both stores, with the budget checked
    between rows and every model pending-tested before a row is done.
``merge``
    The parts into the phase's committed files: ``word_times/<tranche>.jsonl.gz`` plus
    ``meta.json`` (phase A); ``decodes/tlog.mined.<model>.jsonl.gz`` plus ``meta.json`` (phase
    B); or, for phase C, ``tlog.mined.<model>.jsonl.gz``, ``tlog.gating.<model>.jsonl.gz`` and
    its own ``meta.<model>.json``.
``give_way``
    The preemptible loop: the GPU lock taken non-blocking for one batch at a time, the PAUSE
    file and the disk/RAM floors checked at every batch boundary.
``__main__``
    The CLI the launcher runs on the box (``align``, ``decode``, ``run``, ``merge``).
"""
