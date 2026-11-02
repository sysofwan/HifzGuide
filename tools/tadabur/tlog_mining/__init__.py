"""Phase A's word times (#263): the base teacher's CTC alignment of the gating references.

One clip, whole, through the base teacher; its filename ayah's spaceless phoneme reference as
the transcript; :func:`training.ctc_paths.ctc_best_path` for where each phoneme sits; the
phonetizer's per-word offsets for where one Uthmani word ends and the next begins. The times
depend only on the reference and the teacher's posteriors, never on a decoded mark.

Modules:

``word_times``
    :func:`~tadabur.tlog_mining.word_times.align_words` (pure numpy) and
    :func:`~tadabur.tlog_mining.word_times.build_record`, the committed record of one clip.
``selection``
    Which rows phase A may touch: the gating stratum of ``look``, ``confirm.1`` and
    ``confirm.2``, cross-checked against the committed split tree. Phases B and C are refused.
``batch``
    One batch's work over disjoint whole shards, appending to one part per shard so a killed
    run resumes by key.
``merge``
    The parts into one sorted ``word_times/<tranche>.jsonl.gz`` per tranche plus ``meta.json``.
``give_way``
    The preemptible loop: the GPU lock taken non-blocking for one batch at a time, the PAUSE
    file and the disk/RAM floors checked at every batch boundary.
``__main__``
    The CLI the launcher runs on the box (``align``, ``merge``, ``run``).
"""
