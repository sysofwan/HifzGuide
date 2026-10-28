"""The tlog overlay's row format as data (#193): overlay v1, documented in ``tlog/README.md``.

Four per-row trees, joined on ``(split, shard_file, row)``: ``index`` (S1's clip identity),
``pass1`` (the audio's facts), ``split`` (the frozen speaker/role assignment) and ``overlap``
(S4's evaluation and duplicate screen). Each stage's rows are exactly :data:`STAGES`' fields;
the committed trees are the source of truth, and ``test_tlog_overlay_doc.py`` checks this
table against every first-shard row and against the README's field lists, so the published
format and the data cannot drift apart.

:data:`PUBLIC_DROPPED` is what the **public subset** (S9, the owner's call to publish) drops,
because it is either a speaker identity, the leakage unit, a teacher's answer or the hold-out
assignment itself. ``nn_train_sim`` is listed because the upstream card defines one; no
committed stage carries it.
"""

from __future__ import annotations

#: Stage directory name -> every field of that stage's rows. Order is the written JSON's.
STAGES = {
    "index": ("audio_path", "filename_label", "id_group", "recording_id", "row",
              "schema_version", "shard_file", "split", "tarteel_v1_any_overlap",
              "tarteel_v1_eval_overlap"),
    "pass1": ("audio_path", "audio_sha256", "codec_ok", "decoder", "duration_s", "row",
              "sample_rate", "schema_version", "shard_file", "speech_ratio", "speech_s",
              "split"),
    "split": ("audio_path", "group_key", "role", "row", "shard_file", "speaker_cluster",
              "split"),
    "overlap": ("duplicate_ber", "duplicate_of", "eval_overlap", "excluded", "row",
                "schema_version", "shard_file", "split"),
}
#: The fields the public subset drops, and the whole tree it also drops: S5's teacher decodes
#: (``ident/decodes/``), the teacher's answers with them.
PUBLIC_DROPPED = ("speaker_cluster", "group_key", "nn_train_sim", "role")
#: The tree the public subset does not carry at all: S5's teacher decodes.
PUBLIC_DROPPED_TREE = "ident/decodes"


def public_subset() -> dict[str, tuple[str, ...]]:
    """Every stage's fields without :data:`PUBLIC_DROPPED`, in :data:`STAGES` order."""
    return {stage: tuple(field for field in fields if field not in PUBLIC_DROPPED)
            for stage, fields in STAGES.items()}
