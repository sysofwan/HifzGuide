"""One clip's engine session (#307): its request, and how dense its checks were.

Both arms grade a clip as the app would: the session starts on the clip's own ayah at word 1
(decision 1, #307) and reports every phoneme word of that ayah, under Muraja's strict mode and
its shipped (balanced) one. The checks are handed to the harness unchanged — they are the arm
itself, and a request that rewrote them would grade something else.
"""

from __future__ import annotations

from collections.abc import Sequence

from training.muraja_policy import AUDIT_SCORINGS, engine_runs


def session_request(item: str, surah: int, ayah: int, word_count: int,
                    checks: Sequence[dict]) -> dict:
    """The harness request for one clip's checks, over words ``1..word_count``."""
    return {
        "item": item,
        "surah": surah,
        "ayah": ayah,
        "start_word": 1,
        "report_words": list(range(1, word_count + 1)),
        "checks": list(checks),
        "scorings": engine_runs(AUDIT_SCORINGS),
    }


def preview_window_counts(checks: Sequence[dict]) -> dict[str, int]:
    """Previews and full windows in ``checks``: flushes and the closing stop are neither."""
    counts = {"preview": 0, "window": 0}
    for check in checks:
        if check["flush"] or check.get("stop"):
            continue
        counts["window" if check["hop"] else "preview"] += 1
    return counts


def preview_window_ratio(checks: Sequence[dict]) -> float | None:
    """Previews per full window in ``checks``, or ``None`` when the clip ran no window.

    The ratio is the facet of the cadence a device-representative latency moves most; flushes
    and the closing stop are neither preview nor window, so they are left out.
    """
    counts = preview_window_counts(checks)
    if not counts["window"]:
        return None
    return counts["preview"] / counts["window"]
