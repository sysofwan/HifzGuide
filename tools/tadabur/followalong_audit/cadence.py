"""P3, cadence (#309): device logs' preview:window ratio, and the words a latency moves.

The app's scheduler runs a preview every hop and a full window every third, and the device's
inference latency decides how many previews fit in a window. The four h448-era device logs and
the owner's three exports record that cadence; :func:`log_preview_window_ratio` reads the same
facet from them. The App arm is then run at a no-latency floor (0 ms) and at a slow-device
cadence (130 ms), and :func:`cadence_report` asks how many words changed outcome: more than 2%
is a follow-along finding (the latency setting changes what the audit would have said), exactly
2% is not.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from tadabur.followalong_audit.bridge import outcome

#: The share of words that may change outcome between the latency settings.
CADENCE_THRESHOLD = 0.02

#: A device log's update line. The ``type`` is all this reads; the log's own regex
#: (``training.device_logs._UPDATE``) also parses the hop/overlap it renders.
_UPDATE = re.compile(r"Update: type=(\w+),")


def log_preview_window_ratio(text: str) -> float | None:
    """Previews per window among a device log's update lines; ``None`` when it ran no window."""
    counts = {"preview": 0, "window": 0}
    for line in text.splitlines():
        match = _UPDATE.search(line)
        if match is not None and match[1] in counts:
            counts[match[1]] += 1
    if not counts["window"]:
        return None
    return counts["preview"] / counts["window"]


def cadence_report(base: Sequence[Mapping], other: Sequence[Mapping], scoring: str) -> dict:
    """The words whose ``scoring`` outcome differs between two App-arm runs of the same clips."""
    left = _outcomes(base, scoring)
    right = _outcomes(other, scoring)
    if set(left) != set(right):
        raise ValueError(f"the two runs hold different words: {len(set(left) - set(right))} only "
                         f"in the base run, {len(set(right) - set(left))} only in the other")
    changed = sum(1 for key in left if left[key] != right[key])
    words = len(left)
    share = changed / words if words else 0.0
    return {"scoring": scoring, "words": words, "changed": changed, "share": share,
            "finding": share > CADENCE_THRESHOLD}


def _outcomes(rows: Sequence[Mapping], scoring: str) -> dict[tuple[str, int], str]:
    found = {(row["item"], row["word"]): outcome(row[scoring]) for row in rows}
    if len(found) != len(rows):
        raise ValueError(f"a word appears twice in one run's {scoring} rows")
    return found


# --- the device logs and the latency runs -------------------------------------------------------

#: The latency settings the App arm is run at for P3: the no-latency floor and a slow device.
P3_LATENCIES = (0, 130)
#: The seed the P3 subset is drawn under (the issue's 500-clip subset; ``app-run --size 500``).
P3_SEED = 309


def h448_ratios(commit: str | None = None) -> dict[str, float | None]:
    """The h448-era test logs' previews per window, from the Muraja checkout's ``tools/test_logs``.

    The inventory names which logs those are (``model == h448``); the files live in the checkout
    ``build.sh`` made, not in this repo, and are read, never replayed.
    """
    from training.device_logs import H448, load_inventory, log_dir
    from training.muraja_policy import MURAJA_COMMIT

    directory = log_dir(commit or MURAJA_COMMIT)
    ratios = {}
    for entry in load_inventory():
        if entry.model != H448:
            continue
        path = directory / f"{entry.log}.log"
        ratios[entry.log] = log_preview_window_ratio(path.read_text(encoding="utf-8"))
    return ratios


def owner_ratios(paths: Sequence[Path], notes) -> dict[str, float | None]:
    """The owner's three exports' per-session previews per window, refused on a SHA mismatch."""
    from training.device_exports import read_exports

    _, sessions = read_exports(list(paths), notes)
    return {session.name: log_preview_window_ratio(session.text) for session in sessions}


def p3_section(zero: Sequence[Mapping], other: Sequence[Mapping], *,
               device_ratios: Mapping[str, float | None],
               latencies: Sequence[int] = P3_LATENCIES,
               seed: int | None = None) -> dict:
    """The P3 readout: the device ratios and the two latency runs' changed outcomes.

    A clip one latency run failed has no rows, so it is left out of the comparison and listed in
    ``skipped`` rather than aborting the report; ``cadence_report`` itself still refuses two runs
    whose word sets differ, since comparing a subset silently would read better than it is.
    """
    from training.muraja_policy import AUDIT_SCORINGS

    zero_items = {row["item"] for row in zero}
    other_items = {row["item"] for row in other}
    common = zero_items & other_items
    zero = [row for row in zero if row["item"] in common]
    other = [row for row in other if row["item"] in common]
    return {
        "device_ratios": dict(sorted(device_ratios.items())),
        "latencies_ms": list(latencies),
        "clips": len(common),
        "seed": seed,
        "skipped": sorted(zero_items ^ other_items),
        "cadence": {scoring.name: cadence_report(zero, other, scoring.name)
                    for scoring in AUDIT_SCORINGS},
    }
