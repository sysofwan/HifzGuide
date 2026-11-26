"""The S5 scorer (#311 PR2): verdicts in, attribution, weighted rates and the decision rules out.

Reads the committed draw (``s5_draw/key.jsonl``, ``arms.jsonl.gz``, ``reproduction.json``,
``app_errors.jsonl.gz``), the committed App and oracle tables and the oracle sweep's report, plus
the verdicts file the listening page writes. It **never** writes into ``s5_draw/``: the draw is
frozen, and a scorer that rewrote it would make every earlier number unreadable.

What it computes is the frozen design's (preregistrations §5-§7, draw amendment §4,§5,§7-§10):

* one verdict per served id; the **first** verdict of an original is the one that counts, and a
  repeat only feeds agreement (amendment §4);
* each decisive word through the locked attribution buckets (:mod:`attribution`), with its F1-align
  reference the oracle's own canonical text (:func:`s5_arms.oracle_reference`);
* the inverse-probability-weighted false-flag rate and miss rates, each with a whole-clip bootstrap
  interval (prereg §7, amendment §10) — the edge words are each clip's first and last word of its
  ayah range, read from the App word table, per the §6 erratum;
* the decision rules (prereg §6, amendment §5,§7,§8), including rule 4's "direction only" under 60
  confirmed misses.

No verdicts exist yet: an empty or missing file is a run that attributes nothing and reports
``None`` rates and "direction only", never a division by zero.

Usage (from ``tools/``)::

  python -m tadabur.followalong_audit s5-score --verdicts s5_draw/verdicts.jsonl --out runs/s5
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from tadabur.followalong_audit.app_arm import read_jsonl, read_jsonl_gz
from tadabur.followalong_audit.atomic import write_text_atomically
from tadabur.followalong_audit.attribution import (
    FOLLOW_ALONG, MODEL, UNKNOWN, error_key, false_flag_bucket, kept_outcome, miss_labels,
    model_share, reference_suspects, rule_five, side,
)
from training.muraja_policy import SHIPPED_DEFAULT, STRICT, Scoring

AUDIT = Path(__file__).resolve().parent
DRAW_DIR = AUDIT / "s5_draw"
APP_ARM = AUDIT / "app_arm"
ORACLE_SWEEP = AUDIT / "oracle_sweep"
S2_REPORT = AUDIT.parents[2] / "docs" / "followalong-oracle-sweep.json"

#: The listening page's vocabulary (prereg §4). PR3's UI writes exactly these.
VERDICTS = ("correct", "mistake", "not sure", "different text", "unusable audio")
#: What a mistake's owner ticked, from the same vocabulary.
MISTAKE_KINDS = ("letter", "haraka", "shaddah", "skipped", "added")
#: The one withdrawal a verdict can carry: the word is not in the audio at all.
SUB_REASONS = ("not_in_clip",)
#: The verdicts file beside the worklist.
VERDICTS_PATH = DRAW_DIR / "verdicts.jsonl"
VERDICT_FIELDS = frozenset({"id", "verdict", "kinds", "sub_reason", "whole_played", "note"})
#: The verdicts that decide nothing about a word's correctness.
_EXCLUDED = ("not sure", "different text", "unusable audio")
#: The miss-rate scopes: the headline set H, the two labelled bounds, and the full frame.
SCOPES = ("headline", "s4", "s5_edge", "full")
#: The F-ref bucket's side marker: it is read on both sides (rule 3 reports each), so an
#: attribution carries neither side's name for it.
F_REF_SIDE = "f-ref"
#: Rule 1's buckets, in precedence order; M-nograde counts only on the follow-along side.
RULE_ONE_BUCKETS = ("F1", "F1-align", "M1r", "M1", "M1u", "M-nograde")
#: Rule 4's threshold (prereg §6).
MISS_THRESHOLD = 60


# --- the verdicts file ---------------------------------------------------------------------------

def read_verdicts(path: Path, key: Sequence[Mapping]) -> dict[str, dict]:
    """The verdicts file's rows by served id; a missing file is no verdicts.

    Every row must have exactly :data:`VERDICT_FIELDS`, a verdict from :data:`VERDICTS`, mistake
    kinds exactly when it is a mistake, a sub-reason only on ``unusable audio``, and an id the
    key holds. One id answered twice is refused: two answers cannot both be first.
    """
    target = Path(path)
    ids = {row["id"] for row in key}
    if not target.exists():
        return {}
    verdicts: dict[str, dict] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        _check_verdict(row, ids)
        if row["id"] in verdicts:
            raise ValueError(f"{row['id']} is answered twice")
        verdicts[row["id"]] = row
    return verdicts


def _check_verdict(row: Mapping, ids: set[str]) -> None:
    if not isinstance(row, Mapping) or set(row) != VERDICT_FIELDS:
        raise ValueError(f"a verdict's fields are exactly {sorted(VERDICT_FIELDS)}, not "
                         f"{sorted(row) if isinstance(row, Mapping) else row!r}")
    if row["id"] not in ids:
        raise ValueError(f"{row['id']!r} is not a served id of this draw")
    verdict = row["verdict"]
    if verdict not in VERDICTS:
        raise ValueError(f"{row['id']}: {verdict!r} is not a verdict: {VERDICTS}")
    kinds = row["kinds"]
    if not isinstance(kinds, list) or any(kind not in MISTAKE_KINDS for kind in kinds) or \
            len(set(kinds)) != len(kinds):
        raise ValueError(f"{row['id']}: the mistake kinds are a duplicate-free subset of "
                         f"{MISTAKE_KINDS}, not {kinds!r}")
    if (verdict == "mistake") != bool(kinds):
        raise ValueError(f"{row['id']}: kinds are non-empty exactly for a mistake")
    if row["sub_reason"] is not None and (row["sub_reason"] not in SUB_REASONS
                                         or verdict != "unusable audio"):
        raise ValueError(f"{row['id']}: a sub-reason is {SUB_REASONS} on unusable audio only")
    if not isinstance(row["whole_played"], bool):
        raise ValueError(f"{row['id']}: whole_played is a bool")
    if not isinstance(row["note"], str):
        raise ValueError(f"{row['id']}: note is a string")


def agreement(key: Sequence[Mapping], verdicts: Mapping[str, Mapping]) -> dict:
    """How many repeat pairs the listener answered the same way, top-level verdict only."""
    pairs = agree = 0
    for row in key:
        if row["repeat_of"] is None or row["id"] not in verdicts:
            continue
        original = verdicts.get(row["repeat_of"])
        if original is None:
            continue
        pairs += 1
        agree += original["verdict"] == verdicts[row["id"]]["verdict"]
    return {"pairs": pairs, "agree": agree}


# --- one listened word ---------------------------------------------------------------------------

def attribute_listened(key_row: Mapping, arms: Mapping, verdict: Mapping, *,
                       suspects: Iterable[tuple] = (), scoring: Scoring = STRICT) -> dict:
    """One decisive word through the locked buckets, as the committed attribution line.

    ``suspects`` is :func:`attribution.reference_suspects`' set, and F-ref is placed on the
    follow-along side (both sides are reported by rule 3). The line carries the key's stratum,
    cell and π, the verdict the owner gave, both arms' kept outcomes, and the outcome, bucket,
    labels and side the verdict routes to — everything but the raw traces. A word the App
    flagged and the owner heard correct is a false flag (F1..F3); a mistake the App flagged is a
    correct catch; any other mistake is a miss labelled by :func:`attribution.miss_labels`, its
    first label its bucket. Neither side needs the arm the bucket names: the App cell is the
    row's own.
    """
    app = arms["app"][scoring.name]
    oracle = arms["oracle"][scoring.name]
    teacher = arms["teacher"][scoring.name]
    app_outcome = kept_outcome(app, scoring)
    teacher_outcome = kept_outcome(teacher, scoring)
    result = {
        "id": key_row["id"], "item": key_row["item"], "word": key_row["word"],
        "stratum": key_row["stratum"], "cell": key_row["cell"],
        "pi": key_row["inclusion_probability"], "verdict": verdict["verdict"],
        "app": app_outcome, "teacher": teacher_outcome, "outcome": None, "bucket": None,
        "labels": [], "side": None, "soft_or_target": False,
    }
    if verdict["verdict"] in _EXCLUDED:
        result["outcome"] = "excluded"
        return result
    if verdict["verdict"] == "correct":
        if app_outcome != "flagged":
            result["outcome"] = "true_pass"
            return result
        strict_suspect = _is_suspect(key_row, arms["app"][STRICT.name], suspects)
        bucket = false_flag_bucket(app=app, oracle=oracle, teacher=teacher,
                                   reference=arms["reference"], suspect=strict_suspect,
                                   scoring=scoring)
        result.update(outcome="false_flag", bucket=bucket, labels=[bucket],
                      side=_side(bucket, teacher_outcome, strict_suspect),
                      soft_or_target=_soft_or_target(arms["causes"][scoring.name]))
        return result
    if app_outcome == "flagged":
        result["outcome"] = "correct_catch"
        return result
    labels = miss_labels(app=app, teacher=teacher, scoring=scoring)
    bucket = labels[0]
    result.update(outcome="miss", bucket=bucket, labels=labels,
                  side=_side(bucket, teacher_outcome, False))
    return result


def _side(bucket: str, teacher_outcome: str, suspect: bool) -> str:
    """The bucket's side; F-ref carries :data:`F_REF_SIDE` because rule 3 reads it both ways."""
    if bucket == "F-ref" or suspect:
        return F_REF_SIDE
    return side(bucket, teacher_graded=teacher_outcome != "ungraded")


def _soft_or_target(causes: Iterable[str]) -> bool:
    """Whether a flag rests on the soft or target pairs (prereg §8's allowance)."""
    return any(cause in ("soft_pair", "target_pair") for cause in causes)


def is_reference_suspect(key_row: Mapping, arms: Mapping, suspects: Iterable[tuple]) -> bool:
    """Whether the word's App strict deviation is a reference suspect (F-ref).

    The strict trace decides, whatever scoring the attribution reads: a suspect is a deviation
    the regenerated App run repeated in three tranche groups, and a word with no errors is never
    one.
    """
    return _is_suspect(key_row, arms["app"][STRICT.name], suspects)


def _is_suspect(key_row: Mapping, app: Mapping | None, suspects: Iterable[tuple]) -> bool:
    if app is None or not app["errors"]:
        return False
    surah, ayah = (int(part) for part in key_row["surah_ayah"].split(":"))
    return (surah, ayah, key_row["word"], error_key(app["errors"])) in set(suspects)


# --- the rates -----------------------------------------------------------------------------------

def _weight(record: Mapping) -> float:
    return 1.0 / record["pi"]


def _kept(records: Sequence[Mapping], *, without_edges: bool) -> list[Mapping]:
    return [record for record in records if not (without_edges and record["ayah_edge"])]


def false_flags_per_1000(records: Sequence[Mapping], *, without_edges: bool = False
                         ) -> float | None:
    """False flags per 1,000 verdict-correct words, inverse-probability weighted, all strata."""
    kept = [record for record in _kept(records, without_edges=without_edges)
            if record["verdict"] == "correct"]
    weight = sum(_weight(record) for record in kept)
    if not weight:
        return None
    return 1000.0 * sum(_weight(record) for record in kept if record["app_flagged"]) / weight


def _in_scope(record: Mapping, scope: str) -> bool:
    stratum, cell = record["stratum"], record["cell"]
    if scope == "full":
        return True
    if scope == "s4":
        return stratum == "S4"
    if scope == "s5_edge":
        return stratum == "S5" and cell in ("leading", "trailing")
    if scope == "headline":
        return stratum in ("S1", "S3") or (stratum == "S5" and cell == "interior")
    raise ValueError(f"{scope!r} is not a scope: {SCOPES}")


def misses_per_100(records: Sequence[Mapping], *, scope: str = "headline",
                   without_edges: bool = False) -> float | None:
    """Missed mistakes per 100 mistakes in the scope; a correct catch is denominator only."""
    if scope not in SCOPES:
        raise ValueError(f"{scope!r} is not a scope: {SCOPES}")
    missed = [record for record in _kept(records, without_edges=without_edges)
              if record["verdict"] == "mistake" and _in_scope(record, scope)]
    weight = sum(_weight(record) for record in missed)
    if not weight:
        return None
    return 100.0 * sum(_weight(record) for record in missed
                       if not record["app_flagged"]) / weight


def rate_interval(records: Sequence[Mapping], statistic: Callable[[list], float | None], *,
                  seed: int, iterations: int = 4000, alpha: float = 0.05
                  ) -> list[float | None]:
    """The 95% interval of ``statistic`` over whole-clip resamples (prereg §7).

    :func:`training.decode_evalset.cluster_bootstrap`'s percentile arithmetic, over the same clip
    clusters — with two deliberate differences. The clips are resampled in sorted-key order, not
    the insertion order ``cluster_bootstrap`` walks, so the same seed gives a different (equally
    valid) realisation; sorting keeps the interval independent of the order the rows arrive in. A
    resample the statistic cannot compute (a draw of clips with no word of the rate's kind) is
    left out rather than counted as zero, so an undefined rate stays undefined instead of gaining
    false precision.
    """
    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(record["item"], []).append(record)
    if not grouped:
        return [None, None]
    keys = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sample: list = []
        for _index in keys:
            sample.extend(grouped[keys[rng.randrange(len(keys))]])
        value = statistic(sample)
        if value is not None:
            draws.append(value)
    if not draws:
        return [None, None]
    draws.sort()
    return [draws[int(alpha / 2 * len(draws))],
            draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]]


def ayah_edge_words(app_grades: Iterable[Mapping]) -> set[tuple[str, int]]:
    """Each clip's first and highest word number in the App word table (§6 erratum).

    Read from the table's word numbers alone, never from what the App graded: the sensitivity
    says what the ayah is, not what the engine made of it. A one-word ayah's single word.
    """
    words: dict[str, set[int]] = defaultdict(set)
    for row in app_grades:
        words[row["item"]].add(int(row["word"]))
    return {(item, word) for item, numbers in words.items()
            for word in {min(numbers), max(numbers)}}


# --- the decision rules --------------------------------------------------------------------------

def candidate_pattern_words(report: Mapping) -> int:
    """The most words a candidate S2 pattern covers; 0 when the sweep found none."""
    return max((int(pattern["words"]) for pattern in report.get("patterns", ())
                if pattern.get("candidate")), default=0)


def evaluate_rules(attributions: Sequence[Mapping], *, oracle_pattern_words: int = 0) -> dict:
    """The four rules over attributed words (prereg §6, amendment §5,§7,§8). Never raises on none."""
    decisive = [row for row in attributions if row["outcome"] in ("false_flag", "miss")]
    return {
        "rule_1": _rule_one(decisive, oracle_pattern_words),
        "rule_3": _rule_three(decisive),
        "rule_4": _rule_four(decisive),
        "rule_5": _rule_five(decisive),
    }


def _rule_one(words: Sequence[Mapping], oracle_pattern_words: int) -> dict:
    """Rule 1: three instances in three clips, per bucket, in every stratum."""
    by_bucket: dict[str, list[Mapping]] = defaultdict(list)
    f_ref: list[Mapping] = []
    for row in words:
        if row["bucket"] == "F-ref":
            f_ref.append(row)
        elif row["bucket"] in RULE_ONE_BUCKETS and row["bucket"] == row["labels"][0]:
            if row["bucket"] == "M-nograde" and row["side"] != FOLLOW_ALONG:
                continue
            by_bucket[row["bucket"]].append(row)
    buckets = {}
    for name in RULE_ONE_BUCKETS:
        rows = by_bucket[name]
        clips = len({row["item"] for row in rows})
        buckets[name] = {"instances": len(rows), "clips": clips,
                         "triggered": len(rows) >= 3 and clips >= 3}
    return {"buckets": buckets,
            "f_ref": {"instances": len(f_ref), "clips": len({row["item"] for row in f_ref})},
            "oracle_pattern_words": oracle_pattern_words,
            "oracle_pattern_triggered": oracle_pattern_words >= 10,
            "follow_along_first": (any(entry["triggered"] for entry in buckets.values())
                                   or oracle_pattern_words >= 10)}


def _rule_three(words: Sequence[Mapping]) -> dict:
    """Rule 3: the model side's share of the listened attributed words, before any fix.

    Unweighted, F-ref read on each side in turn, and an M-nograde word neither arm graded is on
    neither side: it is counted by stratum and bounded by the two extremes. Straddling 50% is the
    owner's call, not the readout's.
    """
    unknown = [row for row in words if row["side"] == UNKNOWN]
    known = [row for row in words if row["side"] != UNKNOWN]
    follow_along_share = _model_share(known, FOLLOW_ALONG)
    total = len(known) + len(unknown)
    follow_along_extreme = (None if follow_along_share is None
                            else follow_along_share * len(known) / total)
    model_extreme = (None if follow_along_share is None
                     else (follow_along_share * len(known) + len(unknown)) / total)
    straddles = (follow_along_extreme is not None
                 and follow_along_extreme < 0.5 <= model_extreme)
    return {
        "label": "before any follow-along fix",
        "model_share": {"f_ref_follow_along": follow_along_share,
                        "f_ref_model": _model_share(known, MODEL)},
        "unknown": {"words": len(unknown),
                    "by_stratum": dict(sorted(Counter(row["stratum"] for row in unknown).items()))},
        "extremes": {"unknown_follow_along": follow_along_extreme,
                     "unknown_model": model_extreme},
        "straddles_50": straddles,
        "owner_decides": straddles,
        "weighted_without_s4": _weighted_share(words),
    }


def _model_share(words: Sequence[Mapping], f_ref: str) -> float | None:
    """``attribution.model_share`` of the words, ``None`` when not one has a side."""
    if not words:
        return None
    return model_share((_bucket(row) for row in words), f_ref=f_ref)


def _bucket(row: Mapping) -> str | tuple[str, bool]:
    """The row's bucket as :func:`attribution.model_share` reads it: M-nograde with its grade."""
    if row["bucket"] == "M-nograde":
        return (row["bucket"], row["teacher"] != "ungraded")
    return row["bucket"]


def _bucket_side(row: Mapping, f_ref: str = FOLLOW_ALONG) -> str:
    """The row's side by the bucket rules, F-ref read on ``f_ref``'s side."""
    graded = row["teacher"] != "ungraded" if row["bucket"] == "M-nograde" else None
    return side(row["bucket"], f_ref=f_ref, teacher_graded=graded)


def _weighted_share(words: Sequence[Mapping]) -> float | None:
    """The π⁻¹-weighted model fraction without S4, F-ref on the follow-along side."""
    kept = [row for row in words if row["side"] != UNKNOWN and row["stratum"] != "S4"]
    weight = sum(_weight(row) for row in kept)
    if not weight:
        return None
    model = sum(_weight(row) for row in kept if _bucket_side(row) == MODEL)
    return model / weight


def _rule_four(words: Sequence[Mapping]) -> dict:
    """Rule 4: 60 confirmed misses decide the miss side; under it, direction only."""
    misses = sum(1 for row in words if row["outcome"] == "miss")
    decided = misses >= MISS_THRESHOLD
    return {"confirmed_misses": misses, "threshold": MISS_THRESHOLD, "decides": decided,
            "status": "decided" if decided else "direction only"}


def _rule_five(words: Sequence[Mapping]) -> dict | None:
    """Rule 5: are the false flags learnable, and does strict mode deserve an allowance?"""
    pairs = []
    for row in words:
        if row["outcome"] == "false_flag" and row["bucket"] in ("F2", "F3"):
            pairs.append((row["bucket"], row["teacher"]))
        elif row["outcome"] == "miss" and row["labels"][-1] in ("M2", "M3"):
            pairs.append((row["labels"][-1], row["teacher"]))
    if not pairs:
        return None
    third = [row for row in words if row["bucket"] == "F3"]
    soft = sum(1 for row in third if row["soft_or_target"])
    share = soft / len(third) if third else None
    return {**rule_five(pairs), "soft_or_target_share_of_f3": share,
            "strict_mode_allowance": share is not None and share >= 0.5}


# --- the run -------------------------------------------------------------------------------------

def score(*, draw: Path = DRAW_DIR, verdicts: Path = VERDICTS_PATH, out: Path,
          iterations: int = 4000) -> dict:
    """Score the committed draw against ``verdicts`` and write ``summary.json`` and
    ``attribution.jsonl`` under ``out``. Returns the summary.

    The bootstrap seed is the draw's own (``subseed(draw seed, "bootstrap")``, amendment §10), so a
    later re-run over the same verdicts gives the same intervals.
    """
    from tadabur.followalong_audit.draw import subseed

    key = read_jsonl(Path(draw) / "key.jsonl")
    seeds = json.loads((Path(draw) / "draw_seed.json").read_text(encoding="utf-8"))
    seed = subseed(seeds["seed"], "bootstrap")
    answered = read_verdicts(Path(verdicts), key)
    originals = [row for row in key if row["repeat_of"] is None]
    arms = {(row["item"], row["word"]): row
            for row in read_jsonl_gz(Path(draw) / "arms.jsonl.gz")}
    suspects = reference_suspects(read_jsonl_gz(Path(draw) / "app_errors.jsonl.gz"))
    edges = ayah_edge_words(read_jsonl_gz(APP_ARM / "word_grades.jsonl.gz"))
    patterns = candidate_pattern_words(
        json.loads(S2_REPORT.read_text(encoding="utf-8")) if S2_REPORT.exists() else {})
    heard = [row for row in originals if row["id"] in answered]
    strict = [attribute_listened(row, arms[(row["item"], row["word"])], answered[row["id"]],
                                 suspects=suspects) for row in heard]
    summary = {
        "verdicts": {"served": len(key), "first_verdicts": len(heard),
                     "excluded": _excluded(heard, answered),
                     "agreement": agreement(key, answered)},
        "bootstrap": {"seed": seed, "iterations": iterations},
    }
    blocks = {}
    for scoring in (STRICT, SHIPPED_DEFAULT):
        rows = strict if scoring is STRICT else [
            attribute_listened(row, arms[(row["item"], row["word"])], answered[row["id"]],
                               suspects=suspects, scoring=scoring) for row in heard]
        records = _records(rows, edges)
        blocks[_block_name(scoring)] = {"attribution": _attribution_counts(rows),
                                "rates": _rates(records, seed, iterations),
                                "rules": evaluate_rules(rows, oracle_pattern_words=patterns)}
    summary.update(blocks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomically(out / "summary.json",
                          json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    write_text_atomically(out / "attribution.jsonl", "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(strict, key=lambda row: row["id"])))
    return summary


def _block_name(scoring: Scoring) -> str:
    """The summary's name for a scoring: the shipped default is the ``balanced`` reading."""
    return "balanced" if scoring is SHIPPED_DEFAULT else scoring.name


def _excluded(heard: Sequence[Mapping], answered: Mapping[str, Mapping]) -> dict:
    counts = Counter(answered[row["id"]]["verdict"] for row in heard)
    return {name: counts.get(name, 0) for name in _EXCLUDED} | {
        "not_in_clip": sum(1 for row in heard
                           if answered[row["id"]]["sub_reason"] == "not_in_clip")}


def _records(rows: Sequence[Mapping], edges: set[tuple[str, int]]) -> list[dict]:
    """The rate records: one per attributed word, in the scorer's record shape."""
    return [{"item": row["item"], "word": row["word"], "stratum": row["stratum"],
             "cell": row["cell"], "pi": row["pi"], "verdict": row["verdict"],
             "app_flagged": row["app"] == "flagged",
             "ayah_edge": (row["item"], row["word"]) in edges}
            for row in rows]


def _attribution_counts(rows: Sequence[Mapping]) -> dict:
    flags = Counter(row["bucket"] for row in rows if row["outcome"] == "false_flag")
    labels = Counter(row["bucket"] for row in rows if row["outcome"] == "miss")
    return {"false_flag_buckets": {name: flags.get(name, 0) for name in
                                   ("F1", "F1-align", "F-ref", "F2", "F3")},
            "miss_labels": {name: labels.get(name, 0) for name in
                            ("M1r", "M1", "M1u", "M-nograde", "M2", "M3")},
            "correct_catches": sum(1 for row in rows if row["outcome"] == "correct_catch"),
            "true_passes": sum(1 for row in rows if row["outcome"] == "true_pass")}


def _rate(records: Sequence[Mapping], statistic: Callable[[list], float | None], seed: int,
          iterations: int) -> dict:
    """A rate with its interval; an undefined rate has no interval to report."""
    estimate = statistic(records)
    without = statistic(records, without_edges=True)
    return {"estimate": estimate,
            "interval": (None if estimate is None else
                         rate_interval(records, statistic, seed=seed, iterations=iterations)),
            "without_edges": {
                "estimate": without,
                "interval": (None if without is None else rate_interval(
                    records, _without_edges(statistic), seed=seed, iterations=iterations))}}


def _without_edges(statistic: Callable[..., float | None]) -> Callable[[list], float | None]:
    def without(sample: list) -> float | None:
        return statistic(sample, without_edges=True)
    return without


def _rates(records: Sequence[Mapping], seed: int, iterations: int) -> dict:
    return {"false_flags_per_1000": _rate(records, false_flags_per_1000, seed, iterations),
            "misses_per_100": {
                ("full_frame" if scope == "full" else scope): _rate(
                    records, _scoped(scope), seed, iterations)
                for scope in SCOPES}}


def _scoped(scope: str) -> Callable[..., float | None]:
    def scoped(sample: list, *, without_edges: bool = False) -> float | None:
        return misses_per_100(sample, scope=scope, without_edges=without_edges)
    return scoped
