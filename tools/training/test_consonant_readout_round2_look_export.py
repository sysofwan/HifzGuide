"""The look export names each flagged word's consonant substitutions (#225).

The catch-loss audit marks a word as a harder pair (س/ص/ث) from what the flag says was heard in
place of what was expected, so ``look_grades.jsonl.gz`` carries ``consonants`` beside ``causes``:
one ``[cause, "expected→heard"]`` pair per consonant error behind the flag, ``[]`` for an
unflagged word and for a flag with no consonant error.
"""

from __future__ import annotations

import gzip
import json

from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import round2_fixtures as fx


def test_the_look_export_names_each_flagged_words_consonant_substitutions(tmp_path):
    repo = fx.init_repo(tmp_path / "repo")
    spec = fx.filled_spec(root=repo / "readout", prereg=fx.commit_prereg(repo))
    layout = ag.Layout(spec.root)
    layout.root.mkdir()
    layout.models.write_text(json.dumps(fx.look_manifest(spec)), encoding="utf-8")
    items = [fx.tlog_item(n, 1 + n % 5) for n in range(4)]
    flags = {("c95", 0, 1): fx.consonant_flag("س→ص"), ("c95", 1, 1): fx.TASHKEEL_FLAG}
    sources = fx.TlogSources(tmp_path / "files", items,
                             lambda model, item, position: flags.get((model, item, position)),
                             spec=spec)

    ar.run_look(sources, layout, spec=spec)

    with gzip.open(layout.root / ar.LOOK_GRADES, "rt", encoding="utf-8") as stream:
        rows = {(row["model"], row["key"], row["position"]): row
                for row in map(json.loads, stream)}
    assert rows["c95", items[0].key, 1]["consonants"] == [["other_consonant", "س→ص"]]
    assert rows["c95", items[1].key, 1]["consonants"] == []  # a tashkeel flag
    assert rows["c95", items[0].key, 0]["consonants"] == []  # not flagged
