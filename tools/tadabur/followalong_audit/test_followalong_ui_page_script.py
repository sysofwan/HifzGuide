"""The page's own selection logic runs (page script #311 PR3, reviewer r7 R7.1).

The locked server tests POST answers over HTTP and never execute the page; the page's script is
what actually records a verdict, and it is where a selection could be discarded. This runs that
script — the ``<script type="module">`` block exactly as served — through the shared runner
(:mod:`tadabur.followalong_audit.page_node`) with a minimal DOM and a ``fetch`` stub, so the real
``choose``/``save``/``show`` path is exercised: an unanswered item is chosen, a mistake's kinds are
ticked and the first answer is posted; a stored answer initializes the draft, a new selection is
not reverted by it, a note being written survives a selection, and changing an answer posts the
answer it replaces.
"""

from __future__ import annotations

from pathlib import Path

from tadabur.followalong_audit.page_node import run
from tadabur.followalong_audit.s5_score import MISTAKE_KINDS, SUB_REASONS, VERDICTS

CHECKS = 15

SCENARIO = r"""
await tick();  // the page's own load() settles against the stubs
check("the sites are served in order", () =>
  assert.deepEqual(page.state.sites.map((site) => site.key), ["k0", "k1"]));
check("an unanswered item starts with no choice", () => assert.equal(page.state.choice, null));

await page.save();
check("save without a choice posts nothing", () => {
  assert.equal(requests.length, 0);
  assert.equal(el("status").textContent, "Choose a verdict first.");
});

page.choose("mistake");
check("a chosen verdict stays chosen", () => assert.equal(page.state.choice, "mistake"));
check("mistake shows the kinds", () =>
  assert.equal(el("kinds").classList.contains("hidden"), false));
check("the chosen button is the only one marked", () => assert.deepEqual(
  el("choices").children.filter((button) => button.classList.contains("chosen"))
    .map((button) => button.dataset.verdict), ["mistake"]));

await page.save();
check("a mistake needs a kind", () => assert.equal(requests.length, 0));

const kind = el("kinds").children[0].children[0];
kind.checked = true;
fire(kind, "change");
el("note").value = "first note";
await page.save();
check("the first verdict is posted", () => {
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], { key: "k0", verdict: "mistake",
    kinds: [payload.vocabulary.kinds[0]], sub_reason: null, note: "first note", previous: null });
});
check("the save counts as answered", () =>
  assert.equal(el("progress").textContent, "1 of 2 answered"));

page.show(1);
check("a stored answer initializes the draft", () =>
  assert.equal(page.state.choice, "correct"));
el("note").value = "my own note";
page.choose("different text");
check("a new selection is not reverted by the stored answer", () =>
  assert.equal(page.state.choice, "different text"));
check("a new selection keeps the note being written", () =>
  assert.equal(el("note").value, "my own note"));
check("kinds hide when the verdict leaves mistake", () =>
  assert.equal(el("kinds").classList.contains("hidden"), true));

await page.save();
check("changing an answer posts the answer it replaces", () => {
  assert.equal(requests.length, 2);
  assert.equal(requests[1].verdict, "different text");
  assert.equal(requests[1].note, "my own note");
  assert.deepEqual(requests[1].previous, { verdict: "correct", kinds: [],
                                           sub_reason: null, note: "stored" });
});

fire(document, "keydown",
     { key: "1", target: { matches: () => false }, preventDefault() {} });
check("the numeric shortcut selects a verdict", () =>
  assert.equal(page.state.choice, payload.vocabulary.verdicts[0]));
"""


def _payload() -> dict:
    sites = [
        {"key": "k0", "surah_ayah": "2:3", "before": "الذين ", "carrier": "يؤمنون",
         "after": " بالغيب", "answer": None},
        {"key": "k1", "surah_ayah": "2:4", "before": "و", "carrier": "الذين",
         "after": " يؤمنون", "answer": {"verdict": "correct", "kinds": [], "sub_reason": None,
                                       "note": "stored"}},
    ]
    return {"sites": sites, "progress": {"answered": 1, "total": 2},
            "vocabulary": {"verdicts": list(VERDICTS), "kinds": list(MISTAKE_KINDS),
                           "sub_reasons": list(SUB_REASONS)}}


def test_the_page_script_selects_saves_and_changes_verdicts(tmp_path):
    result = run(SCENARIO, _payload(), Path(tmp_path))
    assert result == {"checks": CHECKS, "requests": 2}, result
