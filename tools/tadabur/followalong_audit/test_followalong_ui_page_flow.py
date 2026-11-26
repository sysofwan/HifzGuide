"""The page's click flow runs (page ergonomics #311 PR4, owner request).

The owner asked the page to "auto save and go next and auto play when i make a verdict", and chose
the shape: a click on correct / not sure / different text is the whole answer — one POST, the next
item, its clip playing at once; a mistake and an unusable clip open their detail (kinds, the
sub-reason) and wait, then Enter or Save does the same; the last item stays shown instead of
wrapping; a refused save never advances. This drives the served ``<script type="module">`` through
the shared runner (:mod:`tadabur.followalong_audit.page_node`) with clicks, kinds, Enter and the
player, including the two ways a click can lie: the second half of a double click on the item that
just appeared, and a click that lands while an answer is in flight.

The sibling ``test_followalong_ui_page_script.py`` pins the PR3 draft rules through ``choose`` and
``save``; this file owns the click flow, so a regression in one cannot hide behind the other.
"""

from __future__ import annotations

from pathlib import Path

from tadabur.followalong_audit.page_node import run
from tadabur.followalong_audit.s5_score import MISTAKE_KINDS, SUB_REASONS, VERDICTS

CHECKS = 26

SCENARIO = r"""
await settle();   // the page's own load() settles against the stubs
check("the first item is shown and plays at once", () => {
  assert.equal(page.state.index, 0);
  assert.equal(played(), 1);
});

elapse(500);   // the step's guard is over: the owner has had the clip
el("note").value = "typed before choosing";
click("correct");
check("a click shows its verdict at once", () => assert.equal(page.state.choice, "correct"));
await settle();
check("a click posts exactly one answer, note included", () => {
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], { key: "k0", verdict: "correct", kinds: [],
    sub_reason: null, note: "typed before choosing", previous: null });
});
check("the step lands on the next item", () => assert.equal(page.state.index, 1));
check("the next item's clip plays at once", () => assert.equal(played(), 2));
check("the step clears the draft it left", () => {
  assert.equal(el("note").value, "");
  assert.equal(page.state.choice, null);
});
check("the answered count follows the file", () =>
  assert.equal(el("progress").textContent, "1 of 4 answered"));

click("correct");
await settle();
check("the second half of a double click answers nothing", () => {
  assert.equal(requests.length, 1);
  assert.equal(page.state.index, 1);
});
click("correct");
await settle();
check("a click inside the step's guard answers nothing", () => {
  assert.equal(requests.length, 1);
  assert.equal(page.state.choice, null);
});

elapse(500);
holdNext();
click("not sure");
await tick();
await tick();
click("correct");
check("a click while a save is in flight keeps the draft it does not send", () => {
  assert.equal(page.state.choice, "not sure");
  assert.equal(el("status").textContent, "saving…");
});
fire(el("prev"), "click");   // and the owner steps back before the reply lands
release();
await settle();
check("a reply never moves a page the owner has navigated", () => {
  assert.equal(requests.length, 2);
  assert.equal(page.state.index, 0);   // the owner stepped back while the answer was in flight
  assert.equal(played(), 3);
});
check("the answered count still follows the file", () =>
  assert.equal(el("progress").textContent, "2 of 4 answered"));
fire(el("next"), "click");
check("Next goes to the first unanswered item", () => {
  assert.equal(page.state.index, 2);
  assert.equal(played(), 4);
});

elapse(500);
setRefuse({ status: 500, error: "disk full" });
click("not sure");
await settle();
check("a refused save stays on its item and says why", () => {
  assert.equal(page.state.index, 2);
  assert.equal(requests.length, 2);
  assert.equal(el("status").textContent, "disk full");
  assert.equal(played(), 4);
});
setRefuse({ thrown: "network down" });
click("different text");
await settle();
check("an unreachable box stays on its item and says why", () => {
  assert.equal(page.state.index, 2);
  assert.equal(requests.length, 2);
  assert.equal(el("status").textContent, "not saved (network down)");
});
setRefuse({ status: 409, error: "another tab" });
failSites(true);
click("correct");
await settle();
check("a 409 whose reload also fails still says why", () => {
  assert.equal(page.state.index, 2);
  assert.equal(requests.length, 2);
  assert.equal(el("status").textContent, "another tab");
});
failSites(false);

click("mistake");
await settle();
check("a mistake posts nothing on the click", () => {
  assert.equal(requests.length, 2);
  assert.equal(page.state.index, 2);
  assert.equal(el("kinds").classList.contains("hidden"), false);
});
enter();
await settle();
check("Enter without a kind saves nothing and asks for one", () => {
  assert.equal(requests.length, 2);
  assert.equal(el("status").textContent, "Tick what was wrong.");
});

const kind = el("kinds").children[0].children[0];
kind.checked = true;
fire(kind, "change");
enter();
await settle();
check("Enter with a kind posts it and steps on", () => {
  assert.equal(requests.length, 3);
  assert.deepEqual(requests[2], { key: "k2", verdict: "mistake",
    kinds: [payload.vocabulary.kinds[0]], sub_reason: null, note: "", previous: null });
  assert.equal(page.state.index, 3);
});
check("the step plays the new item's clip", () => assert.equal(played(), 5));

elapse(500);
click("unusable audio");
await settle();
check("an unusable clip waits like a mistake", () => {
  assert.equal(requests.length, 3);
  assert.equal(page.state.index, 3);
  assert.equal(el("sub").classList.contains("hidden"), false);
});
el("sub-reason").value = payload.vocabulary.sub_reasons[0];
enter();
await settle();
check("Enter posts the unusable clip with its reason, on the last item", () => {
  assert.equal(requests.length, 4);
  assert.deepEqual(requests[3], { key: "k3", verdict: "unusable audio", kinds: [],
    sub_reason: payload.vocabulary.sub_reasons[0], note: "", previous: null });
  assert.equal(page.state.index, 3);
  assert.equal(el("status").textContent, "saved — done: this is the last item");
});

click("correct");
await settle();
check("a re-answer of the last item posts the answer it replaces", () => {
  assert.equal(requests.length, 5);
  assert.deepEqual(requests[4].previous, { verdict: "unusable audio", kinds: [],
    sub_reason: payload.vocabulary.sub_reasons[0], note: "" });
  assert.equal(page.state.index, 3);
});

fire(el("prev"), "click");
check("Back shows the earlier item and plays it", () => {
  assert.equal(page.state.index, 2);
  assert.equal(played(), 6);
});
blockAutoplay(true);
fire(el("prev"), "click");
await settle();
check("a refused autoplay says tap to play", () => {
  assert.equal(page.state.index, 1);
  assert.equal(played(), 7);
  assert.equal(el("hint").classList.contains("hidden"), false);
});
blockAutoplay(false);
fire(el("prev"), "click");
await settle();
check("the next play clears the hint", () => {
  assert.equal(page.state.index, 0);
  assert.equal(played(), 8);
  assert.equal(el("hint").classList.contains("hidden"), true);
});
"""


def _payload() -> dict:
    sites = [
        {"key": "k0", "surah_ayah": "2:3", "before": "الذين ", "carrier": "يؤمنون",
         "after": " بالغيب", "answer": None},
        {"key": "k1", "surah_ayah": "2:4", "before": "و", "carrier": "الذين",
         "after": " يؤمنون", "answer": None},
        {"key": "k2", "surah_ayah": "2:5", "before": "", "carrier": "أُو۟لَـٰٓئِكَ",
         "after": " عَلَىٰ هُدًى", "answer": None},
        {"key": "k3", "surah_ayah": "2:6", "before": "إِنَّ ", "carrier": "ٱلَّذِينَ",
         "after": " كَفَرُوا۟", "answer": None},
    ]
    return {"sites": sites, "progress": {"answered": 0, "total": len(sites)},
            "vocabulary": {"verdicts": list(VERDICTS), "kinds": list(MISTAKE_KINDS),
                           "sub_reasons": list(SUB_REASONS)}}


def test_a_verdict_click_saves_steps_on_and_plays_the_next_item(tmp_path):
    result = run(SCENARIO, _payload(), Path(tmp_path))
    assert result == {"checks": CHECKS, "requests": 5}, result
