"""The page's click flow runs (page ergonomics #311 PR4, owner request).

The owner asked the page to "auto save and go next and auto play when i make a verdict", and chose
the shape: a click on correct / not sure / different text / unusable audio is the whole answer —
one POST, the next item, its clip playing at once; a mistake waits for at least one kind, then
Enter or Save does the same; the last item stays shown instead of wrapping; a refused save never
advances. This runs the served ``<script type="module">`` through Node's ``vm`` with a DOM and a
``fetch`` stub, and drives it the way the owner does: verdict buttons, the kind checkboxes, Enter
and the player.

The sibling ``test_followalong_ui_page_script.py`` pins the PR3 draft rules through ``choose`` and
``save``; this file owns the click flow, so a regression there cannot hide behind a passing draft.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from tadabur.followalong_audit.s5_score import MISTAKE_KINDS, SUB_REASONS, VERDICTS

PAGE = Path(__file__).resolve().parent / "s5_ui_page.html"
CHECKS = 16


def _script() -> str:
    """The page's module block, as the server sends it."""
    html = PAGE.read_text(encoding="utf-8")
    return html.split('<script type="module">', 1)[1].split("</script>", 1)[0]


HARNESS = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const source = fs.readFileSync(process.argv[2], "utf8");
const payload = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const requests = [];
let refuse = null;

class El {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.listeners = {};
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.type = "";
    this.className = "";
    this.src = "";
    this.textContent = "";
    this.played = 0;
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, on) => {
        const chosen = on === undefined ? !classes.has(name) : Boolean(on);
        if (chosen) classes.add(name); else classes.delete(name);
        return chosen;
      },
    };
  }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  play() { this.played += 1; }
  matches() { return false; }
}

const ids = new Map();
const document = {
  getElementById(id) {
    if (!ids.has(id)) ids.set(id, new El("div"));
    return ids.get(id);
  },
  createElement(tag) { return new El(tag); },
  querySelectorAll(selector) {
    if (selector === "#choices button") return document.getElementById("choices").children;
    if (selector === "#kinds input")
      return document.getElementById("kinds").children
        .flatMap((label) => label.children)
        .filter((child) => child instanceof El && child.tagName === "input");
    throw new Error(`unexpected selector ${selector}`);
  },
  listeners: {},
  addEventListener(type, handler) { (document.listeners[type] ||= []).push(handler); },
};
globalThis.document = document;

const answered = () => payload.sites.filter((site) => site.answer !== null).length;
globalThis.fetch = async (url, options = {}) => {
  if (url === "/api/sites") return { ok: true, status: 200, json: async () => payload };
  if (url !== "/api/verdict") throw new Error(`unexpected fetch ${url}`);
  const body = JSON.parse(options.body);
  if (refuse !== null) {
    const denied = refuse;
    refuse = null;
    if (denied.thrown) throw new Error(denied.thrown);
    return { ok: false, status: denied.status, json: async () => ({ error: denied.error }) };
  }
  requests.push(body);
  const site = payload.sites.find((candidate) => candidate.key === body.key);
  site.answer = { verdict: body.verdict, kinds: body.kinds, sub_reason: body.sub_reason,
                  note: body.note };
  return { ok: true, status: 200, json: async () => ({
    answer: site.answer, progress: { answered: answered(), total: payload.sites.length } }) };
};

vm.runInThisContext(`${source}\nglobalThis.page = { show, choose, save, state };`,
                    { filename: "s5_ui_page.js" });

const page = globalThis.page;
const el = (id) => document.getElementById(id);
const fire = (target, type, event = {}) => {
  for (const handler of target.listeners[type] || []) handler(event);
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const settle = async () => { for (let step = 0; step < 5; step += 1) await tick(); };
const button = (verdict) =>
  el("choices").children.find((candidate) => candidate.dataset.verdict === verdict);
const click = (verdict) => fire(button(verdict), "click");
const enter = () =>
  fire(document, "keydown", { key: "Enter", target: { matches: () => false }, preventDefault() {} });
const played = () => el("audio").played;
const checks = [];
const check = (name, assertion) => { assertion(); checks.push(name); };

(async () => {
  await settle();   // the page's own load() settles against the stubs
  check("the first item is shown and plays at once", () => {
    assert.equal(page.state.index, 0);
    assert.equal(played(), 1);
  });

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
    assert.equal(el("progress").textContent, "1 of 3 answered"));

  el("sub-reason").value = payload.vocabulary.sub_reasons[0];
  refuse = { status: 500, error: "disk full" };
  click("unusable audio");
  await settle();
  check("a refused save stays on its item and says why", () => {
    assert.equal(page.state.index, 1);
    assert.equal(requests.length, 1);
    assert.equal(el("status").textContent, "disk full");
    assert.equal(played(), 2);
  });

  refuse = { thrown: "network down" };
  click("correct");
  await settle();
  check("an unreachable box stays on its item and says why", () => {
    assert.equal(page.state.index, 1);
    assert.equal(requests.length, 1);
    assert.equal(el("status").textContent, "not saved (network down)");
    assert.equal(played(), 2);
  });

  click("mistake");
  await settle();
  check("a mistake posts nothing on the click", () => {
    assert.equal(requests.length, 1);
    assert.equal(el("kinds").classList.contains("hidden"), false);
  });
  enter();
  await settle();
  check("Enter without a kind saves nothing and asks for one", () => {
    assert.equal(requests.length, 1);
    assert.equal(el("status").textContent, "Tick what was wrong.");
  });

  const kind = el("kinds").children[0].children[0];
  kind.checked = true;
  fire(kind, "change");
  enter();
  await settle();
  check("Enter with a kind posts it and steps on", () => {
    assert.equal(requests.length, 2);
    assert.deepEqual(requests[1], { key: "k1", verdict: "mistake",
      kinds: [payload.vocabulary.kinds[0]], sub_reason: null, note: "", previous: null });
    assert.equal(page.state.index, 2);
  });
  check("the step plays the new item's clip", () => assert.equal(played(), 3));

  click("different text");
  await settle();
  check("the last item saves and does not wrap", () => {
    assert.equal(requests.length, 3);
    assert.equal(page.state.index, 2);
    assert.equal(el("status").textContent, "saved — done: this is the last item");
  });
  click("correct");
  await settle();
  check("a re-answer of the last item posts the answer it replaces", () => {
    assert.equal(requests.length, 4);
    assert.equal(requests[3].verdict, "correct");
    assert.deepEqual(requests[3].previous, { verdict: "different text", kinds: [],
      sub_reason: null, note: "" });
    assert.equal(page.state.index, 2);
  });
  check("Back shows the earlier item and plays it", () => {
    fire(el("prev"), "click");
    assert.equal(page.state.index, 1);
    assert.equal(played(), 4);
  });

  console.log(JSON.stringify({ checks: checks.length, requests: requests.length }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
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
    ]
    return {"sites": sites, "progress": {"answered": 0, "total": len(sites)},
            "vocabulary": {"verdicts": list(VERDICTS), "kinds": list(MISTAKE_KINDS),
                           "sub_reasons": list(SUB_REASONS)}}


def _node() -> str:
    """The Node binary; a test dependency, never a reason to skip the page's own script."""
    node = shutil.which("node")
    assert node is not None, (
        "node is required to run the page's own script (tools/README.md, Tests): install it "
        "(fnm/nvm on the Mac) rather than running this suite without the page regression")
    return node


def test_a_verdict_click_saves_steps_on_and_plays_the_next_item(tmp_path):
    node = _node()
    (tmp_path / "page.js").write_text(_script(), encoding="utf-8")
    (tmp_path / "payload.json").write_text(json.dumps(_payload(), ensure_ascii=False),
                                           encoding="utf-8")
    (tmp_path / "harness.cjs").write_text(HARNESS, encoding="utf-8")
    done = subprocess.run([node, str(tmp_path / "harness.cjs"), str(tmp_path / "page.js"),
                           str(tmp_path / "payload.json")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {"checks": CHECKS, "requests": 4}, done.stdout
