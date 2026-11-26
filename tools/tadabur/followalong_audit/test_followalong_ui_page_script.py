"""The page's own selection logic runs (page script #311 PR3, reviewer r7 R7.1).

The locked server tests POST answers over HTTP and never execute the page; the page's script is
what actually records a verdict, and it is where a selection could be discarded. This runs that
script — the ``<script type="module">`` block exactly as served — through Node's ``vm`` with a
minimal DOM and a ``fetch`` stub, so the real ``choose``/``save``/``show`` path is exercised:
an unanswered item is chosen, a mistake's kinds are ticked and the first answer is posted; a
stored answer initializes the draft, a new selection is not reverted by it, a note being written
survives a selection, and changing an answer posts the answer it replaces.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from tadabur.followalong_audit.s5_score import MISTAKE_KINDS, SUB_REASONS, VERDICTS

PAGE = Path(__file__).resolve().parent / "s5_ui_page.html"
CHECKS = 15


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
  play() { this.played = (this.played || 0) + 1; }
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
globalThis.fetch = async (url, options = {}) => {
  if (url === "/api/sites") return { ok: true, json: async () => payload };
  if (url !== "/api/verdict") throw new Error(`unexpected fetch ${url}`);
  const body = JSON.parse(options.body);
  requests.push(body);
  const site = payload.sites.find((candidate) => candidate.key === body.key);
  if (site.answer !== null && JSON.stringify(site.answer) !== JSON.stringify(body.previous))
    return { ok: false, status: 409, json: async () => ({ error: "another tab" }) };
  site.answer = { verdict: body.verdict, kinds: body.kinds, sub_reason: body.sub_reason,
                  note: body.note };
  return { ok: true, json: async () => ({ answer: site.answer,
    progress: { answered: requests.length, total: payload.sites.length } }) };
};

vm.runInThisContext(`${source}\nglobalThis.page = { show, choose, save, state };`,
                    { filename: "s5_ui_page.js" });

const page = globalThis.page;
const el = (id) => document.getElementById(id);
const fire = (target, type, event = {}) => {
  for (const handler of target.listeners[type] || []) handler(event);
};
const checks = [];
const check = (name, assertion) => { assertion(); checks.push(name); };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

(async () => {
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
         "after": " يؤمنون", "answer": {"verdict": "correct", "kinds": [], "sub_reason": None,
                                       "note": "stored"}},
    ]
    return {"sites": sites, "progress": {"answered": 1, "total": 2},
            "vocabulary": {"verdicts": list(VERDICTS), "kinds": list(MISTAKE_KINDS),
                           "sub_reasons": list(SUB_REASONS)}}


def _node() -> str:
    """The Node binary; a test dependency, never a reason to skip the only page-script check."""
    node = shutil.which("node")
    assert node is not None, (
        "node is required to run the page's own script (tools/README.md, Tests): install it "
        "(fnm/nvm on the Mac) rather than running this suite without the page regression")
    return node


def test_the_page_script_selects_saves_and_changes_verdicts(tmp_path):
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
    assert result == {"checks": CHECKS, "requests": 2}, done.stdout
