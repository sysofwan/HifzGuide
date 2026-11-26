"""The Node runner both S5 page tests share (#311 PR4, selfcheck r4 S4.3).

The S5 page's logic is the served ``<script type="module">``, so it is exercised by running exactly
that source under Node with a small DOM and a ``fetch`` stub. Two tests drive it — the PR3 draft
rules and the PR4 click flow — and the emulation lives here once: a page change (a new element, a
new selector, a new reply field) is patched in one place, so a half-patch cannot leave one test
asserting against a DOM the page no longer uses.

A test supplies only its data (the sites and vocabulary the stub serves) and its scenario: the
JavaScript run after the page's own ``load()`` settles, with these in scope —

``page`` (``show``/``choose``/``save``/``state``), ``el(id)``, ``payload``, ``requests`` (the
answer bodies the page has sent), ``click(verdict)``, ``enter()``, ``fire(target, type, event)``,
``tick()``/``settle()``, ``elapse(ms)`` (moves the clock a step's click guard reads),
``played()`` (how often the player was told to play), ``check(name, assertion)``, ``assert``,
``setRefuse(reply)`` (the next answer is refused: ``{status, error}`` or ``{thrown: message}``),
``holdNext()``/``release(refusal?)`` (hold one reply in flight, then resolve it — refused when a
``{status, error}`` is given), ``blockAutoplay(on)`` and
``failSites(on)``.

The runner prints ``{"checks": n, "requests": n}``; the caller asserts the shape it expects.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "s5_ui_page.html"


def script() -> str:
    """The page's module block, as the server sends it."""
    html = PAGE.read_text(encoding="utf-8")
    return html.split('<script type="module">', 1)[1].split("</script>", 1)[0]


def node() -> str:
    """The Node binary; a test dependency, never a reason to skip a page-script regression."""
    binary = shutil.which("node")
    assert binary is not None, (
        "node is required to run the page's own script (tools/README.md, Tests): install it "
        "(fnm/nvm on the Mac) rather than running the suite without the page regression")
    return binary


def run(scenario: str, payload: dict, tmp_path: Path) -> dict:
    """Run the page with ``scenario`` after its load; return the runner's counts."""
    (tmp_path / "page.js").write_text(script(), encoding="utf-8")
    (tmp_path / "payload.json").write_text(json.dumps(payload, ensure_ascii=False),
                                           encoding="utf-8")
    (tmp_path / "scenario.js").write_text(scenario, encoding="utf-8")
    (tmp_path / "harness.cjs").write_text(HARNESS, encoding="utf-8")
    done = subprocess.run([node(), str(tmp_path / "harness.cjs"), str(tmp_path / "page.js"),
                           str(tmp_path / "payload.json"), str(tmp_path / "scenario.js")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


HARNESS = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const source = fs.readFileSync(process.argv[2], "utf8");
const payload = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const scenario = fs.readFileSync(process.argv[4], "utf8");

const requests = [];
let refuse = null;        // the next answer is refused, once
let holdReply = false;    // the next answer's reply waits for release()
let heldReply = null;
let blockedPlay = false;
let sitesFail = false;
let nowOffset = 0;        // elapse(): the clock a step's click guard reads
const realNow = Date.now;
Date.now = () => realNow() + nowOffset;

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
  play() {
    this.played += 1;
    return blockedPlay ? Promise.reject(new Error("autoplay refused")) : undefined;
  }
  blur() { if (document.activeElement === this) document.activeElement = null; }
  matches() { return false; }
}

const ids = new Map();
const document = {
  activeElement: null,
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

const accept = (body) => {
  const site = payload.sites.find((candidate) => candidate.key === body.key);
  site.answer = { verdict: body.verdict, kinds: body.kinds, sub_reason: body.sub_reason,
                  note: body.note };
  return { answer: site.answer,
           progress: { answered: requests.length, total: payload.sites.length } };
};

globalThis.fetch = async (url, options = {}) => {
  if (url === "/api/sites") {
    if (sitesFail) throw new Error("the box is unreachable");
    // A fresh copy per read, as HTTP gives: the page's snapshot must not alias the server's.
    return { ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(payload)) };
  }
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
  if (site.answer !== null && JSON.stringify(site.answer) !== JSON.stringify(body.previous))
    return { ok: false, status: 409, json: async () => ({ error: "another tab" }) };
  if (holdReply) {
    holdReply = false;
    return await new Promise((resolve) => {
      heldReply = (refusal) => resolve(refusal
        ? { ok: false, status: refusal.status, json: async () => ({ error: refusal.error }) }
        : { ok: true, status: 200, json: async () => accept(body) });
    });
  }
  return { ok: true, status: 200, json: async () => accept(body) };
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
const elapse = (ms) => { nowOffset += ms; };
const checks = [];
const check = (name, assertion) => { assertion(); checks.push(name); };
const click = (verdict) =>
  fire(el("choices").children.find((button) => button.dataset.verdict === verdict), "click");
const enter = () =>
  fire(document, "keydown", { key: "Enter", target: { matches: () => false }, preventDefault() {} });

const helpers = {
  page, el, fire, tick, settle, elapse, check, requests, payload, assert, click, enter,
  played: () => el("audio").played,
  setRefuse: (value) => { refuse = value; },
  holdNext: () => { holdReply = true; },
  release: (refusal) => { const reply = heldReply; heldReply = null; if (reply) reply(refusal); },
  blockAutoplay: (on = true) => { blockedPlay = on; },
  failSites: (on = true) => { sitesFail = on; },
};

(async () => {
  await new Function("env", `return (async () => {
    const { page, el, fire, tick, settle, elapse, check, requests, payload, assert, click, enter,
            played, setRefuse, holdNext, release, blockAutoplay, failSites } = env;
${scenario}
  })()`)(helpers);
  console.log(JSON.stringify({ checks: checks.length, requests: requests.length }));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""
