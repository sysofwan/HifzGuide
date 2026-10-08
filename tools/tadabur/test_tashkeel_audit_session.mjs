// Tests for the listening page's logic (tashkeel_audit_session.mjs), with a fake server,
// player and view. Run by test_tashkeel_audit_page.py through `node --test`.
import assert from "node:assert/strict";
import { test } from "node:test";

import { ANSWER_GUARD_MS, createSession } from "./tashkeel_audit_session.mjs";

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness(heard = [null, null, null]) {
  let now = 0;
  const saves = [];
  const plays = [];
  const errors = [];
  const sites = heard.map((h, i) => ({ key: `k${i}`, heard: h, note: "", choices: ["a", "b"] }));
  const session = createSession({
    api: {
      sites: async () => sites.map((site) => ({ ...site })),
      save: (body) => { const d = deferred(); saves.push({ body, ...d }); return d.promise; },
    },
    player: { play: (key) => plays.push(key), stop: () => plays.push(null) },
    view: { render() {}, saving() {}, error: (message) => errors.push(message) },
    clock: () => now,
  });
  const tick = (ms = ANSWER_GUARD_MS) => { now += ms; };
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const consistent = () => {
    const { playing, key } = session.state();
    assert.ok(playing === null || playing === key, `playing ${playing} while ${key} is shown`);
  };
  return { session, saves, plays, errors, tick, settle, consistent };
}

test("a double tap answers only the site on screen", async () => {
  const h = harness();
  await h.session.load();
  h.tick();
  const first = h.session.answer("a", "");
  assert.equal(await h.session.answer("b", ""), false);  // the second tap, same instant
  assert.equal(h.saves.length, 1);
  h.saves[0].resolve();
  assert.equal(await first, true);
  assert.equal(h.session.state().index, 1);
  assert.equal(await h.session.answer("b", ""), false);  // site 1 just appeared: guarded
  assert.equal(h.saves.length, 1);
  h.consistent();
});

test("a second answer waits for the first save, so completions cannot reorder", async () => {
  const h = harness();
  await h.session.load();
  h.tick();
  const first = h.session.answer("a", "");
  h.tick();
  assert.equal(await h.session.answer("b", ""), false);  // in flight: refused, not queued
  assert.equal(h.session.move(1), false);                 // and the site cannot change
  h.saves[0].resolve();
  await first;
  h.tick();
  const second = h.session.answer("b", "");
  assert.deepEqual(h.saves.map((s) => s.body.key), ["k0", "k1"]);
  h.saves[1].resolve();
  assert.equal(await second, true);
  assert.equal(h.session.state().answered, 2);
});

test("a failed save changes nothing but the error, and the site can be answered again", async () => {
  const h = harness();
  await h.session.load();
  h.tick();
  const attempt = h.session.answer("a", "note");
  h.saves[0].reject(new Error("disk full"));
  assert.equal(await attempt, false);
  assert.deepEqual(h.session.state(), {
    index: 0, key: "k0", playing: null, saving: false, answered: 0, total: 3,
  });
  assert.match(h.errors.at(-1), /disk full/);
  const retry = h.session.answer("a", "note");
  h.saves[1].resolve();
  assert.equal(await retry, true);
  assert.equal(h.session.state().answered, 1);
});

test("each save names the answer it replaces", async () => {
  const h = harness(["a", null]);
  await h.session.load();               // resumes at the first unanswered site
  assert.equal(h.session.state().key, "k1");
  h.tick();
  assert.equal(h.session.move(-1), true);
  h.tick();
  const change = h.session.answer("b", "");
  assert.deepEqual(h.saves[0].body, { key: "k0", heard: "b", note: "", previous: "a" });
  h.saves[0].resolve();
  await change;
});

test("resuming starts at the first unanswered site and counts what is saved", async () => {
  const h = harness(["a", "b", null]);
  await h.session.load();
  assert.equal(h.session.state().key, "k2");
  assert.equal(h.session.state().answered, 2);
});

test("every change of site stops playback, and what plays is the site on screen", async () => {
  const h = harness();
  await h.session.load();
  h.session.play();
  h.consistent();
  h.tick();
  h.session.move(1);
  h.consistent();
  assert.deepEqual(h.plays.slice(-2), [null, "k1"]);  // stopped, then the new site
  h.tick();
  h.session.jump();
  h.consistent();
  h.tick();
  const answering = h.session.answer("a", "");
  h.saves[0].resolve();
  await answering;
  h.consistent();
  assert.equal(h.session.state().playing, h.session.state().key);
});
