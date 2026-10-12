// Tests for the listening page's logic (tashkeel_audit_session.mjs), with a fake server,
// player and view. Run by test_tashkeel_audit_page.py through `node --test`.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ANSWER_GUARD_MS, createSession, highlightParts, trimNote,
} from "./tashkeel_audit_session.mjs";

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

// `natural`, when given, makes every site also ask whether it sounds natural, with that
// stored answer; `review` gives each site's review note (queued for another listen).
function harness(heard = [null, null, null], natural = null, review = []) {
  let now = 0;
  const saves = [];
  const plays = [];
  const errors = [];
  const drafts = [];
  const queuedViews = [];
  const reviews = [];
  const sites = heard.map((h, i) => ({
    key: `k${i}`, heard: h, note: "", choices: ["a", "b", "unclear"], review: review[i] ?? null,
    ...(natural && { natural: natural[i], natural_choices: ["natural", "unnatural"] }),
  }));
  const session = createSession({
    api: {
      sites: async () => sites.map((site) => ({ ...site })),
      save: (body) => { const d = deferred(); saves.push({ body, ...d }); return d.promise; },
      review: async (body) => {
        reviews.push(body);
        return { review: body.review ? "You marked this site to review." : null };
      },
    },
    player: { play: (key) => plays.push(key), stop: () => plays.push(null) },
    view: {
      render() {}, saving() {}, error: (message) => errors.push(message),
      drafted: (draft) => drafts.push({ ...draft }),
      queued: (note, counts) => queuedViews.push({ note, queued: counts.queued }),
    },
    clock: () => now,
  });
  const tick = (ms = ANSWER_GUARD_MS) => { now += ms; };
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const consistent = () => {
    const { playing, key } = session.state();
    assert.ok(playing === null || playing === key, `playing ${playing} while ${key} is shown`);
  };
  return {
    session, saves, plays, errors, drafts, queuedViews, reviews, tick, settle, consistent,
  };
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
    index: 0, key: "k0", playing: null, saving: false, answered: 0, total: 3, queued: 0,
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

test("a site that asks whether it sounds natural is saved once, with both answers", async () => {
  const h = harness([null, null], [null, null]);
  await h.session.load();
  h.tick();
  assert.equal(await h.session.answer("a", ""), false);  // a draft: shown, not saved
  assert.deepEqual(h.drafts, [{ heard: "a" }]);
  assert.equal(h.saves.length, 0);
  assert.equal(h.session.state().index, 0);
  const saving = h.session.judge("unnatural", "a click");
  assert.deepEqual(h.saves[0].body,
    { key: "k0", heard: "a", natural: "unnatural", note: "a click", previous: null });
  h.saves[0].resolve();
  assert.equal(await saving, true);
  assert.equal(h.session.state().answered, 1);
  assert.equal(h.session.state().index, 1);
});

test("either answer may come first, and a draft is dropped when the site changes", async () => {
  const h = harness([null, null], [null, null]);
  await h.session.load();
  h.tick();
  await h.session.judge("natural", "");
  h.session.move(1);
  h.tick();
  h.session.move(-1);
  h.tick();
  assert.equal(await h.session.answer("b", ""), false);  // the earlier judgement is gone
  const saving = h.session.judge("natural", "");
  assert.deepEqual(h.saves.map((s) => [s.body.heard, s.body.natural]), [["b", "natural"]]);
  h.saves[0].resolve();
  await saving;
});

test("changing one answer of a saved pair saves the pair and names the one it replaces", async () => {
  const h = harness(["a", null], ["natural", null]);
  await h.session.load();
  assert.equal(h.session.state().key, "k1");
  assert.equal(h.session.state().answered, 1);
  h.tick();
  h.session.move(-1);
  h.tick();
  const change = h.session.judge("unnatural", "");
  assert.deepEqual(h.saves[0].body, {
    key: "k0", heard: "a", natural: "unnatural", note: "",
    previous: { heard: "a", natural: "natural" },
  });
  h.saves[0].resolve();
  assert.equal(await change, true);
});

test("the naturalness question is guarded, and refused where it is not asked", async () => {
  const plain = harness();
  await plain.session.load();
  plain.tick();
  assert.equal(await plain.session.judge("natural", ""), false);
  assert.equal(plain.saves.length, 0);
  const h = harness([null], [null]);
  await h.session.load();
  assert.equal(await h.session.judge("natural", ""), false);  // the site just appeared
  assert.deepEqual(h.drafts, []);
});

test("a highlight inside a word is joined to both neighbours, so no ligature hides it", () => {
  // 43:35's لَمَّا: the م sits between لَ and اا, where the font would draw لم as one glyph.
  assert.deepEqual(highlightParts("ذَاالِكَ لَ", "م", "اا مَتَااعُ"),
    ["ذَاالِكَ لَ\u200D", "\u200Dم\u200D", "\u200Dاا مَتَااعُ"]);
});

test("no joiner is added where the letters do not join", () => {
  // A word edge, or the carrier's own mark after it.
  assert.deepEqual(highlightParts("قَالَ ", "ك", "ُم هُوَ"), ["قَالَ ", "ك", "ُم هُوَ"]);
  assert.deepEqual(highlightParts("", "ب", " ءِذَاا"), ["", "ب", " ءِذَاا"]);
  // 2:129's رَببَنَاا: ر cannot join forward, so the highlighted ب keeps its initial form.
  assert.deepEqual(highlightParts("رَ", "ب", "بَنَاا"), ["رَ", "ب\u200D", "\u200Dبَنَاا"]);
  // 6:64's يُنَججِۦۦك: the small yeh after the carrier joins nothing.
  assert.deepEqual(highlightParts("يُنَ", "جِ", "ۦۦك"), ["يُنَ\u200D", "\u200Dجِ", "ۦۦك"]);
  // A right-joining carrier joins back but never forward.
  assert.deepEqual(highlightParts("قَ", "ا", "لَ"), ["قَ\u200D", "\u200Dا", "لَ"]);
  // The hidden letter joins nothing.
  assert.deepEqual(highlightParts("لَ", "\u25CC", "اا"), ["لَ", "\u25CC", "اا"]);
});

test("a queued site keeps its answer until it is saved again, then the next queued plays", async () => {
  const h = harness(["a", null, "b", "a"], null, [null, null, "cut?", "cut?"]);
  await h.session.load();
  assert.equal(h.session.state().key, "k1");          // resume still means unanswered
  assert.equal(h.session.state().queued, 2);
  h.tick();
  assert.equal(h.session.nextQueued(), true);
  assert.equal(h.session.state().key, "k2");
  h.tick();
  const confirm = h.session.answer("b", "");          // the same answer takes it off the list
  assert.deepEqual(h.saves[0].body, { key: "k2", heard: "b", note: "", previous: "b" });
  h.saves[0].resolve();
  assert.equal(await confirm, true);
  assert.equal(h.session.state().queued, 1);
  assert.equal(h.session.state().key, "k3");          // the next queued site, not k1
  h.tick();
  const last = h.session.answer("a", "");
  h.saves[1].resolve();
  await last;
  assert.equal(h.session.state().queued, 0);
  assert.equal(h.session.state().key, "k1");          // then on to the unanswered ones
  assert.equal(h.session.nextQueued(), false);
  h.consistent();
});

test("Trimmed keeps its note through a naturalness answer given second, in either order", async () => {
  for (const second of ["judge", "trimmed"]) {
    const h = harness([null, null], [null, null]);
    await h.session.load();
    h.tick();
    if (second === "judge") assert.equal(await h.session.trimmed("no waqf"), false);  // a draft
    else assert.equal(await h.session.judge("unnatural", "no waqf"), false);
    if (second === "judge") h.session.judge("unnatural", "no waqf");
    else h.session.trimmed("no waqf");
    assert.deepEqual(h.saves[0].body, {
      key: "k0", heard: "unclear", natural: "unnatural", note: "trim; no waqf", previous: null,
    });
  }
});

test("choosing another heard answer after Trimmed drops the trim", async () => {
  const h = harness([null], [null]);
  await h.session.load();
  h.tick();
  assert.equal(await h.session.trimmed(""), false);
  assert.equal(await h.session.answer("a", ""), false);
  h.session.judge("natural", "");
  assert.deepEqual(h.saves[0].body, {
    key: "k0", heard: "a", natural: "natural", note: "", previous: null,
  });
});

test("Trimmed answers unclear and puts trim before the listener's note", async () => {
  const h = harness();
  await h.session.load();
  h.tick();
  const trimmed = h.session.trimmed("  waqf ");
  assert.deepEqual(h.saves[0].body, { key: "k0", heard: "unclear", note: "trim; waqf",
                                      previous: null });
  h.saves[0].resolve();
  assert.equal(await trimmed, true);
  assert.equal(trimNote(""), "trim;");
  assert.equal(trimNote("Trimmed at the start"), "trim; Trimmed at the start");
  assert.equal(trimNote("not trimmed"), "trim; not trimmed");
  assert.equal(trimNote("Trim; again"), "Trim; again");
});

test("queueing a site for review shows its note without redrawing the page", async () => {
  const h = harness([null, null]);
  await h.session.load();
  assert.equal(await h.session.markForReview(true), false);  // the site just appeared
  h.tick();
  assert.equal(await h.session.markForReview(true), true);
  assert.deepEqual(h.reviews, [{ key: "k0", review: true }]);
  assert.deepEqual(h.queuedViews.at(-1), { note: "You marked this site to review.", queued: 1 });
  assert.equal(await h.session.markForReview(false), true);
  assert.deepEqual(h.queuedViews.at(-1), { note: null, queued: 0 });
});

test("a review write blocks answers and moving until it is done", async () => {
  const pending = deferred();
  const shown = [];
  let now = 0;
  const session = createSession({
    api: {
      sites: async () => [{ key: "k0", heard: null, note: "", choices: ["a"], review: null },
                          { key: "k1", heard: null, note: "", choices: ["a"], review: null }],
      save: async () => {},
      review: () => pending.promise,
    },
    player: { play() {}, stop() {} },
    view: { render() {}, saving() {}, error() {}, drafted() {},
            queued: (note, counts) => shown.push({ note, queued: counts.queued }) },
    clock: () => now,
  });
  await session.load();
  now += ANSWER_GUARD_MS;
  const marking = session.markForReview(true);
  assert.equal(session.move(1), false);
  assert.equal(session.nextQueued(), false);
  assert.equal(await session.answer("a", ""), false);
  assert.equal(await session.markForReview(false), false);
  pending.resolve({ review: "You marked this site to review." });
  assert.equal(await marking, true);
  assert.equal(session.state().key, "k0");
  assert.deepEqual(shown, [{ note: "You marked this site to review.", queued: 1 }]);
});
