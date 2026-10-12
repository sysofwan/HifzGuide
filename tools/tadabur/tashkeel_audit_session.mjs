// The listening page's logic, with no DOM in it: which site is on screen, what plays, and
// how an answer is saved. tashkeel_audit_page.html wires it to the document;
// test_tashkeel_audit_session.mjs drives it with fakes (run by test_tashkeel_audit_page.py).
//
// Rules it keeps:
// * What plays is always the site on screen: every change of site stops playback first.
// * One answer is saved at a time. While a save is in flight, and for ANSWER_GUARD_MS after
//   a site appears, answers and navigation are ignored, so a double tap or a held key
//   cannot answer a site nobody has listened to.
// * An answer counts only once the server has written it. A failed save changes nothing
//   but the error shown, and the same site stays on screen to be answered again.
// * Each save names the answer it replaces (`previous`), so the server refuses one made
//   from a stale view (another tab or device answered the site since).
// * A site that also asks whether it sounds natural (it has a `natural` field) is saved
//   once both of its answers are chosen, as one verdict: the first choice is only a draft
//   (shown, not saved), and it is dropped when the site changes. Once saved, changing
//   either answer saves the pair again.
// * A site queued for another listen (its `review` is the note saying why) keeps its answer
//   until a new one is saved; saving it, changed or not, takes it off the list and moves to
//   the next queued site, or else the next unanswered one.
// * "Trimmed" answers unclear with a note starting TRIMMED_ANSWER: the clip's opening is
//   cut, so the site cannot be judged. The onset guard (owner_notes.py) reads that start as
//   trimmed whatever the rest says. On a site that also asks whether it sounds natural, the
//   intent is kept until the second answer saves the pair, and choosing another heard
//   answer drops it.
// * Queueing a site for review is one write like an answer: guarded the same way, and its
//   result is shown only if that site is still on screen.

export const ANSWER_GUARD_MS = 400;
export const TRIMMED_ANSWER = "trim;";

// The note a "Trimmed" answer saves: TRIMMED_ANSWER, then the listener's own note.
export function trimNote(note) {
  const rest = note.trim();
  if (rest.toLowerCase().startsWith(TRIMMED_ANSWER)) return rest;
  return rest === "" ? TRIMMED_ANSWER : `${TRIMMED_ANSWER} ${rest}`;
}

const ZWJ = "\u200D";
// Unicode joining types (ArabicShaping.txt) of the letters the references use: a dual-
// joining letter connects to both neighbours, a right-joining one only to the letter
// before it, and the tatweel joins both ways. Everything else (ء, the small waw and yeh,
// the hidden letter, a space) joins neither; combining marks are transparent.
const DUAL_JOINING = new Set("بتثجحخسشصضطظعغفقكلمنهيڇںـ");
const RIGHT_JOINING = new Set("اٲدذرزو");
const MARK = /\p{M}/u;
const lastLetter = (text) => [...text].reverse().find((char) => !MARK.test(char));

// Whether the letters on either side of a seam are drawn joined: the one before it can
// join forward and the one after it can join back. A seam before a combining mark is
// never split, since the mark belongs to the letter before it.
const joins = (left, right) => {
  const before = lastLetter(left);
  const after = right[0];
  return before !== undefined && after !== undefined && !MARK.test(after)
    && DUAL_JOINING.has(before) && (DUAL_JOINING.has(after) || RIGHT_JOINING.has(after));
};

// The reference split around the highlight, with a zero-width joiner on each side of the
// highlight where the letters there are drawn joined. Without it the font may draw the
// highlighted letter inside a ligature with its neighbour (the لم of لما), and a highlight
// inside a ligature has no width; the joiner breaks the ligature and keeps the joined
// forms. Where the letters do not join (a space, ر before the carrier, a small waw after
// it) no joiner is added, so no letter is forced into a joined form.
export function highlightParts(before, carrier, after) {
  const opens = joins(before, carrier);
  const closes = joins(carrier, after);
  return [
    opens ? before + ZWJ : before,
    (opens ? ZWJ : "") + carrier + (closes ? ZWJ : ""),
    closes ? ZWJ + after : after,
  ];
}

export function createSession({ api, player, view, clock = () => Date.now() }) {
  let sites = [];
  let index = 0;
  let saving = false;
  let shownAt = -Infinity;
  let playing = null;
  let draft = {};
  let trimDrafted = false;  // "Trimmed" was chosen and its pair is not saved yet

  const current = () => sites[index];
  const asksNatural = (site) => "natural" in site;
  const complete = (site) => site.heard !== null && (!asksNatural(site) || site.natural !== null);
  const answered = () => sites.filter(complete).length;
  const queued = () => sites.filter((site) => site.review).length;
  const counts = () => ({
    position: index + 1, total: sites.length, answered: answered(), queued: queued(),
  });

  // The answer the server holds for the site, as `previous` names it.
  function stored(site) {
    if (site.heard === null) return null;
    return asksNatural(site) ? { heard: site.heard, natural: site.natural } : site.heard;
  }

  function first(from, wanted) {
    for (let k = 0; k < sites.length; k++) {
      const i = (from + k) % sites.length;
      if (wanted(sites[i])) return i;
    }
    return -1;
  }
  const firstUnanswered = (from) => first(from, (site) => !complete(site));
  const firstQueued = (from) => first(from, (site) => site.review);

  function stop() {
    player.stop();
    playing = null;
  }

  function play(options = {}) {
    const site = current();
    if (!site) return;
    player.play(site.key, options);
    playing = site.key;
  }

  function show(i, { autoplay = false } = {}) {
    stop();
    draft = {};
    trimDrafted = false;
    index = i;
    shownAt = clock();
    view.render(current(), counts());
    if (autoplay) play();
  }

  const busy = () => saving || clock() - shownAt < ANSWER_GUARD_MS;

  // Save `choice` ({heard} or {natural}) with the rest of the site's answer, or keep it as
  // a draft while the other answer is still missing.
  async function submit(choice, note) {
    const site = current();
    if (!site || busy()) return false;
    const saved = asksNatural(site) ? { heard: site.heard, natural: site.natural }
      : { heard: site.heard };
    const answer = { ...saved, ...draft, ...choice };
    if (Object.values(answer).includes(null)) {
      draft = { ...draft, ...choice };
      view.drafted(draft);
      return false;
    }
    if (trimDrafted) note = trimNote(note);
    saving = true;
    view.saving(true);
    try {
      await api.save({ key: site.key, ...answer, note, previous: stored(site) });
    } catch (error) {
      view.error(`Not saved (${error.message}). Answer this site again.`);
      return false;
    } finally {
      saving = false;
      view.saving(false);
    }
    const wasQueued = Boolean(site.review);
    Object.assign(site, answer, { note, review: null });
    view.error("");
    const queuedNext = wasQueued ? firstQueued(index + 1) : -1;
    const next = queuedNext >= 0 ? queuedNext : firstUnanswered(index + 1);
    show(next >= 0 ? next : index, { autoplay: next >= 0 });
    return true;
  }

  return {
    async load() {
      sites = await api.sites();
      if (sites.length) show(Math.max(0, firstUnanswered(0)));
    },
    play,
    move(delta) {
      if (saving || !sites.length) return false;
      show(Math.min(sites.length - 1, Math.max(0, index + delta)), { autoplay: true });
      return true;
    },
    jump() {
      const next = firstUnanswered(index);
      if (saving || next < 0) return false;
      show(next, { autoplay: true });
      return true;
    },
    // The next site queued for another listen after the one on screen.
    nextQueued() {
      const next = firstQueued(index + 1);
      if (saving || next < 0) return false;
      show(next, { autoplay: true });
      return true;
    },
    // Queue the site on screen for another listen (`on`), or take it off the list.
    async markForReview(on) {
      const site = current();
      if (!site || busy()) return false;
      saving = true;
      view.saving(true);
      try {
        site.review = (await api.review({ key: site.key, review: on })).review;
      } catch (error) {
        view.error(`Not saved (${error.message}).`);
        return false;
      } finally {
        saving = false;
        view.saving(false);
      }
      view.error("");
      if (current() === site) view.queued(site.review, counts());
      return true;
    },
    // What was said at the carrier.
    answer(heard, note) {
      if (!busy()) trimDrafted = false;
      return submit({ heard }, note);
    },
    // The clip's opening is cut: unclear, with a note starting TRIMMED_ANSWER.
    trimmed(note) {
      if (busy()) return Promise.resolve(false);
      trimDrafted = true;
      return submit({ heard: "unclear" }, note);
    },
    // Whether the site sounds natural (only a site with a `natural` field asks it).
    judge(natural, note) {
      return asksNatural(current() ?? {}) ? submit({ natural }, note) : Promise.resolve(false);
    },
    state: () => ({
      index, key: current()?.key ?? null, playing, saving,
      answered: answered(), total: sites.length, queued: queued(),
    }),
  };
}
