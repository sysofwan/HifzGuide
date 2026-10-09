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

export const ANSWER_GUARD_MS = 400;

export function createSession({ api, player, view, clock = () => Date.now() }) {
  let sites = [];
  let index = 0;
  let saving = false;
  let shownAt = -Infinity;
  let playing = null;
  let draft = {};

  const current = () => sites[index];
  const asksNatural = (site) => "natural" in site;
  const complete = (site) => site.heard !== null && (!asksNatural(site) || site.natural !== null);
  const answered = () => sites.filter(complete).length;

  // The answer the server holds for the site, as `previous` names it.
  function stored(site) {
    if (site.heard === null) return null;
    return asksNatural(site) ? { heard: site.heard, natural: site.natural } : site.heard;
  }

  function firstUnanswered(from) {
    for (let k = 0; k < sites.length; k++) {
      const i = (from + k) % sites.length;
      if (!complete(sites[i])) return i;
    }
    return -1;
  }

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
    index = i;
    shownAt = clock();
    view.render(current(), { position: index + 1, total: sites.length, answered: answered() });
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
    Object.assign(site, answer, { note });
    view.error("");
    const next = firstUnanswered(index + 1);
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
    // What was said at the carrier.
    answer: (heard, note) => submit({ heard }, note),
    // Whether the site sounds natural (only a site with a `natural` field asks it).
    judge(natural, note) {
      return asksNatural(current() ?? {}) ? submit({ natural }, note) : Promise.resolve(false);
    },
    state: () => ({
      index, key: current()?.key ?? null, playing, saving,
      answered: answered(), total: sites.length,
    }),
  };
}
