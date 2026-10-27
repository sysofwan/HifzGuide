# tlog S6: the overlap between audit strata (2026-10-27)

**Status:** frozen 2026-10-27, before any audit verdict. Decided by the architect (r22, binding);
no owner decision is needed, because this note restores the frozen text rather than changing it.
It clarifies `docs/tlog-s6-amendment.md` (sha256
`5967fe3008ed822d0b61828da23b0c4d86d079667f203c8bb54d23ddb6402aa4`) §4 and §5. It changes no
threshold and no stratum definition, adds no stratum, and filters no row.

Pinned by `tools/tadabur/test_tlog_ident_run.py`; the amendment itself is frozen and is not
edited.

## 1. The rule: no precedence between strata

Amendment §4 defines each stratum by **its own predicate** and says nothing about a row that
satisfies two of them:

- `both`: `label_source == "both"`;
- `non_recitation`: decoded whole-clip, `id_floor_ok == false`.

A decoded row that fails the floor **and** matches its filename class satisfies both, so it is in
both eligible sets (on tlog: 450 of the 2,439 floor-failing rows). §4's draw is "the first n of
each stratum by salted rank", so:

- **each stratum draws from its own full eligible set** — `non_recitation` samples all 2,439
  floor-failing rows, `both` samples every `label_source == "both"` row;
- a row drawn in two strata is **heard once** and **scored in each**, against that stratum's
  label under test (§5):
  - a `right ayah` verdict on a shared row is correct for `both`;
  - the same verdict counts as a recitation heard, against `non_recitation`.

The worklist therefore carries `strata` (every stratum the row was drawn in) and `stratum` (the
first labelled one, the single label the page uses for its right/other choice). Neither field is
served to the listening page. The draw is `audit_draw()` in `tools/tadabur/tlog_ident_run.py`.

## 2. The deviation this removes

The first draw (committed 2026-10-27, before any verdict) applied an `AUDIT_PRIORITY` precedence
in code: `both` took a row that satisfied both predicates. That was an implementation choice, not
a decision, and §4 does not support it. It made `non_recitation` sample the 1,989 floor-failing
rows that do *not* match their filename class, excluding all 450 that do; seven of the full
frame's first 25 were excluded by it.

`AUDIT_PRIORITY` was removed and the `non_recitation` stratum redrawn over the whole 2,439 while
the session had **zero** answers. The `both` 40 stand as drawn. Rows the restricted draw had
taken only to fill the 25 are dropped from the served list; rows already drawn in another
stratum are served once. The served count stays at 150 because none of the newly drawn rows was
already under another stratum.

Removing the precedence also restores the below-bar stratum's own set. That stratum needs the
best class to *differ* from the filename class, and S5 records `label_source == "both"` exactly
when the two are one text class; a `both` row is therefore never a below-bar disagreement, even
when its label strings differ (repeated ayat, e.g. `26:190` and `26:8`).

## 3. The §9 rule (unchanged in substance, pre-stated 2026-10-27 on #191 and #173)

> Sample: the first 25 rows by the frozen salt's rank among all decoded rows with
> `id_floor_ok == false` (2,439), whatever other stratum they also belong to. Count the verdicts
> **right ayah** or **other ayah**, i.e. recitation heard. **unclear** counts as neither. If the
> count is ≥ 3, the 5% rule's whole-pool consequence is suspended pending a dated floor revision
> decided by the owner. Otherwise it proceeds as frozen, through the follow-up issue.

`render_audit()` computes the count over exactly that sample and prints the outcome. The
worklist marks those rows `floor_sample`; the report counts the marked rows and reports every
unmarked `non_recitation` row as an extra.

## 4. Fallback: a sitting that already began

If any verdict exists before the corrected list is live, the worklist stands and the corrected 25
is assembled instead by marking the prefix rows it already holds and appending the rows it lacks:
`with_floor_supplement()` returns the corrected worklist, whose appended part is exactly the
full-frame prefix rows absent from it (at most seven), which the owner hears in a short second
sitting. The restricted-only extras stay unmarked, so they are reported, never counted. §9 then
reads the same 25 as the primary path, so no frame weighting is used.
