# b_u erratum: precision of the clips the identifier names (#190)

**Status:** approved by the owner and frozen **2026-10-25**, **post-look**. This is a dated
erratum, not an edit, of the frozen S5 pre-registration
[`docs/tlog-ident-preregistration.md`](tlog-ident-preregistration.md) (SHA-256
`a383fcfdbd375c9e6c052d8032788a8eda69bd2d9fbd12fcd3b921b1fbb6a3ab`), which in turn cites the
frozen hold-out text and is itself never edited. The Tarteel v1 look half had already been
scored when this erratum was written.

## The rule

`b_u` is the **smallest** grid bar at which, of the Tarteel v1 look clips the identifier
**names** at that bar (floor passed, margin ≥ bar, labels hidden), at least **99%** are their
own text class. The bar grid (0.00 … 0.50 by 0.01) and the 99% are unchanged.

## The drafting error it fixes

The original wording counted own-class assignments over **all** look clips. That share can only fall
as the bar rises, because raising the bar only removes assignments, so "the smallest bar reaching
99%" of it is vacuous by construction. The intended quantity — the precision of the
clips the rule accepts — rises with the bar, mirroring how `b` is an error-rate cap met at the
smallest bar.

## Disclosure

The architect's read-only diagnostic on the pre-fix engine measured 97.88% own-class over all
look clips at bar 0.00, and 99.68% precision among the named clips at bar 0.01. Under this
erratum `b_u` is therefore 0.01. The calibration records `b_u` under this rule and, for
transparency, the literal wording's result as `b_u_literal` (None on the look).

## Scope

`b_u` names ayat for rows without a filename label (reports and a future public release). It
touches no gate bar, no floor, no test verdict and no training: `tlog.training` uses no labels.
Nothing else in the S5 pre-registration changes.
