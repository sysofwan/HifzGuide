# Margin definition pin: the competitor search (#190)

**Status:** a dated, **post-look** and **definitional** amendment, written **2026-10-26** against
engine commit `8138e8e`. It is an addition, not an edit, to the frozen hold-out text
[`docs/tlog-holdout-preregistration.md`](tlog-holdout-preregistration.md) (SHA-256
`204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086`) and to the frozen S5
pre-registration [`docs/tlog-ident-preregistration.md`](tlog-ident-preregistration.md) (SHA-256
`a383fcfdbd375c9e6c052d8032788a8eda69bd2d9fbd12fcd3b921b1fbb6a3ab`). Both documents stay
byte-identical. The Tarteel v1 look half had already been scored when this was written. It changes
no threshold, no grid value and no input.

## The pinned definition

The competitor of a decode whose winning span is W:

1. for each shortlisted window, take every contiguous **word span** of the window's words outside
   W's word span;
2. score each span by edlib 1.3.9.post1's **first optimal infix alignment** (mode `HW`, task
   `path`) of the decode inside that span's text;
3. that alignment's class is the ayah with the most aligned words, ties to the earlier ayah — the
   frozen attribution;
4. the competitor is the best-ranked such alignment whose class differs from W's, under the frozen
   tie-break: the score, then each window's own winning alignment on a tie, then the earlier span;
5. `m = W.score − competitor.score`, or `W.score` when there is no competitor.

The search is exhaustive over those word spans, bounded only by "a span cannot score above its
parent", so a tied or lower-scoring different-class span is found.

## The disclosed gap

A different-class alignment that starts or ends **inside** a word shared with a same-class optimum
of equal or higher score is not reachable from any word span, so it can be missed. Two fixtures
hold it (`tools/tadabur/test_tlog_ident.py`): with the first two ayat's references as single words
the pinned search reports `m = 0.5` where a character-level search gives 0, and with 1:3 as
`"بتثح"` it reports `m = 0.625` where that search gives at most 0.125. Both tests assert the
pinned values, so the gap is recorded rather than silently re-derived.

The gap is **one-directional**: a missed competitor makes the margin larger, never smaller, so `m`
can only be **overstated**. The identifier can therefore err toward naming a row, never toward
leaving it unnamed.

## Why a pin rather than an exact search

Class is a property of the alignment path, not of a location: one `(start, end)` can carry several
co-optimal paths that credit different ayat, and "most aligned words" is an argmax over the whole
path. An exact search would have to range over every `(start, end, path)` triple, which no finite
search of this shape reaches, so the unit of location has to be defined either way. The frozen
text's own unit is the **word span** (`docs/tlog-holdout-preregistration.md:100`), which makes this
pin the most faithful finite reading.

## Scope

`m` decides one thing: the label of a row that has **no** filename label. Relabelling uses Δ, not
`m`, and `b_u` feeds no gate bar, no floor, no test verdict and no training. The three search fixes
before this pin changed no floor, no bar, no assignment and no retention.

## The bound measurement

S5b (#204) runs one character-start oracle pass on the Tarteel look set plus the fresh confirmation
sets, after the floor is confirmed and before `b_u` is finalised: for every free character start,
edlib `SHW` from that start to the run's end, the frozen attribution, the best different class, and
`m_oracle = min(m, m_oracle)`. It reports the share of clips whose `m` differs, whether each of
those crosses the derived `b_u`, and `b_u` re-derived under the oracle. If the pass changes neither
`b_u` nor any look clip's assignment, this pin is final; otherwise the owner chooses between
keeping the pin and adopting the oracle through a further dated amendment. No automatic switch.
