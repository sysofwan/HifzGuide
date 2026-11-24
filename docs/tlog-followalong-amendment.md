# tlog follow-along audit: confirm.1 and confirm.2 are reused under the mining amendment (2026-11-24)

**Status:** dated 2026-11-24, frozen the day it lands, and **before any draw or listening** of the
follow-along audit (issue [#310](https://github.com/sysofwan/HifzGuide/issues/310)). This is an
amendment, not an edit: it records a reuse already allowed, and changes nothing.

Issue: [#310](https://github.com/sysofwan/HifzGuide/issues/310). Parent: #305 (the follow-along
audit); tlog #173 and mining #261.

## 1. The reuse

The follow-along audit grades every claim-gated **confirm.1** and **confirm.2** gating clip —
5,742 clips after #248's exclusion — through the App arm, the Oracle arm and the Teacher arm. That
is a word-level decode and audit of both tranches by our own models.

[`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md) § 1 already allows it: once a
tranche's claim and its round's verdict are committed, the tranche may be "decoded by any of our
models, listened to and audited at word level". Both tranches meet that condition. This document
records the reuse; it makes **no new claim** and no rule change.

## 2. Status

confirm.1 and confirm.2 are **already mined** and therefore **never sealed again**: no later round
may treat either tranche as sealed or replay it as a gating set. Its rows may be reported beside a
later round's result, never counted in it. This is the mining amendment's status, unchanged.

## 3. Frozen documents this note builds on

- [`docs/tlog-holdout-preregistration.md`](tlog-holdout-preregistration.md), SHA-256
  `204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086`: never edited, byte-identical
  to the frozen text this note builds on.
- [`docs/tlog-mining-amendment.md`](tlog-mining-amendment.md), SHA-256
  `f02bae6271bc1472262164131dee9eaf8e1e80d6f8c1419720421d890ea9313c`: unchanged by this note.

The audit's committed inputs, by full commit:

| slice | commit |
|---|---|
| S0 — the re-pinned harness | `c4bdea30bbaee4bafc81b2489a2a558529b4beb4` |
| S1 — the tracer and gate P1b | `c7c6cd072b7438794f2de582ac685cc606bf8923` |
| S3 — the App arm, pools, P2 and P3 | `9f3f6da7e40da458ef27a38876a31eed49d1cc7c` |
| S2 — the oracle sweep and the S1 fill | `314dd8653735a22f0f2d11d32e469b116715ed72` |

## 4. Approvals

Owner approval: pending

Principal cross-check: pending

## 5. Changelog

- 2026-11-24: written and frozen; records the reuse of confirm.1/confirm.2 under the mining
  amendment's § 1, with no new claim and no rule change.
