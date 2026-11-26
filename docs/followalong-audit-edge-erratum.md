# Follow-along audit: draw amendment 1, erratum to §6 — the edge words are the ayah's (2026-11-25)

**Status:** dated 2026-11-25, approved by the owner on 2026-11-26 **before** the S5 scorer (#311
PR2) computes any edge sensitivity, and **before any listening**. This is an erratum, not an edit:
it lands as a new file, `docs/followalong-audit-edge-erratum.md`.
[`docs/followalong-audit-draw-amendment.md`](followalong-audit-draw-amendment.md) (SHA-256
`3c5cc33803a3497c80cba2f08b6de146afd955d5bd1262b70f3207cd40369d91`) and
[`docs/followalong-audit-preregistration.md`](followalong-audit-preregistration.md) (SHA-256
`33d3ba26dbe6df3a72b0a03637934274a1c2526ac9e0dc306afd1545ea28ed39`) stay byte-identical, and so
does the committed draw (`tools/tadabur/followalong_audit/s5_draw/`, seed `9665042067186208708`).
It changes no number, no stratum, no drawn word and no rule.

Owner approval: approved by the owner 2026-11-26
(https://github.com/sysofwan/HifzGuide/issues/311#issuecomment-7186142023)

## The correction

Amendment §6 gave the sensitivity's edge words two definitions, and they disagree:

1. "that ayah's first and last Muraja phoneme words";
2. "the same edges as the S5 leading/trailing classes".

The S5 classes are measured from the App-graded span (`strata_run._edges`): an S5 word is leading
or trailing when it lies before the clip's first App-graded word or after its last. A reviewer
found that 17 of the 20 drawn S5 edge words are not the ayah's first or last word, so the two
definitions pick different words.

**The second definition is struck.** The edge words in the §3/§7 sensitivity ("every headline
rate is also reported without the first and last word of each clip's ayah range",
preregistration lines 230 and 390) are, for each clip, the words in the App word table that are
the ayah's **first Muraja phoneme word** (word 1) and its **last** (the clip's highest word
number). A one-word ayah loses that word.

This follows the preregistration's own words, "each clip's ayah range". It also depends only on
the ayah's text, never on what the App graded, so the sensitivity cannot move with the outcome it
tests.

The S5 leading/trailing classes keep their frozen role, unchanged:

- they are draw buckets;
- S5-edge words are a separately labelled bound outside the headline (preregistration §7).

There is still **one** edge-sensitivity row per headline rate, and no second row is added.
