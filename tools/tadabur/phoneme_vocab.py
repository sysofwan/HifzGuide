"""Canonical 43-class Muaalem phoneme vocabulary and CTC greedy decode.

This is the phoneme head's output vocabulary — the single source of truth shared
by three places that must agree exactly:

1. The upstream model ``obadx/muaalem-model-v3_2`` (its ``vocab.json`` ``phonemes``
   level, and ``config.level_to_vocab_size["phonemes"] == 43``).
2. Muraja's on-device ``PhonemeVocabulary`` / ``phonemeMap`` (Swift), which mirrors
   the model so decoded ids map to the same characters the scorer expects.
3. This repo's Tadabur filter, which decodes the model's phoneme logits into the
   character strings the ported ``.balanced`` scorer (``tadabur.normalization`` /
   ``tadabur.scorer``) compares against the ``quran-transcript`` reference.

If any of these drift, the fine-tune labels or the on-device decode would be
silently corrupted. The authoritative reference is a committed Muraja snapshot
(``fixtures/muraja_phoneme_vocabulary.json``, exported from Muraja's Swift
``PhonemeVocabulary``); ``test_phoneme_vocab.py`` asserts *both* this constant and
the live model (1) equal that snapshot exactly and fails loudly on any mismatch — so
this constant and the model can no longer silently agree while drifting from Muraja.

Index ``0`` (``[PAD]``) is the CTC blank (``config.pad_token_id == 0``); greedy
decode collapses repeats and drops it.
"""

from __future__ import annotations

from collections.abc import Sequence

# CTC blank == pad token. Collapsed away during greedy decode; never emitted.
PHONEME_PAD_ID = 0
PAD_TOKEN = "[PAD]"

# The 43 phoneme classes ordered by class id (index == id). Kept in lockstep with
# Muraja's on-device ``PhonemeVocabulary`` (asserted by test against the committed
# ``fixtures/muraja_phoneme_vocabulary.json`` snapshot and the live model); do not
# reorder or edit — the ids are the model's output classes. Includes the 29 Arabic
# consonants (with ``ٲ``/``ا``), the three short-vowel tashkeel (``َ ُ ِ``), the
# madd/tajweed markers, and the ghunna variants (``۾``/``ں``) the normalizer folds.
PHONEME_ID_TO_CHAR: tuple[str, ...] = (
    PAD_TOKEN,  # 0  CTC blank
    "\u0621",   # 1  ء
    "\u0628",   # 2  ب
    "\u062A",   # 3  ت
    "\u062B",   # 4  ث
    "\u062C",   # 5  ج
    "\u062D",   # 6  ح
    "\u062E",   # 7  خ
    "\u062F",   # 8  د
    "\u0630",   # 9  ذ
    "\u0631",   # 10 ر
    "\u0632",   # 11 ز
    "\u0633",   # 12 س
    "\u0634",   # 13 ش
    "\u0635",   # 14 ص
    "\u0636",   # 15 ض
    "\u0637",   # 16 ط
    "\u0638",   # 17 ظ
    "\u0639",   # 18 ع
    "\u063A",   # 19 غ
    "\u0641",   # 20 ف
    "\u0642",   # 21 ق
    "\u0643",   # 22 ك
    "\u0644",   # 23 ل
    "\u0645",   # 24 م
    "\u0646",   # 25 ن
    "\u0647",   # 26 ه
    "\u0648",   # 27 و
    "\u064A",   # 28 ي
    "\u0627",   # 29 ا
    "\u06E6",   # 30 ۦ  small yaa
    "\u06E5",   # 31 ۥ  small waw
    "\u064E",   # 32 َ   fatha
    "\u064F",   # 33 ُ   damma
    "\u0650",   # 34 ِ   kasra
    "\u06EA",   # 35 ۪   empty centre low stop
    "\u0640",   # 36 ـ   tatweel
    "\u0672",   # 37 ٲ   alef with wavy hamza above
    "\u0687",   # 38 ڇ
    "\u06BA",   # 39 ں   ghunna noon
    "\u06FE",   # 40 ۾   ghunna meem
    "\u06DC",   # 41 ۜ   small high seen
    "\u0619",   # 42 ؙ   small damma
)

NUM_PHONEME_CLASSES = len(PHONEME_ID_TO_CHAR)

# char -> id, the exact mapping the model's ``vocab.json`` ``phonemes`` level holds.
PHONEME_CHAR_TO_ID: dict[str, int] = {
    char: idx for idx, char in enumerate(PHONEME_ID_TO_CHAR)
}


def greedy_ctc_decode(class_ids: Sequence[int]) -> str:
    """Collapse a per-frame class-id sequence into a phoneme string.

    Standard CTC greedy decode: collapse consecutive identical ids, then drop the
    blank (``PHONEME_PAD_ID``), and map the survivors to their phoneme characters.
    Input is one frame-argmax id per timestep (already argmaxed over the 43 logits).
    """
    decoded: list[str] = []
    previous: int | None = None
    for class_id in class_ids:
        if class_id != previous:
            previous = class_id
            if class_id != PHONEME_PAD_ID:
                decoded.append(PHONEME_ID_TO_CHAR[class_id])
    return "".join(decoded)
