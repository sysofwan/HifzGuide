#!/usr/bin/env python3
"""
Generate quran.db from multiple data sources:
  - qpc-hafs-word-by-word.json — QPC Hafs word text
  - ayah_phonemes.json — phoneme reference strings
  - qpc-v2-15-lines.db — mushaf page layout (604 pages × 15 lines)
  - qpc-v2-glyphs.db — per-page glyph text for QCF2 fonts
  - quran-metadata-surah-name.json — surah names/metadata
  - quran-transcript library — phoneme groups, char-level mappings

Tables:
  surahs(id, name_arabic, name_simple, name_english, verses_count, revelation_place, bismillah_pre)
  ayahs(surah, ayah, text, phonemes)
  words(surah, ayah, word, word_id, text, glyph_text)
  word_map(surah, ayah, phoneme_word, text_word)
  phoneme_groups(surah, ayah, group_idx, group_text, ph_start, ph_end, uthmani_word)
  phoneme_char_map(surah, ayah, uthmani_idx, ph_start, ph_end, deleted, uthmani_word)
  pages(page_number, line_number, line_type, is_centered, first_word_id, last_word_id, surah_number)

Usage:
  python3 generate_quran_db.py
"""

import json
import sqlite3
import re
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
QPC_PATH = DATA_DIR / "qpc-hafs-word-by-word.json"
MUSHAF_DB_PATH = DATA_DIR / "qpc-v2-15-lines.db"
GLYPH_DB_PATH = DATA_DIR / "qpc-v2-glyphs.db"
SURAH_META_PATH = DATA_DIR / "quran-metadata-surah-name.json"
COMMON_LIGATURES_PATH = DATA_DIR / "ligatures-common.json"
OUTPUT_PATH = SCRIPT_DIR.parent / "ios" / "Muraja" / "Resources" / "quran.db"

try:
    from quran_transcript import Aya, quran_phonetizer, chunck_phonemes
    from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

    MOSHAF = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=6,
        madd_aared_len=2,
    )
    HAS_QURAN_TRANSCRIPT = True
except ImportError:
    HAS_QURAN_TRANSCRIPT = False
    print("WARNING: quran-transcript not installed; phoneme_groups/char_map will be empty")


def is_verse_number(text: str) -> bool:
    """Check if text is purely Eastern Arabic numerals (verse-end marker)."""
    stripped = text.strip()
    if not stripped:
        return False
    return all(
        0x0660 <= ord(ch) <= 0x0669 or 0x06F0 <= ord(ch) <= 0x06F9
        for ch in stripped
    )


# ---------------------------------------------------------------------------
# Muqatta'at letter merging
# ---------------------------------------------------------------------------

_MUQATTAAT_LETTER_PREFIXES = [
    "ءَلِف", "لَاا", "مِۦ", "ممم", "رَاا", "صَاا", "طَاا",
    "هَاا", "سِۦ", "يَاا", "حَاا", "عَيي", "قَاا", "نُۥ", "كَاا",
]


def _is_letter_name(token: str) -> bool:
    return any(token.startswith(p) for p in _MUQATTAAT_LETTER_PREFIXES)


def merge_muqattaat_phonemes(phon_str: str, text_words: list[tuple[int, str]]) -> str:
    """Merge initial phoneme tokens for muqatta'at letters into a single token."""
    real_words = [(idx, text) for idx, text in text_words if not is_verse_number(text.strip())]
    if not real_words:
        return phon_str

    first_text = real_words[0][1]
    _MUQATTAAT_LETTERS = set('الٓمٓصرطهكيعصسحقنۚ')
    has_muqattaat_mark = any(c in first_text for c in '\u0653\u06DA')
    is_pure_letters = len(first_text) <= 6 and all(c in _MUQATTAAT_LETTERS for c in first_text)
    if not has_muqattaat_mark and not is_pure_letters:
        return phon_str

    ph_words = phon_str.split()
    merge_count = 0
    for token in ph_words:
        if _is_letter_name(token):
            merge_count += 1
        else:
            break
    if merge_count <= 1:
        return phon_str

    merged = ''.join(ph_words[:merge_count])
    return ' '.join([merged] + ph_words[merge_count:])


# ---------------------------------------------------------------------------
# Mapping-based builders using quran-transcript
# ---------------------------------------------------------------------------

def _phonetize_ayah(surah: int, ayah: int):
    """Run quran_phonetizer for an ayah. Returns (uthmani, result) or None."""
    if not HAS_QURAN_TRANSCRIPT:
        return None
    try:
        seg = Aya(surah, ayah).get()
        result = quran_phonetizer(seg.uthmani, MOSHAF)
        return seg.uthmani, result
    except Exception:
        return None


# Ayahs where quran-transcript's phonetizer fails (ending with sukoon).
# Phonemes extracted from muaalem-annotated-v3 dataset with word boundaries
# manually verified.
_FALLBACK_PHONEMES: dict[str, str] = {
    "55:17": "رَببُ لمَشرِقَينِ وَرَببُ لمَغرِبَين",
    "90:8": "ءَلَم نَجڇعَللَهُۥۥ عَينَين",
    "90:9": "وَلِسَاانَوووَشَفَتَين",
    "90:10": "وَهَدَينَااهُ ننننَجڇدَين",
    "106:1": "لِءِۦۦلَاافِ قُرَيش",
    "106:2": "ءِۦۦلَاافِهِم رِحلَتَ ششِتَااااءِ وَصصَيف",
    "106:3": "فَليَعبُدُۥۥ رَببَ هَااذَ لبَيت",
    "106:4": "ءَللَذِۦۦ ءَطڇعَمَهُممممِںںںجُۥۥعِوووَءَاامَنَهُممممِن خَوف",
}


def _build_char_to_word(uthmani: str) -> list[int]:
    """Map each uthmani character to its 1-based word index (0 for spaces)."""
    word_idx = 1
    char_to_word: list[int] = []
    for ch in uthmani:
        if ch == ' ':
            word_idx += 1
            char_to_word.append(0)
        else:
            char_to_word.append(word_idx)
    return char_to_word


def _build_ph_to_words(uthmani: str, phonemes: str, mappings, char_to_word: list[int]) -> list[set[int]]:
    """Map each phoneme character to the set of uthmani word indices it belongs to."""
    ph_to_words: list[set[int]] = [set() for _ in range(len(phonemes))]
    for uth_i, m in enumerate(mappings):
        if m.deleted or uthmani[uth_i] == ' ':
            continue
        uw = char_to_word[uth_i]
        for pi in range(m.pos[0], m.pos[1]):
            if pi < len(phonemes):
                ph_to_words[pi].add(uw)
    return ph_to_words


def generate_phonemes_with_word_boundaries(surah: int, ayah: int) -> str | None:
    """Generate phoneme string from quran-transcript with word boundaries preserved.

    The phonetizer merges words when tajweed rules connect them (idgham, ikhfaa, etc.).
    This function uses the character-level mappings to re-insert spaces at word boundaries,
    so each text word gets its own phoneme word.
    """
    pair = _phonetize_ayah(surah, ayah)
    if pair is None:
        return None

    uthmani, result = pair
    phonemes = result.phonemes
    char_to_word = _build_char_to_word(uthmani)
    ph_to_words = _build_ph_to_words(uthmani, phonemes, result.mappings, char_to_word)

    current_word = 0
    reconstructed = ''
    for i, ch in enumerate(phonemes):
        if ch == ' ':
            # Original space — skip, we'll insert our own at word transitions
            continue
        w = min(ph_to_words[i]) if ph_to_words[i] else current_word
        if w > current_word and current_word > 0:
            reconstructed += ' '
        reconstructed += ch
        current_word = w

    return reconstructed


def build_mappings_for_ayah(
    surah: int, ayah: int, phonemes_str: str
) -> tuple[
    list[tuple[int, str, int, int, int]],   # phoneme_groups rows
    list[tuple[int, int, int, int, int]],    # phoneme_char_map rows
    list[tuple[int, int]],                   # word_map pairs
] | None:
    """Use quran-transcript to build phoneme groups, char map, and word map."""
    pair = _phonetize_ayah(surah, ayah)
    if pair is None:
        return None

    uthmani, result = pair
    mappings = result.mappings
    char_to_word = _build_char_to_word(uthmani)

    # phoneme_char_map rows
    char_map_rows: list[tuple[int, int, int, int, int]] = []
    for uth_i, m in enumerate(mappings):
        uw = char_to_word[uth_i] if uth_i < len(char_to_word) else 0
        char_map_rows.append((uth_i, m.pos[0], m.pos[1], 1 if m.deleted else 0, uw))

    # Build phoneme char → uthmani word(s) from raw phonemes
    raw_phonemes = result.phonemes
    ph_to_words_raw = _build_ph_to_words(uthmani, raw_phonemes, mappings, char_to_word)

    # Remap ph_to_words from raw phoneme positions to phonemes_str positions.
    # Both have the same non-space characters in the same order, but different spacing.
    ph_to_words: list[set[int]] = [set() for _ in range(len(phonemes_str))]
    ri = 0  # index into raw_phonemes
    for si, ch in enumerate(phonemes_str):
        if ch == ' ':
            continue
        while ri < len(raw_phonemes) and raw_phonemes[ri] == ' ':
            ri += 1
        if ri < len(ph_to_words_raw):
            ph_to_words[si] = ph_to_words_raw[ri]
        ri += 1

    # phoneme_groups rows (from the merged phonemes_str)
    groups = chunck_phonemes(phonemes_str)
    group_rows: list[tuple[int, str, int, int, int]] = []
    pos = 0
    for gi, g in enumerate(groups):
        idx = phonemes_str.find(g, pos)
        if idx < 0:
            idx = pos
        ph_start = idx
        ph_end = idx + len(g)
        group_words: set[int] = set()
        for ci in range(ph_start, min(ph_end, len(ph_to_words))):
            group_words.update(ph_to_words[ci])
        uw = min(group_words) if group_words else 0
        group_rows.append((gi, g, ph_start, ph_end, uw))
        pos = ph_end

    # word_map from mappings
    word_map_set: set[tuple[int, int]] = set()
    ph_words = phonemes_str.split()
    ph_pos = 0
    for pwi, pw in enumerate(ph_words):
        pw_start = phonemes_str.find(pw, ph_pos)
        if pw_start < 0:
            pw_start = ph_pos
        pw_end = pw_start + len(pw)
        pw_words: set[int] = set()
        for ci in range(pw_start, min(pw_end, len(ph_to_words))):
            pw_words.update(ph_to_words[ci])
        for tw in pw_words:
            if tw > 0:
                word_map_set.add((pwi + 1, tw))
        ph_pos = pw_end + 1

    if not word_map_set:
        for i in range(1, len(ph_words) + 1):
            word_map_set.add((i, i))

    return group_rows, char_map_rows, sorted(word_map_set)


def main():
    print("Loading qpc-hafs-word-by-word.json...")
    with open(QPC_PATH, "r", encoding="utf-8") as f:
        word_data = json.load(f)

    # Generate phonemes from quran-transcript, with hardcoded fallbacks
    print("Generating phonemes from quran-transcript...")
    phonemes: dict[str, str] = {}
    qt_generated = 0
    fallback_used = 0

    # Iterate all ayahs from the word data
    all_ayah_keys: set[tuple[int, int]] = set()
    for key in word_data:
        parts = key.split(":")
        if len(parts) == 3:
            all_ayah_keys.add((int(parts[0]), int(parts[1])))

    for s, a in sorted(all_ayah_keys):
        ayah_key = f"{s}:{a}"
        qt_phon = generate_phonemes_with_word_boundaries(s, a)
        if qt_phon:
            phonemes[ayah_key] = qt_phon
            qt_generated += 1
        elif ayah_key in _FALLBACK_PHONEMES:
            phonemes[ayah_key] = _FALLBACK_PHONEMES[ayah_key]
            fallback_used += 1
        else:
            print(f"  WARNING: no phonemes for {ayah_key}")
    print(f"  quran-transcript: {qt_generated}, hardcoded fallback: {fallback_used}")

    print("Loading quran-metadata-surah-name.json...")
    with open(SURAH_META_PATH, "r", encoding="utf-8") as f:
        surah_meta = json.load(f)

    print("Loading ligature mappings...")
    with open(COMMON_LIGATURES_PATH, "r", encoding="utf-8") as f:
        common_ligatures = json.load(f)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    print("Creating quran.db...")
    conn = sqlite3.connect(str(OUTPUT_PATH))
    cur = conn.cursor()

    # --- Surahs table ---
    cur.execute("""
        CREATE TABLE surahs (
            id INTEGER PRIMARY KEY,
            name_arabic TEXT NOT NULL,
            name_simple TEXT NOT NULL,
            name_english TEXT NOT NULL,
            verses_count INTEGER NOT NULL,
            revelation_place TEXT,
            bismillah_pre INTEGER NOT NULL DEFAULT 1
        )
    """)
    for sid, meta in surah_meta.items():
        cur.execute(
            "INSERT INTO surahs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (int(sid), meta["name_arabic"], meta["name_simple"], meta["name"],
             meta["verses_count"], meta.get("revelation_place"),
             1 if meta.get("bismillah_pre", True) else 0),
        )

    # --- Ligatures table ---
    cur.execute("""
        CREATE TABLE ligatures (
            key TEXT PRIMARY KEY,
            glyph TEXT NOT NULL
        )
    """)
    for key, glyph in common_ligatures.items():
        cur.execute("INSERT INTO ligatures VALUES (?, ?)", (key, glyph))
    print(f"  ligatures: {len(common_ligatures)}")

    # --- Ayahs table ---
    cur.execute("""
        CREATE TABLE ayahs (
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            text TEXT NOT NULL,
            phonemes TEXT,
            PRIMARY KEY (surah, ayah)
        )
    """)

    # --- Words table ---
    cur.execute("""
        CREATE TABLE words (
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            word INTEGER NOT NULL,
            word_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            glyph_text TEXT,
            PRIMARY KEY (surah, ayah, word)
        )
    """)
    cur.execute("CREATE INDEX idx_words_word_id ON words(word_id)")

    # --- Word map table ---
    cur.execute("""
        CREATE TABLE word_map (
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            phoneme_word INTEGER NOT NULL,
            text_word INTEGER NOT NULL,
            PRIMARY KEY (surah, ayah, phoneme_word, text_word)
        )
    """)

    # --- Phoneme groups table ---
    cur.execute("""
        CREATE TABLE phoneme_groups (
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            group_idx INTEGER NOT NULL,
            group_text TEXT NOT NULL,
            ph_start INTEGER NOT NULL,
            ph_end INTEGER NOT NULL,
            uthmani_word INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (surah, ayah, group_idx)
        )
    """)

    # --- Phoneme char map table ---
    cur.execute("""
        CREATE TABLE phoneme_char_map (
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            uthmani_idx INTEGER NOT NULL,
            ph_start INTEGER NOT NULL,
            ph_end INTEGER NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0,
            uthmani_word INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (surah, ayah, uthmani_idx)
        )
    """)

    # --- Pages table ---
    cur.execute("""
        CREATE TABLE pages (
            page_number INTEGER NOT NULL,
            line_number INTEGER NOT NULL,
            line_type TEXT NOT NULL,
            is_centered INTEGER NOT NULL DEFAULT 0,
            first_word_id INTEGER,
            last_word_id INTEGER,
            surah_number INTEGER,
            PRIMARY KEY (page_number, line_number)
        )
    """)

    # Import pages from mushaf layout DB
    print("Importing mushaf page layout...")
    mushaf_conn = sqlite3.connect(str(MUSHAF_DB_PATH))
    for row in mushaf_conn.execute(
        "SELECT page_number, line_number, line_type, is_centered, "
        "first_word_id, last_word_id, surah_number FROM pages"
    ):
        cur.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?)", row)
    mushaf_conn.close()

    # Load glyph text
    print("Loading glyph text from qpc-v2-glyphs.db...")
    glyph_map: dict[int, str] = {}
    glyph_conn = sqlite3.connect(str(GLYPH_DB_PATH))
    for row in glyph_conn.execute("SELECT id, text FROM words"):
        glyph_map[row[0]] = row[1]
    glyph_conn.close()
    print(f"  Loaded {len(glyph_map)} glyph entries")

    # Group words by surah:ayah
    grouped: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    for key, entry in word_data.items():
        parts = key.split(":")
        if len(parts) != 3:
            continue
        s, a, w = int(parts[0]), int(parts[1]), int(parts[2])
        grouped.setdefault((s, a), []).append((w, entry["id"], entry["text"]))

    # Insert words and build ayah text
    word_count = 0
    ayah_texts: dict[tuple[int, int], str] = {}
    for (s, a), words in sorted(grouped.items()):
        words.sort(key=lambda x: x[0])
        for w_idx, w_id, w_text in words:
            glyph = glyph_map.get(w_id)
            cur.execute("INSERT INTO words VALUES (?, ?, ?, ?, ?, ?)",
                        (s, a, w_idx, w_id, w_text, glyph))
            word_count += 1
        display_words = [w_text for _, _, w_text in words if not is_verse_number(w_text)]
        ayah_texts[(s, a)] = " ".join(display_words)

    # Insert ayahs with phonemes (merge muqatta'at)
    ayah_count = 0
    all_keys = set(ayah_texts.keys())
    for key in phonemes:
        parts = key.split(":")
        if len(parts) == 2:
            all_keys.add((int(parts[0]), int(parts[1])))

    for s, a in sorted(all_keys):
        text = ayah_texts.get((s, a), "")
        phon = phonemes.get(f"{s}:{a}", None)
        if phon and (s, a) in grouped:
            wl = sorted([(w, t) for w, _, t in grouped[(s, a)]], key=lambda x: x[0])
            phon = merge_muqattaat_phonemes(phon, wl)
            phonemes[f"{s}:{a}"] = phon
        cur.execute("INSERT INTO ayahs VALUES (?, ?, ?, ?)", (s, a, text, phon))
        ayah_count += 1

    conn.commit()

    # Build phoneme groups, char map, and word map
    print("Building phoneme groups and mappings...")
    map_count = 0
    group_count = 0
    charmap_count = 0
    mapping_fails = 0

    for key, phon_str in phonemes.items():
        parts = key.split(":")
        if len(parts) != 2:
            continue
        s, a = int(parts[0]), int(parts[1])

        result = build_mappings_for_ayah(s, a, phon_str)

        if result is not None:
            group_rows, char_map_rows, word_map_pairs = result

            for gi, g_text, ph_s, ph_e, uw in group_rows:
                cur.execute("INSERT INTO phoneme_groups VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (s, a, gi, g_text, ph_s, ph_e, uw))
                group_count += 1

            for uth_i, ph_s, ph_e, deleted, uw in char_map_rows:
                cur.execute("INSERT INTO phoneme_char_map VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (s, a, uth_i, ph_s, ph_e, deleted, uw))
                charmap_count += 1

            for pw, tw in word_map_pairs:
                cur.execute("INSERT INTO word_map VALUES (?, ?, ?, ?)", (s, a, pw, tw))
                map_count += 1
        else:
            mapping_fails += 1
            # Fallback: identity word_map
            ph_words = phon_str.split()
            for i in range(1, len(ph_words) + 1):
                cur.execute("INSERT INTO word_map VALUES (?, ?, ?, ?)", (s, a, i, i))
                map_count += 1

    conn.commit()

    if mapping_fails:
        print(f"  Mapping fallbacks: {mapping_fails} ayahs")

    # Print stats
    cur.execute("SELECT COUNT(*) FROM surahs")
    print(f"  surahs: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ayahs")
    print(f"  ayahs: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM words")
    print(f"  words: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM ayahs WHERE phonemes IS NOT NULL")
    print(f"  ayahs with phonemes: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM word_map")
    print(f"  word mappings: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM phoneme_groups")
    print(f"  phoneme groups: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM phoneme_char_map")
    print(f"  char map entries: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(DISTINCT page_number) FROM pages")
    print(f"  pages: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM pages")
    print(f"  page lines: {cur.fetchone()[0]}")

    # Verify samples
    cur.execute(
        "SELECT phoneme_word, text_word FROM word_map WHERE surah=2 AND ayah=2 "
        "ORDER BY phoneme_word, text_word"
    )
    print(f"\n  Sample 2:2 word_map: {cur.fetchall()}")

    cur.execute("SELECT text, phonemes FROM ayahs WHERE surah=1 AND ayah=1")
    row = cur.fetchone()
    print(f"  Sample 1:1 text: {row[0]}")
    print(f"  Sample 1:1 phonemes: {row[1]}")

    cur.execute(
        "SELECT group_idx, group_text, uthmani_word FROM phoneme_groups "
        "WHERE surah=2 AND ayah=2 LIMIT 10"
    )
    print(f"  Sample 2:2 groups: {cur.fetchall()}")

    cur.execute("SELECT name_arabic FROM surahs WHERE id=1")
    print(f"  Sample surah 1: {cur.fetchone()[0]}")

    cur.execute(
        "SELECT page_number, line_number, line_type, first_word_id, last_word_id "
        "FROM pages WHERE page_number=2 LIMIT 5"
    )
    print(f"  Sample page 2 lines: {cur.fetchall()}")

    conn.close()

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nGenerated {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
