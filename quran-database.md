# Quran Database (`quran.db`)

The app bundles a read-only SQLite database at `ios/Muraja/Resources/quran.db` containing Quran text, phoneme references, word-level mappings, mushaf page layout, surah metadata, and ligature glyph strings. It is generated offline by `tools/generate_quran_db.py` from source files in `data/`.

## Schema

### `surahs` — Surah metadata (114 rows)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Surah number (1–114) |
| `name_arabic` | TEXT | Arabic name (e.g. الفاتحة) |
| `name_simple` | TEXT | Transliterated name (e.g. Al-Fatihah) |
| `name_english` | TEXT | English translation with diacritics |
| `verses_count` | INTEGER | Number of ayahs in the surah |
| `revelation_place` | TEXT | `makkah` or `madinah` |
| `bismillah_pre` | INTEGER | `1` if surah has bismillah prefix, `0` otherwise |

### `ayahs` — Ayah text and phonemes (6,236 rows)

| Column | Type | Description |
|--------|------|-------------|
| `surah` | INTEGER | Surah number (PK part 1) |
| `ayah` | INTEGER | Ayah number within surah (PK part 2) |
| `text` | TEXT | Full ayah text in Uthmanic script (verse-end markers excluded) |
| `phonemes` | TEXT | Phoneme representation used by the follow-along algorithm |

**Example:**
```
surah=1, ayah=1
text:     بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ
phonemes: بِسمِ للَااهِ ررَحمَاانِ ررَحِۦۦم
```

### `words` — Per-word text and glyphs (83,668 rows)

| Column | Type | Description |
|--------|------|-------------|
| `surah` | INTEGER | Surah number (PK part 1) |
| `ayah` | INTEGER | Ayah number (PK part 2) |
| `word` | INTEGER | 1-based word index within the ayah (PK part 3) |
| `word_id` | INTEGER | Global word ID across the entire Quran (indexed) |
| `text` | TEXT | Arabic word text in Uthmanic script |
| `glyph_text` | TEXT | QCF2 glyph character for this word (nullable) |

The `word_id` is a monotonically increasing global identifier that spans the entire mushaf. It is used to join words to page layout lines via `pages.first_word_id` / `last_word_id`.

### `word_map` — Phoneme-to-text word alignment (77,432 rows)

| Column | Type | Description |
|--------|------|-------------|
| `surah` | INTEGER | Surah number (PK part 1) |
| `ayah` | INTEGER | Ayah number (PK part 2) |
| `phoneme_word` | INTEGER | 1-based phoneme word index (PK part 3) |
| `text_word` | INTEGER | 1-based text word index (PK part 4) |

Phoneme words don't always align 1:1 with text words (some Arabic words merge or split at the phoneme level). This table maps between the two so the follow-along algorithm's word-level position can be translated to text highlighting. Built by greedy consonant-skeleton matching in the generator script.

### `pages` — Mushaf page layout (9,046 rows)

| Column | Type | Description |
|--------|------|-------------|
| `page_number` | INTEGER | Mushaf page (1–604) (PK part 1) |
| `line_number` | INTEGER | Line within the page (PK part 2) |
| `line_type` | TEXT | `surah_name`, `basmallah`, or `ayah` |
| `is_centered` | INTEGER | `1` if line should be centered (short ayahs) |
| `first_word_id` | INTEGER | Global word ID of first word on this line (nullable) |
| `last_word_id` | INTEGER | Global word ID of last word on this line (nullable) |
| `surah_number` | INTEGER | Surah number for `surah_name` / `basmallah` lines (nullable) |

The 15-line-per-page layout comes from the QPC v2 dataset. Each page has lines of different types — surah headers, basmallah decorations, and ayah text lines.

### `ligatures` — Common glyph mappings (72 rows)

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT PK | Ligature identifier (e.g. `surah_header`, `bismillah`, `makkah`) |
| `glyph` | TEXT | Glyph string for the quran-common font |

Used for decorative elements like the surah header frame, bismillah text, and revelation-place markers.

## Data Sources

| Source file | Tables populated |
|-------------|-----------------|
| `data/qpc-hafs-word-by-word.json` | `words`, `ayahs.text`, `word_map` |
| `data/ayah_phonemes.json` | `ayahs.phonemes`, `word_map` |
| `data/quran-metadata-surah-name.json` | `surahs` |
| `data/ligatures-common.json` | `ligatures` |
| `data/qpc-v2-15-lines.db` | `pages` |
| `data/qpc-v2-glyphs.db` | `words.glyph_text` |

## Regeneration

```bash
cd tools && python generate_quran_db.py
# Output: ios/Muraja/Resources/quran.db
```

## iOS Access Layer

`QuranDatabase.swift` provides a thin wrapper over the SQLite C API. It opens the bundled database read-only via `sqlite3_open_v2` and exposes typed query methods. A singleton `QuranDatabase.shared` is used throughout the app.

### Methods

| Method | Returns | Used by |
|--------|---------|---------|
| `phonemes(surah:ayah:)` | `String?` | PhonemeIndex, FollowAlongManager |
| `hasAyah(surah:ayah:)` | `Bool` | PhonemeIndex |
| `allPositions()` | `[QuranPosition]` | PhonemeIndex |
| `words(surah:ayah:)` | `[String]` | — |
| `wordMap(surah:ayah:)` | `[Int: Int]` | QuranReaderView |
| `linesForPage(_:)` | `[PageLine]` | QuranReaderView, MushafPageView |
| `wordsForRange(firstWordId:lastWordId:)` | `[WordEntry]` | QuranReaderView, MushafPageView |
| `surahName(_:)` | `String?` | — |
| `surahGlyphName(_:)` | `String?` | — |
| `ligature(_:)` | `String?` | MushafPageView |
| `totalPages()` | `Int` | QuranReaderView |
| `firstPositionOnPage(_:)` | `QuranPosition?` | QuranReaderView |
| `pageForPosition(surah:ayah:)` | `Int?` | QuranReaderView |
| `versesCount(surah:)` | `Int` | — |
| `hasBismillah(surah:)` | `Bool` | — |
| `ayahs(from:to:)` | `[(QuranPosition, String)]` | — |

### How the app uses the database

**Mushaf rendering** — `QuranReaderView` calls `linesForPage()` to get the 15-line layout, then `wordsForRange()` to fetch the words for each ayah line. `MushafPageView` renders these words using either QCF2 glyph text (strict mode) or plain Arabic text (native mode). Decorative elements like surah headers and bismillah are rendered using `ligature()` lookups.

**Follow-along tracking** — `PhonemeIndex` loads all ayah positions and phoneme strings to build a sliding context window around the reader's current position. The Smith-Waterman alignment algorithm matches live transcription against this phoneme reference. `FollowAlongManager` uses phoneme word counts for skip detection.

**Word highlighting** — `wordMap()` translates phoneme-level word positions (from the follow-along algorithm) to text-level word positions (for UI highlighting). This is necessary because phoneme tokenization doesn't always match Arabic text tokenization.

**Page navigation** — `firstPositionOnPage()` and `pageForPosition()` enable bidirectional sync between the page-based mushaf view and the position-based follow-along tracker.
