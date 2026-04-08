# HifzGuide

Assets and data generation tools for [Muraja](https://github.com/sysofwan/Muraja) — a real-time Quran follow-along reading checker for iOS.

## Contents

- **`tools/`** — Python scripts for model conversion, data extraction, and DB generation
- **`data/`** — Source data files (Quran text, phonemes, mushaf layout, ligatures)

## Release Assets

Pre-built assets are published as [GitHub Releases](https://github.com/sysofwan/HifzGuide/releases). Each release contains:

| Asset | Description | Size |
|-------|-------------|------|
| `models.zip` | 6-chunk Wav2Vec2-BERT CoreML models (6-bit palettized) | ~443 MB |
| `fonts.zip` | 604 QCF2 page fonts + decorative fonts | ~130 MB |
| `quran.db` | SQLite database with Quran text, phonemes, word mappings | ~48 MB |
| `mel_filters.bin` | Mel spectrogram filter bank | 80 KB |
| `window.bin` | Audio windowing function | 1.6 KB |
| `manifest.json` | Version metadata with SHA256 checksums | ~1 KB |

The Muraja iOS app downloads these assets on first launch.

## Regenerating Assets

```bash
# Setup
cd tools && pip install -r requirements.txt

# Regenerate quran.db from source data
python generate_quran_db.py

# Convert ML model to CoreML
python convert_to_coreml.py

# Palettize model chunks to 6-bit
python palettize_chunks.py
```

## Acknowledgements

- **[Muaalem Model](https://huggingface.co/obadx/muaalem-model-v2)** — Arabic phoneme recognition model by [obadx](https://huggingface.co/obadx), fine-tuned from Meta's [Wav2Vec2-BERT](https://huggingface.co/facebook/w2v-bert-2.0) for Quranic recitation analysis
- **[Quranic Universal Library (QUL)](https://qul.tarteel.ai/)** by [Tarteel AI](https://tarteel.ai/) — QPC V2 page fonts (QCF2), word-by-word glyph mappings, mushaf layout data, and Quran text resources

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
