# Vendored: `obadx/quran-muaalem` Muaalem model

These two files are copied **verbatim** from the upstream Muaalem repo:

- Source repo: <https://github.com/obadx/quran-muaalem>
- Path: `src/quran_muaalem/modeling/`
- Commit: `e9e692c87667ea6353486b2429bfcbaf32670cbe`
- Files: `configuration_multi_level_ctc.py`, `modeling_multi_level_ctc.py`

## Why vendored (not pip-installed / trust_remote_code)

- The upstream `quran-muaalem` package declares `torch>=2.7.0` as a hard
  dependency. `pip install`-ing it into this Linux/CUDA env risks pip replacing
  the Blackwell (`sm_120`) `cu128` torch build with an incompatible wheel — the
  exact failure `tools/requirements-train.txt` warns about.
- The model repo `obadx/muaalem-model-v3_2` ships no `modeling_*.py` and no
  `auto_map` in `config.json`, so `AutoModel.from_pretrained(..., trust_remote_code=True)`
  has no remote code to load.

Copying the two self-contained modeling files (they depend only on `transformers`
and each other) is the deterministic, cu128-safe way to get the model class.

## Updating

Re-copy the two files from the upstream path above, update the commit hash here,
and re-run `python -m pytest tools/tadabur/test_phoneme_vocab.py` to confirm the
phoneme vocabulary is unchanged.
