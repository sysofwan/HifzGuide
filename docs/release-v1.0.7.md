# v1.0.7: the round-3 candidate as the shipped model (draft release, staged)

**Status (2026-11-18):** staged, not published. The GitHub release `v1.0.7` exists as a
**draft** and the Hugging Face repository is staged in a local directory. Publishing both is
the owner's call.

Release notes headline: models.zip now carries the round-3 candidate's **averaged (EMA)**
weights, replacing C95 (#95's model) as the shipped `MuaalemStudent_h448_6BIT.mlmodelc`. The
graph is byte-identical; only the weights change; the I/O contract, filename and asset set are
unchanged.

## The model

| | |
| --- | --- |
| checkpoint (cuda-dev) | `/root/scratch/issue-252/runs/consonant3_candidate/checkpoint.pt` |
| checkpoint (NAS backup) | `/Volumes/NetBackup/HifzGuide/checkpoints/consonant3-candidate-issue254/checkpoint.pt` |
| checkpoint (local copy) | `tools/runs/consonant3_candidate/checkpoint.pt` (gitignored) |
| sha256 | `1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9` |
| size | 1,873,405,033 B |
| run | preset `h448`, step 10000, `ema_decay` 0.999, recipe `aug94-mixture-v2`, seeds 2030/2031 |
| exported weights | **ema** — the round's `choice.json` pins `r3_candidate_ema` at this sha256 |
| `run_config.json` sha256 | `90219ec7fed91529b1ac9f9316bcbb51e70f46df42534c728dd006fe58a611e2` |
| teacher | `obadx/muaalem-model-v3_2` @ `01a1ef9fbe40d144ef845101e89ff924aed3fef5` |
| architecture | 116,318,635 parameters, 43 CTC classes, rotary, `(1,250,160) -> (1,125,43)` fp16 |

The file was copied from the NAS backup and its sha256 recomputed locally; it matches the hash
`docs/consonant-round3-candidate.md` records and the value `tools/tadabur/consonant3_readout/`
pins. The round's verdict is `docs/consonant-round3-readout.md` — this note ships that
candidate, it does not re-decide it.

## Export (macOS)

```bash
cd tools
python3 export_student_coreml.py \
  --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \
  --output-dir student_coreml --nbits 6 --compile --benchmark
bash compile_models.sh student_coreml compiled_models   # -> tools/models.zip
```

- Loaded `h448 … @ step 10000 (ema)`; trace parity on an unseen input: max abs diff `0.00e+00`.
- FP16 222.6 MB → 6-bit 84.4 MB → compiled 84.5 MB (against the 99 MB proven-chunk budget).
- ANE median 12.3 ms/window (M4, `CPU_AND_NE`); duty cycle 0.074 at 6 inferences/s. This is
  host evidence only — single-chunk residency on the iPhone 13 ANE stays unproven (#97 step 4).
- `student_coreml/sizing_h448.json` records `weights: ema`, the checkpoint path and its sha256.
  The verifier refuses a mismatched pairing, so the record and the package cannot disagree.

## Export fidelity (300 gate_eval dev clips, 24,390 teacher phonemes)

```bash
python3 verify_student_export.py --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \
  --eval-set tadabur/gate_eval --export student_coreml --num-clips 300
```

| model | vs teacher | delta vs PyTorch | exact | drift from fp32 |
| --- | --- | --- | --- | --- |
| pytorch fp32 (EMA) | 94.690% | — | 18.67% | 0% |
| coreml fp16 | 94.715% | +0.025 pt | 19.00% | 0.144% |
| coreml 6-bit | 94.707% | +0.016 pt | 18.33% | 0.365% |

**No material drift.** 6-bit costs 0.016 pt of agreement with the teacher, and drift (the upper
bound on damage: characters that moved at all) is 0.365% — the same order as v1.0.6's 0.353%.

Against v1.0.6 on the same 300 clips: PyTorch fp32 94.695% → 94.690% (−0.005 pt), 6-bit
94.645% → 94.707% (+0.062 pt), fp16 drift 0.127% → 0.144%, 6-bit drift 0.353% → 0.365%. The
export path is unchanged; the movement is quantisation noise on a different weight set.

## Assets

| file | size (B) | sha256 |
| --- | --- | --- |
| models.zip | 86,034,195 | `f8dc3311fe4506c505b22c4a205d30fdcf8c3b7a45b9cd1ae9008d1f7b12c439` |
| fonts.zip | 136,608,944 | `975e8f00f9212b78dc03385e7a7769871c4564ce0814dee9394c4824db0ae51b` |
| quran.db | 54,906,880 | `7189c886c23f51ed289e7bb9133299233c153fdc729230a14e019c6bd3c2dddd` |
| mel_filters.bin | 82,240 | `6971fb56fbd4a23b5c56df099fda1aa34a6e1daa060a7437f47d07d1663ce355` |
| window.bin | 1,600 | `36444c5814867e099a289b76c2a6adc4c7fe721bb64edc54910dc74607a7ca6f` |
| manifest.json | 936 | `529030de9400a8ede8d1a03b7c4faa8c33e5c683d3cada5e6fce4e83ecc10204` |

- Staged in `/Users/sysofwan/repos/HifzGuide/.orch/tasks/r3-release/assets/v1.0.7/`; the
  manifest was generated **from those staged files** and every checksum re-read from them
  (all match), `"version": "v1.0.7"`.
- `models.zip` is 8,822 B larger than v1.0.6's 86,025,373 B (+0.01%) — the same graph, new
  weights. Inside it, `model.mil` sha256
  `fb0c6fb1bca04b9de8ef68e1f5f4d157340561b9ecc9f633210e922798c5adbc` is **byte-identical** to
  v1.0.6; `weights/weight.bin` is `d59829f0adf33971b6791ea8b00c33dd264850a6b7ca5f2cb3e094bf1f35fc10`
  against v1.0.6's `4d455c8d…`.
- `fonts.zip`, `quran.db`, `mel_filters.bin` and `window.bin` are copied from the v1.0.6 stage
  and byte-identical (their checksums are unchanged from its manifest).

## Hugging Face stage

```bash
cd tools
python3 stage_hf_model.py --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \
  --processor-config preprocessor_config.json \
  --output-dir ../hf-stage/hifzguide-muaalem-student-h448
```

Staged at `/Users/sysofwan/repos/HifzGuide/.orch/tasks/r3-release/hf-stage/hifzguide-muaalem-student-h448/`
(465.7 MB, not uploaded, no repository created):

| file | sha256 |
| --- | --- |
| `student_state_dict.pt` (445 MB, fp32 EMA) | `c5961ef221e5962b56d9379a977ad2f1156a046c49d05bb701ccf89727642ed4` |
| `config.json` | `fc443ab841676333c97eddff53ce60adf6ef630137f0b21064f20a28e8738322` |
| `phoneme_vocab.json` | `81b5ac466a801723e5229984ab9fe5bc135f69d4a3d6e8c2b9687dfdde9c8b90` |
| `preprocessor_config.json` | `8e6281aad64f97e40534135a59dcc5d33571efae376f2a25adf5551951897ab4` |
| `attribution/noise_pool.json` | `16c099d1099b43108dc1ab0ba7f4e0d3292cf6f17a268eee25f130621f5f574b` |
| `README.md` (draft model card) | see `sha256sums.txt` |
| `LICENSE` | see `sha256sums.txt` |

- `student_state_dict.pt` holds 772 tensors and equals the checkpoint's averaged weights
  exactly, and differs from the live weights in all 772; `build_student(PRESETS["h448"])`
  loads it (`116,318,635` parameters). Its config records `weights: ema`, the checkpoint sha256
  and the teacher pin, so the staged weights cannot be mistaken for the live ones.
- `preprocessor_config.json` is the teacher's own file, byte-identical (`SeamlessM4TFeatureExtractor`,
  16 kHz, 80 mels, stride 2, `padding_value: 1`); the staged copy's sha256 matches the
  downloaded file's.
- The card declares `license: cc-by-nc-4.0` and `base_model: obadx/muaalem-model-v3_2`, and
  records the base model's MIT notice verbatim, both training sets (Tadabur CC BY-NC 4.0,
  tlog CC BY 4.0), the noise pool's per-recording licences, intended use, out-of-scope uses,
  limitations, and the round-3 evaluation summary.
- Repository name and visibility are the owner's; nothing was pushed.

## The draft release

`gh release create v1.0.7 --draft` on `sysofwan/HifzGuide`, title
`v1.0.7 — models.zip ships the round-3 candidate (EMA weights)`, with the five assets plus
`manifest.json`. The tag points at the repository's default-branch commit at draft time
(`bac196470ccf3a9d2f343449a6ca89b4e44a40e5`); the release stays a draft until the owner
publishes it.

Created and checked: `isDraft: true`, target commit `bac1964`, and the six assets uploaded with
the staged names and byte counts — `manifest.json` 936, `mel_filters.bin` 82,240, `window.bin`
1,600, `quran.db` 54,906,880, `models.zip` 86,034,195, `fonts.zip` 136,608,944. The uploaded
`manifest.json`, downloaded again, is byte-identical to the staged file (sha256 `529030de…`).
GitHub creates no tag ref for a draft, so `v1.0.7` becomes a real tag only when the owner
publishes.

## Reproducing the checks

```bash
# asset checksums
python3 -c "import hashlib,json;m=json.load(open('manifest.json'));print(all(hashlib.sha256(open(a['filename'],'rb').read()).hexdigest()==a['sha256'] for a in m['assets']))"
# staged HF weights are the averaged ones
python3 -c "import torch;print(len(torch.load('student_state_dict.pt',weights_only=True)))"   # 772
# the staging tool's tests
scripts/test.sh test_stage_hf_model.py
```

## Disclosures carried into the release notes

- Round 3 has **no control arm**: candidate − C95 (−59.2 false flags per 1,000 graded words,
  95% [−67.8, −51.7] on 5,651 graded clips / 113 surah clusters) cannot separate "more tlog
  data" from "10,000 more steps". Round 2 measured that separation once.
- Catch loss: 16 of 92 decisive removals were real mistakes, 17.4% (Wilson 95%
  [11.0%, 26.4%]) — the point estimate passes the ≤ 20% bar, the upper bound crosses it.
- Tashkeel is flat (−0.03 per 1,000, upper bound +1.38 against a +3.5 guard), and that guard
  vetoes gross regressions only.
- Single-chunk ANE residency on the device is still unproven; the 99 MB budget is
  demonstrated, not specified.
