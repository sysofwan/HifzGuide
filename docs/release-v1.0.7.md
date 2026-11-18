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

**No material drift.** 6-bit moves +0.016 pt of agreement with the teacher, and drift (the
upper bound on damage: characters that moved at all) is 0.365% — the same order as v1.0.6's
0.353%.

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

Staged by the canonical publisher, `training.publish_hf` — the same tool that produced
`sysofwan/hifzguide-muaalem-mini` — in that publication's layout, so a reader needs nothing
from this repository: `AutoModel`/`AutoFeatureExtractor` plus `trust_remote_code`. (The
separate `tools/stage_hf_model.py` was deleted after self-check r1: one layout writer.)

```bash
cd tools
python3 -m training.publish_hf \
  --checkpoint runs/consonant3_candidate/checkpoint.pt --ema \
  --repo-id sysofwan/hifzguide-muaalem-mini-v2 --profile cc-by-nc \
  --out-dir ../hf-stage/hifzguide-muaalem-mini-v2 --json
```

Staged at `/Users/sysofwan/repos/HifzGuide/.orch/tasks/r3-release/hf-stage/hifzguide-muaalem-mini-v2/`
for the repository name `sysofwan/hifzguide-muaalem-mini-v2`: 465.4 MB, nothing uploaded, no
repository created. Ten files, the same set `sysofwan/hifzguide-muaalem-mini` publishes:

| file | size (B) | sha256 | against the published copy |
| --- | --- | --- | --- |
| model.safetensors | 465,374,196 | `7b760511334e53ca115a2dd466e57a9241c4e3c7a30498916b721d0f6d650d89` | new weights, same byte count |
| config.json | 3,528 | `c5e2b10914eee234e930474a828760b7ff68ef07b79f8e270db065c768f18f41` | byte-identical |
| configuration_multi_level_ctc.py | 18,658 | `bc8d1a13d76c1066a99edf164e6a1136a86bb18d92654d726d691cdc7ef6b93f` | byte-identical |
| modeling_multi_level_ctc.py | 5,891 | `21424f97dcce580b312d3954e92164a486b5031aac057c6a425aa4b153819172` | byte-identical |
| phoneme_vocab.json | 496 | `f90eb258cb4b8563261252ccb4c60a699850077566557c8d878784e9282adfb5` | byte-identical |
| preprocessor_config.json | 229 | `9c6008e1229e6914e5c11df2f673c3a0af5fd6ad442426a4b4eb3a4c11d78264` | byte-identical |
| .gitattributes | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` | byte-identical |
| LICENSE | 19,347 | `41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0` | CC BY-NC 4.0 legal code (published is AGPL-3.0) |
| NOTICE.md | 4,401 | `8dfb037ed0280c914f42c1b5235ed327383128e41ab221bdbff6eae08b0c9b4c` | rewritten: NC licence, tlog and noise attributions |
| README.md | 8,701 | `0e60f350b503f7a84aa265b41429080e3096697a29854c6944333954a2e6575b` | rewritten: comparison table |

- The weight set is the publisher's decision, not the caller's: `--ema` reads the average
  through `training.decoding.load_student_from_checkpoint` and raises when a checkpoint has
  none. The teacher revision is the checkpoint's own — `--teacher-revision` may only repeat the
  run config's `01a1ef9f…`, anything else is refused, and a run that records none cannot stage
  at all — so the saved `preprocessor_config.json` cannot describe preprocessing the checkpoint
  never trained against. The frozen `(1,250,160) -> (1,125,43)` contract is checked on the
  loaded model before anything is written, and the stage is atomic: written to `<dir>.tmp` and
  renamed, so a hub failure leaves no half-written directory.
- Loads the way the published repository does: `AutoConfig.from_pretrained(dir,
  trust_remote_code=True)` resolves the vendored `Wav2Vec2BertForMultilevelCTC` through the
  same `auto_map`, `AutoModel` + `AutoFeatureExtractor` load it, and a forward on a fixed
  random `(1,250,160)` window is **bit-identical** to the checkpoint's EMA student
  (max abs diff `0.0`). `model.safetensors` holds 772 tensors, every one equal to the
  checkpoint's averaged weights.
- Re-staged from the tool after self-check r1: eight of the ten files are byte-identical to the
  earlier hand-staged directory (`model.safetensors`, `config.json`, both vendored modules,
  `phoneme_vocab.json`, `preprocessor_config.json`, `LICENSE`, `.gitattributes`); only
  `README.md` (the scorer citation) and `NOTICE.md` (the tlog attribution) changed. The same
  tool with `--profile agpl` reproduces the published repository's `README.md`, `NOTICE.md` and
  `LICENSE` byte for byte, so the AGPL path has not drifted either.
- The card states the identity this weights file cannot: step 10000, averaged weights,
  checkpoint sha256 `1193e211…24ed4a9`, `aug94-mixture-v2`, tlog + Tadabur.

### Comparison against the previous publication

Both students on the same 300 `gate_eval` dev clips, one protocol, 24,390 teacher phonemes,
produced by the committed scorer (`training.hub_agreement`, whose windowing and streamed
decode `verify_student_export` also uses):

```bash
python3 -m training.hub_agreement --eval-set tadabur/gate_eval --num-clips 300 \
  --model mini=<published mini dir> --model mini-v2=<this stage>
```

| model | vs teacher | 95% CI (reciter-clustered) | exact | drift vs mini |
| --- | --- | --- | --- | --- |
| published `sysofwan/hifzguide-muaalem-mini` | 94.707% | [94.177, 95.159] | 16.33% | 0.000% |
| this stage | 94.690% | [94.197, 95.117] | 18.67% | 2.370% |

2.37% of characters differ between the two decodes. They tie on teacher agreement, which is
what two imitations of one teacher should do; the app-visible change is against C95
(−59.2 false flags per 1,000 graded words), and C95 is a **different checkpoint** from the
published mini — all 772 tensors differ. The stage's 94.690% equals the export
verification's PyTorch fp32 row, so the hub load path reproduces the training-time forward.

- Licence: `LICENSE` is the CC BY-NC 4.0 legal code; the card declares
  `license: cc-by-nc-4.0`, `base_model: obadx/muaalem-model-v3_2`. `NOTICE.md` carries the
  vendored MIT notice, the lineage (Muaalem, w2v-bert-2.0), and the Tadabur, tlog and
  Freesound attributions with citations.
- Repository name, visibility and publishing are the owner's; nothing was pushed.

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
# staged HF weights are the averaged ones (772 tensors) and load through trust_remote_code
python3 -c "from safetensors.torch import load_file; print(len(load_file('model.safetensors')))"
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('.', trust_remote_code=True)"
# the publisher's tests: trust_remote_code load, EMA selection, teacher pin, atomic stage
scripts/test.sh training/test_publish_hf.py
# the committed scorer behind the card's comparison table
scripts/test.sh training/test_hub_agreement.py
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
