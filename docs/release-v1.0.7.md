# v1.0.7: the round-3 candidate as the shipped model (draft release, staged)

**Status (2026-11-18):** staged, not published. The GitHub release `v1.0.7` exists as a
**draft** and the Hugging Face repository is staged in a local directory. Publishing both is
the owner's call.

Release notes headline: models.zip now carries the round-3 candidate's **averaged (EMA)**
weights, replacing `sysofwan/hifzguide-muaalem-mini` as the published
`MuaalemStudent_h448_6BIT.mlmodelc`. The previous *published* model is that Hub repository;
C95 is an internal checkpoint the round measured against and was never shipped. The graph is
byte-identical; only the weights change; the I/O contract, filename and asset set are
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
repository created. Ten files, the same set `sysofwan/hifzguide-muaalem-mini` publishes. The
card was revised afterwards, adding the word-flag row recorded below, and the upload
directory re-rendered beside it at
`/Users/sysofwan/repos/HifzGuide/.orch/tasks/mini-flags/hf-stage/hifzguide-muaalem-mini-v2/`:
the table below is that directory, and `README.md` is its only file that differs from the
first staging. Re-rendered by the same tool from the stage, not the checkpoint, so no weight,
code, licence or notice file was rewritten:

```bash
cd tools
python3 -m training.publish_hf --from-stage <the r3-release stage> \
  --repo-id sysofwan/hifzguide-muaalem-mini-v2 --out-dir <this task's hf-stage>/hifzguide-muaalem-mini-v2
```

`--from-stage` copies the stage file for file and re-renders `README.md`/`NOTICE.md` from the
weight file's own header; it refuses a stage whose `model.safetensors` is not the
one the card's results were measured on (`RELEASE_STAGE_SAFETENSORS_SHA256`, the
`7b760511…6d650d89` above), so the prose cannot be re-rendered onto other weights.

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
| NOTICE.md | 5,039 | `b1a0ff7953ec3ddc9b663c497b82c6805da4fd06b0c9115bea18aeaebccb026e` | rewritten: NC licence, MIT text, dataset credits, links to this record |
| README.md | 4,999 | `62c3045b3c47b2847310cea743430a3995148e040d22ca6b690983348153bfe7` | rewritten consumer card: quick start, contract, two comparisons, limits |

- The `cc-by-nc` profile is this release's card, not a reusable licence template: it is bound
  to checkpoint sha256 `1193e211…24ed4a9` **and** the ema weight set, and `publish_hf`
  refuses any other checkpoint or the live weights before writing a file. The card's
  comparison, training history and readout measured that artifact only, so choosing a licence
  cannot attach them to different weights (`tools/training/test_publish_hf.py` covers both
  refusals).
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
- The card is consumer-facing, per the owner-approved proposal: what the model does, how to
  load it, the I/O contract, two measured comparisons against the previously published
  `hifzguide-muaalem-mini` (base-model agreement and word flags), and the licence. It carries
  no experiment labels, checkpoint identifiers or hashes. The proposal's block is unchanged
  apart from the word-flag row and its paragraph, added once the published mini was decoded
  on the same presumed-correct recordings (598 words including tables, up from 465). Artifact
  provenance stays in this record, which the public `NOTICE.md` links: step 10000, averaged
  weights, checkpoint sha256 `1193e211…24ed4a9`, `aug94-mixture-v2`, tlog + Tadabur.
  `NOTICE.md` carries only readable upstream links, the MIT text and the dataset credits, with
  no revision or checksum identifiers.
- Licence: `LICENSE` is the CC BY-NC 4.0 legal code; the card declares
  `license: cc-by-nc-4.0`, `base_model: obadx/muaalem-model-v3_2`. `NOTICE.md` carries the
  vendored MIT notice, the lineage (Muaalem, w2v-bert-2.0), and the Tadabur, tlog and
  Freesound attributions with citations.
- Repository name, visibility and publishing are the owner's; nothing was pushed.

### Comparison against the previous publication

Both students on the same 300 `gate_eval` dev clips, one protocol, 24,390 teacher phonemes,
produced by the committed scorer (`training.hub_agreement`, whose windowing and streamed
decode `verify_student_export` also uses, and which now runs the same
`decode_evalset.check_provenance` guard before scoring):

```bash
huggingface-cli download sysofwan/hifzguide-muaalem-mini \
  --revision a2bebf7c2e9a74d9ff5c3269594132d7caea7696 --local-dir /tmp/hfmini
shasum -a 256 /tmp/hfmini/model.safetensors   # 2570fd37…84bb9, the LFS sha256 below
python3 -m training.hub_agreement --eval-set tadabur/gate_eval --num-clips 300 \
  --model mini=/tmp/hfmini --model mini-v2=<this stage>
```

The comparison model is the `mini` repository at Hub revision
`a2bebf7c2e9a74d9ff5c3269594132d7caea7696` (the branch head when the table was measured),
whose `model.safetensors` LFS sha256 is
`2570fd3780a8435a11eb20e336bac29ac6f4dedd7c367d39abc20627e6784bb9`. This note records that
revision and hash (the public NOTICE links the record instead), so moving `main` cannot
silently change what the table measured. The `mini` publication records no training-checkpoint
hash of its own (its card, `config.json` and commit history name none), so its immutable Hub
artifact is the identity behind the row; the original training checkpoint cannot be recovered
from the publication. It is **not** C95 — all 772 tensors differ.

| model | vs teacher | 95% CI (reciter-clustered) | exact | drift vs mini |
| --- | --- | --- | --- | --- |
| published `sysofwan/hifzguide-muaalem-mini` | 94.707% | [94.177, 95.159] | 16.33% | 0.000% |
| this stage | 94.690% | [94.197, 95.117] | 18.67% | 2.370% |

2.37% of characters differ between the two decodes. They tie on teacher agreement, which is
what two imitations of one teacher should do; against the published mini the measured
difference is in word flags (below). The round's gate change is against C95 (−59.2 false
flags per 1,000 graded words), an internal checkpoint that never shipped and a
**different checkpoint** from the published mini — all 772 tensors differ. The stage's 94.690%
equals the export verification's PyTorch fp32 row, so the hub load path reproduces the
training-time forward.

### Word flags on the presumed-correct tlog population

The card's second row is measured, not inferred from agreement. The published `mini` (Hub
revision `a2bebf7c…ca7696`, `model.safetensors` sha256 `2570fd37…84bb9`) was decoded on the
round's pooled confirm read and graded by the same harness as the round's table; the v2 and
base columns reproduce round 3's committed numbers bit for bit (`docs/mini-flags-readout.md`,
population and bootstrap there). Over the 43,840 words all three models graded, the published
mini flags **167.7** words per 1,000 commonly graded words, this stage **78.1** and the teacher
**73.6**; the paired clustered 95% difference mini − this stage is **+89.599 [+80.161,
+101.759]**. Reporting only: no gate, verdict or pre-registration reads it.

The population is **presumed** correct, not adjudicated: `stratum_of` keeps clean,
non-straddling `label_source == both` rows, which establishes the ayah's identity, and
`docs/tlog-gating-amendment.md` §5 defines gating as clips the teacher *names* correctly — all
52 owner-verified real-mistake decodes still identify as `both`. A word flag on one of those is
a real catch, so **some of the row's flags may be real mistakes and the reduction is not an
adjudicated false-positive reduction**. No catch-loss audit was run for the published mini on
this population, so neither version's catch of real mistakes is measured there either; the card
says both.

The card prints those figures to one decimal (167.7 / 78.1 / 73.6 and 89.6 [80.2, 101.8]).

#### Source and dataset identities

`NOTICE.md` and the card link the upstream repositories; their immutable identities live
here, so the public text carries no revision or checksum identifiers:

- vendored `modeling`/`configuration` source: `obadx/quran-muaalem` commit
  `e9e692c87667ea6353486b2429bfcbaf32670cbe` (`src/quran_muaalem/modeling/`);
- `tarteel-ai/tlog` revision `9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51`;
- Freesound noise pool `obadx/freesound-commercial-50k-noise-only` revision
  `a5f48ab83b64ba1d35ee4ec363dae75d4e2c69d3`, with per-recording attribution in
  `tools/tadabur/augmentation/noise_pool.json`;
- teacher `obadx/muaalem-model-v3_2` revision `01a1ef9fbe40d144ef845101e89ff924aed3fef5`;
- checkpoint sha256 `1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9`;
- previous publication `sysofwan/hifzguide-muaalem-mini` at Hub revision
  `a2bebf7c2e9a74d9ff5c3269594132d7caea7696`, `model.safetensors` sha256
  `2570fd3780a8435a11eb20e336bac29ac6f4dedd7c367d39abc20627e6784bb9`.

## The draft release

`gh release create v1.0.7 --draft` on `sysofwan/HifzGuide`, with the five assets plus
`manifest.json`. The tag points at the repository's default-branch commit at draft time
(`bac196470ccf3a9d2f343449a6ca89b4e44a40e5`); the release stays a draft until the owner
publishes it.

Created and checked: `isDraft: true`, target commit `bac1964`, and the six assets uploaded with
the staged names and byte counts — `manifest.json` 936, `mel_filters.bin` 82,240, `window.bin`
1,600, `quran.db` 54,906,880, `models.zip` 86,034,195, `fonts.zip` 136,608,944. The uploaded
`manifest.json`, downloaded again, is byte-identical to the staged file (sha256 `529030de…`).
GitHub creates no tag ref for a draft, so `v1.0.7` becomes a real tag only when the owner
publishes.

The notes were rewritten after the owner's review: title
`v1.0.7 — models.zip ships updated student weights (EMA)`, and the body names the previous
**published** model (`hifzguide-muaalem-mini`) as the comparison baseline instead of C95. It
carries the export verification, that same-population 94.69% / 94.71% tie, the drift, and the
limit that these numbers do not establish better error detection; experiment labels, gate
names, issue numbers and checkpoint hashes are gone. The manifest checksum stays for download
verification, following v1.0.6's notes, which carry the same line. The full measurements and
disclosures below remain in this technical record and in
`docs/consonant-round3-readout.md`.

## Reproducing the checks

```bash
# asset checksums
python3 -c "import hashlib,json;m=json.load(open('manifest.json'));print(all(hashlib.sha256(open(a['filename'],'rb').read()).hexdigest()==a['sha256'] for a in m['assets']))"
# staged HF weights are the averaged ones (772 tensors) and load through trust_remote_code
python3 -c "from safetensors.torch import load_file; print(len(load_file('model.safetensors')))"
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('.', trust_remote_code=True)"
# the publisher's tests: trust_remote_code load, EMA selection, teacher pin, atomic stage
scripts/test.sh training/test_publish_hf.py
# the committed scorer behind the comparison table
scripts/test.sh training/test_hub_agreement.py
```

## Disclosures retained in this technical record

- Round 3 has **no control arm**: candidate − C95 (−59.2 false flags per 1,000 graded words,
  95% [−67.8, −51.7] on 5,651 graded clips / 113 surah clusters) cannot separate "more tlog
  data" from "10,000 more steps". Round 2 measured that separation once.
- Catch loss: 16 of 92 decisive removals were real mistakes, 17.4% (Wilson 95%
  [11.0%, 26.4%]) — the point estimate passes the ≤ 20% bar, the upper bound crosses it.
- Tashkeel is flat (−0.03 per 1,000, upper bound +1.38 against a +3.5 guard), and that guard
  vetoes gross regressions only.
- Single-chunk ANE residency on the device is still unproven; the 99 MB budget is
  demonstrated, not specified.
