"""Muaalem phoneme-head fine-tuning (LoRA) and evaluation.

LoRA fine-tune of the Muaalem phoneme head on the filtered Tadabur subset,
plus the two-sided evaluation harness. See ADR-0001 and issues #7-#11.

Also owns the waqf-head distillation soft labels (``waqf_distill``): the 20 ms
Recitation VAD teacher pooled 2:1 to Muaalem's 40 ms lattice, per ADR-0004; and the
waqf frame-classification head + detached joint loss (``waqf_head``) that rides that
lattice with a stop-gradient backbone.

The whole-clip phoneme-only fine-tune (ADR-0004's ablation rung (2)) lives in
``whole_clip_phoneme`` — LoRA on the phoneme head over fixed windows, with a 16 GB memory
preflight — fed by the phoneme-only windowed CTC collator in ``windowed_batch``. The joint
detached-waqf run (rung (3)) is ``joint_waqf``, and ``ablation_ladder`` ties the two to the
segmented rung (1): the deterministic (2)↔(3) phoneme identity check, the should-accept /
should-reject deltas across all three rungs, and the LoRA-native lever fired when the
whole-clip move regresses should-reject.

Runs on Linux + CUDA (see tools/environment.yml).
"""
