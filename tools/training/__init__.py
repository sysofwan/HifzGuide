"""Muaalem phoneme-head fine-tuning (LoRA) and evaluation.

LoRA fine-tune of the Muaalem phoneme head on the filtered Tadabur subset,
plus the two-sided evaluation harness. See ADR-0001 and issues #7-#11.

Also owns the waqf-head distillation soft labels (``waqf_distill``): the 20 ms
Recitation VAD teacher pooled 2:1 to Muaalem's 40 ms lattice, per ADR-0004; and the
waqf frame-classification head + detached joint loss (``waqf_head``) that rides that
lattice with a stop-gradient backbone.

Runs on Linux + CUDA (see tools/environment.yml).
"""
