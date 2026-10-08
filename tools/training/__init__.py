"""Muaalem phoneme-head fine-tuning (LoRA), distillation and evaluation.

LoRA fine-tune of the Muaalem phoneme head on the filtered Tadabur subset,
plus the two-sided evaluation harness. See ADR-0001 and issues #7-#11.

The fixed 5 s window grid every whole-clip label builder cuts — the window contract,
the clip-relative recitation grid and the 20 ms → 40 ms lattice geometry — lives in
``windowing``. The whole-clip phoneme-only fine-tune lives in ``whole_clip_phoneme`` —
LoRA on the phoneme head over fixed windows, with a 16 GB memory preflight — fed by the
windowed CTC collator in ``windowed_batch``.

Runs on Linux + CUDA (see tools/environment.yml).
"""
