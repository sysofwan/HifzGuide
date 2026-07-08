"""Tadabur quality-filtering pipeline.

Filters the Tadabur dataset down to a training-ready subset by scoring each
clip's decoded phonemes against the quran-transcript reference with a faithful
Python port of Muraja's `.balanced` scorer. See ADR-0001 and issues #2-#6.

Runs on Linux + CUDA (see tools/environment.yml).
"""
