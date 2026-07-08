"""Vendored Muaalem model classes (``obadx/quran-muaalem``).

The ``Wav2Vec2BertForMultilevelCTC`` model + config are copied **verbatim** from
``obadx/quran-muaalem`` (``src/quran_muaalem/modeling/``) rather than pip-installed:
the upstream package pins ``torch>=2.7.0`` as a hard dependency, and installing it
would let pip silently replace this environment's Blackwell/``sm_120`` ``cu128``
torch build with a CPU wheel that cannot run on the GPU (see
``tools/requirements-train.txt``). The model repo ships no ``modeling_*.py`` and no
``auto_map``, so ``trust_remote_code=True`` has nothing to load either.

See ``VENDORED.md`` for the exact upstream commit these files were taken from; to
update, re-copy the two files from that path and re-pin the commit.
"""

from .configuration_multi_level_ctc import Wav2Vec2BertForMultilevelCTCConfig
from .modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC

__all__ = [
    "Wav2Vec2BertForMultilevelCTC",
    "Wav2Vec2BertForMultilevelCTCConfig",
]
