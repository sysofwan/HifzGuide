"""Reading the non-professional corpora at pinned revisions (#156).

Four Hub datasets of app-user and amateur recitation, verified 2026-10-15. Each clip is one
recording; its item is the run of canonical (quran-transcript Uthmani) words it recites:

* **Tarteel v1** (:data:`TARTEEL_V1`, ``greentechapps/user_recited_verses_full``, CC BY 4.0):
  Tarteel v1 app users, every clip kept by the publisher's filter (both of two ASR models under
  CER 0.3, so biased towards easier clips). Its ``text`` is undiacritised, so a clip is mapped
  by its ``sura``/``ayah`` columns, never by its text, to the whole ayah. No reciter id.
* **RetaSy** (:data:`RETASY`, ``RetaSy/quranic_audio_dataset``, licence unknown: internal
  evaluation, aggregates only): non-Arab volunteers on phones, each clip labelled ``correct``
  or ``in_correct`` "with the diacritics, regardless of tajweed". Only Hafs clips with one of
  those two labels are read. ``Surah`` is a name (:data:`RETASY_SURAHS`; the prayers and
  adhan it also holds are no Quran) and ``Aya`` the text of an ayah or of a run of its words:
  it is placed on the surah's canonical words by :func:`locate` (a loose skeleton, as the text
  is written in several orthographies).
* **qdat** (:data:`QDAT`, ``obadx/qdat``, MIT): 160 amateur reciters, each recording one
  passage about ten times. Its card quotes 2:32's wording, qdat_bench's 5:109; the audio decides
  (:data:`QDAT_PASSAGE`, checked against both by ``training.nonpro_eval``). Its tajweed fields
  mark tajweed lapses, not consonant errors.
* **qdat_bench** (:data:`QDAT_BENCH`, ``obadx/qdat_bench``, licence unstated: internal
  evaluation, aggregates only): one qdat clip per reciter (159; the same bytes as the qdat clip of
  the same ``id``) with an expert-annotated ``phonetic_transcript`` (quran-transcript's script).
  It is never decoded again: its clips' decodes are qdat's.

Every read is pinned to the dataset's revision; a shard is downloaded into a cache directory
and its blob deleted once read (:func:`iter_rows`), so no bulk audio is kept. A clip is named by
its dataset's own unique id (:attr:`Clip.audio_filename`) and identified by the SHA-256 of the
audio bytes as published.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .qud_text import skeleton


@dataclass(frozen=True)
class Dataset:
    """One Hub dataset at its pinned revision, and whether per-clip outputs may be committed
    (``open``: its licence allows it; otherwise aggregates and clip ids only)."""

    name: str
    repo: str
    revision: str
    shards: tuple[str, ...]
    licence: str
    open: bool


TARTEEL_V1 = Dataset(
    "tarteel_v1", "greentechapps/user_recited_verses_full",
    "eae2dbb47ac763fb2c0cad00c1699f4b58521987",
    tuple(f"data/train-{i:05d}-of-00008.parquet" for i in range(8)), "CC BY 4.0", True)
RETASY = Dataset(
    "retasy", "RetaSy/quranic_audio_dataset", "b1fcc39cbc045f367bb07e39025a0e3aaeabf34f",
    tuple(f"data/train-{i:05d}-of-00003.parquet" for i in range(3)), "unknown", False)
QDAT = Dataset(
    "qdat", "obadx/qdat", "624d5827f9062bc0203170a00f44375b2adca1f1",
    tuple(f"data/train-{i:05d}-of-00002.parquet" for i in range(2)), "MIT", True)
QDAT_BENCH = Dataset(
    "qdat_bench", "obadx/qdat_bench", "53d7221da573b2b9e92d9542507564ae1b88fd30",
    ("data/train-00000-of-00001.parquet",), "unstated", False)
#: The corpora decoded, in report order (qdat_bench's clips are qdat's).
DECODED = (TARTEEL_V1, RETASY, QDAT)
DATASETS = {d.name: d for d in (*DECODED, QDAT_BENCH)}

#: RetaSy's ``Surah`` names that are Quran, and their surah; ``Ayat al-Kursi`` is 2:255's words.
RETASY_SURAHS = {
    "Al-Faatihah": 1, "Ayat al-Kursi": 2, "Al-Asr": 103, "Al-Humazah": 104, "Al-Kauthar": 108,
    "Al-Kafiroon": 109, "An-Nasr": 110, "Al-Masad": 111, "Al-Ikhlas": 112, "Al-Falaq": 113,
    "An-Nas": 114,
}
#: The RetaSy labels read: the clip-level human verdict, "with the diacritics, regardless of
#: the rules of Tajweed" (its card).
CORRECT, IN_CORRECT = "correct", "in_correct"
RETASY_LABELS = (CORRECT, IN_CORRECT)
#: qdat's passage: 5:109's canonical words 8-15, ``قَالُوا۟ لَا عِلْمَ لَنَآ إِنَّكَ أَنتَ عَلَّٰمُ
#: ٱلْغُيُوبِ`` (``training.nonpro_eval`` checks the audio against this and
#: :data:`QDAT_ALTERNATIVE`).
QDAT_PASSAGE = (5, 109, 8, 15)
#: The card's wording, 2:32 whole.
QDAT_ALTERNATIVE = (2, 32, 1, 10)
#: qdat's labels kept with each clip (its card: separate madd, the tight noon, concealment of
#: the sakin noon, ``target``, which the card's author did not understand either).
QDAT_TAJWEED = ("separate_tide", "the_tight_noon", "concealment", "target")


@dataclass(frozen=True)
class Clip:
    """One selected recording: its dataset and id, the canonical words it recites (surah, ayah,
    1-based ``word_from``..``word_to``), its reciter (``""`` where the dataset has none), the
    dataset labels kept with it, and its audio bytes when read with audio."""

    dataset: str
    audio_filename: str
    surah: int
    ayah: int
    word_from: int
    word_to: int
    reciter: str
    labels: dict = field(default_factory=dict)
    audio: bytes = b""

    @property
    def surah_ayah(self) -> str:
        return f"{self.surah}:{self.ayah}"


@dataclass(frozen=True)
class LeftOut:
    """A row not selected, and why."""

    audio_filename: str
    reason: str


def audio_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- placing RetaSy's text --------------------------------------------------------------------

#: Folds on top of :func:`tadabur.qud_text.skeleton`: the Farsi ya and kaf RetaSy writes, and ta
#: marbuta; alif and hamza are dropped, as the orthographies disagree on both.
_LOOSE = str.maketrans({"ی": "ي", "ک": "ك"})


def loose_skeleton(word: str) -> str:
    return re.sub("[اء]", "", skeleton(word.translate(_LOOSE))).replace("ة", "ه")


class Unplaced(ValueError):
    """A text that is no run of an ayah's canonical words."""


def locate(text: str, surah: int, ayat: Sequence[int],
           words_of: Callable[[int, int], tuple[list[str], list[str]]]) -> tuple[int, int, int]:
    """Where ``text`` sits among ``ayat`` of ``surah``: ``(ayah, word_from, word_to)``.

    Each word must match the canonical word's :func:`loose_skeleton` in its Uthmani or its
    imlaei spelling (``words_of(surah, ayah)``), in a contiguous run. A text that is a whole ayah
    is that ayah even where it is also a run inside another (``ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ``); two
    whole ayat of one text (109:3 and 109:5) are one item's words either way, and the first is
    taken. Raises :class:`Unplaced` otherwise."""
    target = [loose_skeleton(w) for w in text.split()]
    runs, whole = [], []
    for ayah in ayat:
        uthmani, imlaei = words_of(surah, ayah)
        keys = [loose_skeleton(w) for w in uthmani]
        alternatives = [loose_skeleton(w) for w in imlaei] if len(imlaei) == len(keys) else keys
        for i in range(len(keys) - len(target) + 1):
            if all(t in (keys[i + k], alternatives[i + k]) for k, t in enumerate(target)):
                run = (ayah, i + 1, i + len(target))
                runs.append(run)
                if len(target) == len(keys):
                    whole.append(run)
    if len(runs) == 1:
        return runs[0]
    if whole:
        return whole[0]
    raise Unplaced(f"{text!r} is {'no' if not runs else 'more than one'} run of surah {surah}'s "
                   f"words")


def canonical_words(surah: int, ayah: int) -> tuple[list[str], list[str]]:
    """An ayah's Uthmani and imlaei words, via quran-transcript."""
    from quran_transcript import Aya

    aya = Aya(surah, ayah).get()
    return list(aya.uthmani_words), list(aya.imlaey_words)


def ayat_in(surah: int) -> int:
    from quran_transcript import Aya

    return Aya(surah, 1).get().num_ayat_in_sura


# --- selection --------------------------------------------------------------------------------


def select(dataset: Dataset, record: dict) -> Clip | LeftOut:
    """One parquet row (its audio's bytes included or not) as a :class:`Clip`, or why not."""
    audio = record.get("audio") or {}
    if dataset is TARTEEL_V1:
        name = audio["path"]
        words, _ = canonical_words(record["sura"], record["ayah"])
        return Clip(dataset.name, name, record["sura"], record["ayah"], 1, len(words), "",
                    {"cer_nemo": record["cer_nemo"], "cer_wav": record["cer_wav"],
                     "completeness": record["completeness"]}, audio.get("bytes") or b"")
    if dataset is RETASY:
        name = audio["path"]
        if (record["reciter_qiraah"] or "").lower() != "hafs":
            return LeftOut(name, "not hafs")
        if record["final_label"] not in RETASY_LABELS:
            return LeftOut(name, f"label {record['final_label']}")
        surah = RETASY_SURAHS.get(record["Surah"])
        if surah is None:
            return LeftOut(name, f"no Quran: {record['Surah']}")
        ayat = [255] if surah == 2 else range(1, ayat_in(surah) + 1)
        try:
            ayah, word_from, word_to = locate(record["Aya"], surah, ayat, canonical_words)
        except Unplaced:
            return LeftOut(name, f"no ayah's words: {record['Surah']}")
        return Clip(dataset.name, name, surah, ayah, word_from, word_to, record["reciter_id"],
                    {"final_label": record["final_label"], "golden": record["golden"]},
                    audio.get("bytes") or b"")
    if dataset is QDAT:
        surah, ayah, word_from, word_to = QDAT_PASSAGE
        return Clip(dataset.name, record["id"], surah, ayah, word_from, word_to,
                    record["original_id"].split("_")[0],
                    {k: int(record[k]) for k in QDAT_TAJWEED}, audio.get("bytes") or b"")
    raise ValueError(f"{dataset.name} is not decoded")


def iter_rows(dataset: Dataset, shard: str, cache_dir: Path,
              batch_rows: int = 64) -> Iterator[dict]:
    """A shard's rows with their audio, downloaded into ``cache_dir`` at the dataset's revision;
    its cached blob is deleted once read."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from .shard_reader import remove_shard_blob

    path = hf_hub_download(dataset.repo, shard, repo_type="dataset", revision=dataset.revision,
                           cache_dir=str(cache_dir))
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
            yield from batch.to_pylist()
    finally:
        remove_shard_blob(path)


def metadata(dataset: Dataset) -> list[dict]:
    """Every row of a dataset with its audio's ``path`` but not its bytes, in shard order:
    column reads by HTTP range requests."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rows = []
    for shard in dataset.shards:
        with HfFileSystem().open(f"datasets/{dataset.repo}@{dataset.revision}/{shard}",
                                 block_size=1 << 16) as f:
            parquet = pq.ParquetFile(f)
            rows += parquet.read(columns=["audio.path", *(
                c for c in parquet.schema_arrow.names if c != "audio")]).to_pylist()
    return rows


def row_groups(dataset: Dataset) -> list[tuple[str, int]]:
    """Every ``(shard, row group)`` of a dataset (parquet footers only)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out = []
    for shard in dataset.shards:
        with HfFileSystem().open(f"datasets/{dataset.repo}@{dataset.revision}/{shard}") as f:
            out += [(shard, g) for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
    return out


def row_group(dataset: Dataset, shard: str, group: int) -> list[dict]:
    """One row group of a shard, audio included, by HTTP range requests (never the whole
    shard)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{dataset.repo}@{dataset.revision}/{shard}",
                             block_size=32 << 20) as f:
        return pq.ParquetFile(f).read_row_group(group).to_pylist()
