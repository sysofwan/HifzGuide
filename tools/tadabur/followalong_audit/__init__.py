"""The follow-along audit (epic #305): score a device-style session against the app's decoder.

Muraja's engine grades every update its transcriber hands it, but the device-format ``.log``
cuts each overlap to 30 Characters (``TranscribeLog.swift:43-46``), so a replay of that log is
not the full-overlap session the device runs. This package builds the three arms the audit
compares:

* **App** -- the full hops and overlaps a driver emits from Muraja's own offline transcriber,
  over the app's CoreML model (``tools/muraja_harness/transcribe_checks.swift``), replayed
  through the harness;
* **Oracle** -- the clip's reference phonemes placed on its phase A teacher word times
  (:mod:`tadabur.followalong_audit.oracle`), replayed through the same harness; and
* **P1b** -- the same app checks with each overlap clipped to 30 Characters, the gate that
  shows the ``.log`` loses a grade decision (:mod:`tadabur.followalong_audit.p1b`).

:mod:`tadabur.followalong_audit.tracer` runs the three and writes the per-word tables; the
seeded clip draw and its provenance live in :mod:`tadabur.followalong_audit.clips`, and the
driver's release assets are checked in :mod:`tadabur.followalong_audit.assets`.
"""
