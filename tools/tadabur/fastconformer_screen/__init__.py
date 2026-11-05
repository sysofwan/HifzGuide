"""#267's screen of NVIDIA's Arabic FastConformer as a fourth mining detector.

The screen is pre-registered on #267: a detector only (never a label, reference or training
signal), decoded on CPU in its own environment, scored with one letter-level rule on every
column, and read by a frozen GO / PRIOR / STOP rule. The data and the code live side by side,
as in :mod:`tadabur.tlog_mining`.

Modules:

``mapping``
    Step 2, text only: the Imlaei spelling of a segment, the edit-distance alignment a model's
    undiacritised text is read through, and the site/edits location (:func:`locate_site`,
    :func:`locate_edit`).
``outcomes``
    The four letter-level outcomes (heard / prescribed / other / none), the same rule for the
    teacher, C95, the round-2 EMA and FastConformer, and unique catches.
``sets``
    Which sets may be decoded (ship-a, edits, qdat and nothing else), the G6 removal, the
    teacher's letter-level bars, and the dated erratum the decode refuses without.
``gates``
    Steps 1-2's refusals: the pinned model identity, the mapping gate, the ONNX parity gate and
    the card's CTC-WER sanity line.
``verdict``
    The frozen GO / PRIOR / STOP rule and its four rates.
``decode``
    The decode driver: CPU by default, PAUSE and the RAM floor honoured at batch boundaries, and
    any GPU batch under #263's non-blocking yield helper.
"""
