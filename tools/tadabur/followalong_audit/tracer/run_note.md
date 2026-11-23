# S1 tracer run (#307)

20/20 clips produced both arms.

Selection: seed 7, 20 clips from 5,742 of 5,762 phase A gating rows; #248's exclusion left out 20 clips of 7 ayat (106:1, 106:2, 106:3, 106:4, 55:17, 90:10, 90:8).

The App arm ran Muraja's transcriber at 40 ms per inference; the Oracle arm fills the clip's reference phonemes onto the phase A teacher word times and is cut on the zero-latency Python schedule (decision 2). The arms also differ in their own audio and output gates and silence handling, so the preview:window ratios below are not a latency-only comparison and neither bounds the other.

App rows are the words Muraja's grade store kept (S0's `word_traces` population); the Oracle arm covers every word of the ayah.

| clip | ayah | wall s | run 2 s | app preview:window | oracle preview:window | aligned |
| --- | --- | --- | --- | --- | --- | --- |
| 107_4_2416882393.flac#0 | 107:4 | 1.2 | 0.4 | - | - | True |
| 1_2_4071256168.flac#0 | 1:2 | 0.6 | 0.6 | - | - | True |
| 3_193_undefined.flac#0 | 3:193 | 3.1 | 3.2 | 6.4 | 6.4 | True |
| 97_1_8623697117.flac#0 | 97:1 | 0.7 | 0.9 | - | - | True |
| 10_74_undefined.flac#0 | 10:74 | 4.9 | 4.7 | 5.8 | 5.8 | True |
| 112_4_4581723145.flac#0 | 112:4 | 0.4 | 0.3 | - | - | True |
| 66_1_4141565767.flac#0 | 66:1 | 4.4 | 4.1 | 6.3 | 7.0 | True |
| 112_4_4731121740.flac#0 | 112:4 | 0.5 | 0.6 | - | - | True |
| 38_77_3417292978.flac#0 | 38:77 | 1.2 | 1.0 | 22.0 | 23.0 | True |
| 78_38_8626214945.flac#0 | 78:38 | 1.5 | 1.7 | 7.8 | 7.8 | True |
| 41_25_2252359315.flac#0 | 41:25 | 4.2 | 4.4 | 5.8 | 5.8 | True |
| 1_7_6262501364.flac#0 | 1:7 | 2.4 | 2.5 | 7.6 | 6.0 | True |
| 67_5_7450392312.flac#0 | 67:5 | 2.7 | 2.5 | 6.9 | 6.9 | True |
| 1_3_3599349059.flac#0 | 1:3 | 0.2 | 0.3 | - | - | True |
| 3_111_2159625809.flac#0 | 3:111 | 2.1 | 2.1 | 18.0 | 7.0 | True |
| 78_29_6672421811.flac#0 | 78:29 | 0.6 | 0.6 | - | - | True |
| 89_4_3447928277.flac#0 | 89:4 | 0.6 | 0.5 | - | - | True |
| 67_5_2134960157.flac#0 | 67:5 | 2.7 | 2.8 | 6.5 | 7.6 | True |
| 18_12_6812142192.flac#0 | 18:12 | 1.2 | 1.1 | 10.5 | 10.5 | True |
| 110_2_5822422253.flac#0 | 110:2 | 0.6 | 0.6 | - | - | True |

P1b: pass — 8 changed word decision(s); 20/20 drawn clips replayed.
App tables hold the words the engine's grade store kept; 1 clip(s) graded words at a check but kept none (no window or hop advanced the reader):
- 1_3_3599349059.flac: checks graded words [1, 2], `final` kept none

Failures:
- none
