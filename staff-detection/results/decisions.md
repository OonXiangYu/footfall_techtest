# Key decisions: ablations on the sample video

Each row changes one setting from the current default (bold) to the alternative, everything else
unchanged. Made by `eval/decisions.py`; staff hits and purity are measured on the 55 labelled frames.

| # | decision: alternative -> default | metric | alternative | default |
|---|---|---|---|---|
| 1 | detection view: original frame -> **panorama** | staff hits of 55 | 15 of 55 (27.3%) | **47 of 55 (85.5%)** |
| 1 | detection view: original frame -> **panorama** | boxes / frame | 2.8 | **15.7** |
| 2 | detection conf: 0.10 -> **0.25** | staff hits of 55 | 48 of 55 (87.3%) | **47 of 55 (85.5%)** |
| 2 | detection conf: 0.10 -> **0.25** | boxes / frame | 37.9 | **15.7** |
| 3 | tracker gate: 0.20 R -> **0.05 R** | track purity | 34.6% (47/136) | **97.9% (47/48)** |
| 3 | tracker gate: 0.20 R -> **0.05 R** | staff track changes | 19 | **28** |
| 4 | linking ratio test: off -> **0.5** | chain purity | 39.5% (47/119) | **97.9% (47/48)** |
| 4 | linking ratio test: off -> **0.5** | staff chain changes | 20 | **25** |
| 5 | duplicate-box removal: off -> **on** | chain purity | 97.9% (47/48) | **97.9% (47/48)** |
| 5 | duplicate-box removal: off -> **on** | staff chain changes | 29 | **25** |
| 5 | duplicate-box removal: off -> **on** | chain on the seated stretch (frames 0-347), frames | 229 (0-228) | **348 (0-347)** |
| 6 | final result: sample_confirmed.json | hits of 55 | - | **50 of 55 (90.9%)** |
| 6 | final result: sample_confirmed.json | median error (px) | - | **15.6** |
| 7 | oracle (hand tag labels): tag_gt.csv | hits of 55 | - | **17 of 55 (30.9%)** |
| 7 | oracle (hand tag labels): tag_gt.csv | median error (px) | - | **16.1** |
