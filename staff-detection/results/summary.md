# Final run on the sample video

Semi-automatic mode with 44 confirmations, all whole-track (`examples/sample_confirmed.json`).
Scored against the hand labels in `data/gt.csv` (`eval/eval_run.py`).

## Position (the 55 frames with a labelled chest point, about one per second)

| | |
|---|---|
| hits (prediction within 0.10 of the fisheye radius = 48 px) | **50 of 55** (91%) |
| median distance to the labelled point | **15.6 px** |
| frames with a prediction | 51 of 55 |

## Presence (every labelled frame; the staff member is in all of them)

| | |
|---|---|
| frames marked present | 1260 of 1337 |
| precision / recall | 1.00 / 0.94 |

## Frame by frame (against the chest point interpolated between labelled frames)

| | |
|---|---|
| present frames within 0.10 R of the interpolated point | 752 of 1260 |
| frames on the wrong person (far from the point, another track on it) | 56 |
| frames where the point stands still while far from the GT | 6 |
| frames claimed by two staff tracks at once | 7 |
| jumps of more than 0.10 R between frames | 15 |
| present frames with no staff box (held or interpolated, at most 0.5 s) | 197 |

The interpolated point lags while the staff member walks, so the frame-by-frame distances are
pessimistic. `error_timeline.png` shows the distance per frame and `timeline.png` shows presence.

## Detection: original frame vs panorama (`eval/eval_detection.py`)

| view | labelled frames where a box contains the chest point | boxes / frame |
|---|---|---|
| original fisheye frame | 15 of 55 (27.3%) | 2.8 |
| unwarped panorama | 47 of 55 (85.5%) | 15.7 |
