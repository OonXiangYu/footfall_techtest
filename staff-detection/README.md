# Staff detection in overhead fisheye video

Finds the staff member (the person wearing the name tag) in an overhead fisheye video and writes,
for every frame, whether they are present and where (`staff.csv`), plus an annotated video and a
presence timeline.

## Setup

Python 3.11, from the `staff-detection/` folder:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
```

The last line downloads the detector and pose weights into this folder. Commands below use
`.venv\Scripts\python.exe`; on macOS / Linux use `.venv/bin/python`.

## Semi-automatic mode

A single video is not enough data for the automatic tag detector to work across outfits, so in this
mode a person confirms the staff member once, on a review page, and everything else stays automatic.

1. **Run and review.** Detection, tracking and linking run as usual; then a review window shows the
   people most likely to be wearing the tag, 12 per page (chest crop, and the full frame with the
   person outlined):

   ```
   .venv\Scripts\python.exe src\staff_detect.py --video data\sample.mp4 --out outputs\run_review --tag review
   ```

   - Click a candidate to mark it as the staff member (click again to undo).
   - The staff member may change clothes: confirm one candidate per outfit.
   - `n` / `p`: next / previous page. `Enter`: done.

   The confirmations are saved to `outputs\run_review\confirmed.json`, and the run finishes with
   `staff.csv`, `annotated.mp4` and `timeline.png` in the same folder.

2. **Rerun without the review page**, e.g. after changing settings, reusing the confirmations:

   ```
   .venv\Scripts\python.exe src\staff_detect.py --video data\sample.mp4 --out outputs\run_review --tag review --confirmed outputs\run_review\confirmed.json
   ```

   Confirmations are stored by detection, so they stay valid when tracking or linking changes.

3. **Options.**
   - `--confirm-scope chain` (default): a confirmation marks the person's whole track (all the
     pieces the tracker linked together) as the staff member. `--confirm-scope window` keeps only
     the part within `--confirm-window` seconds (default 5) of the confirmed frame; use it if a
     track sometimes switches to another person.
   - `--expand-similarity 0.8` (default): after confirmation, other tracks are added as the staff
     member if their torso colours match a confirmed track this closely, they never overlap in
     time with a staff track, and the staff member could have walked between them.
     Higher is stricter.
   - `--no-expand`: only the confirmed tracks count as staff.

4. **Check against the hand labels** (for this sample video, which has `data\gt.csv`):

   ```
   .venv\Scripts\python.exe src\eval_pipeline.py outputs\run_review
   .venv\Scripts\python.exe src\eval_review.py outputs\run_review
   ```

   `eval_pipeline.py` scores `staff.csv`. `eval_review.py` simulates users with the labels: how
   many of the top candidates are the staff member; the results when every staff candidate on
   page 1 (or pages 1-2) is confirmed, and with one confirmation per outfit; for both confirm
   scopes, with and without expansion; and the purity of the tracks expansion adds.

### Results on the sample video

Simulated with the hand labels (`eval_review.py`), whole-track confirmation:

| | confirmations | position hits (of 55 labelled frames) | median error | frames with the staff marked | marked frames really the staff |
|---|---|---|---|---|---|
| confirm every staff candidate on page 1 | 6 | 9 | 44 px | 454 | 79% |
| confirm every staff candidate on pages 1-2 | 9 | 15 | 43 px | 718 | 84% |
| one confirmation per outfit, 2 clicks, + expansion | 2 | 11 | 14 px | 329 | 100% |
| oracle (hand tag labels on every 5th frame) | - | 21 | 20 px | 687 | - |

A hit is a predicted point within 0.10 of the fisheye radius (48 px) of the labelled chest point.
4 of the top 5 review candidates were the staff member (6 of the 12 on page 1). Confirming more
candidates covers more of the video, but some tracks switch to another person for part of their
length, so purity drops below 100%. The staff member is hard to find while seated (the tag is
not visible), so the seated stretch at the start is rarely among the candidates.
