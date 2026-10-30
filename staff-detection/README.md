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

   `eval_pipeline.py` scores `staff.csv`. `eval_review.py` simulates a user with the labels: how
   many of the top candidates are the staff member, the results after 1, 2 and 3 confirmations,
   and the purity of the tracks added by the expansion at several thresholds.

### Results on the sample video

| | position hits (of 55 labelled frames) | median error |
|---|---|---|
| semi-automatic, 2 confirmations + expansion | 11 | 13.9 px |
| oracle (hand tag labels on every 5th frame) | 21 | 19.5 px |

A hit is a predicted point within 0.10 of the fisheye radius (48 px) of the labelled chest point.
4 of the top 5 review candidates were the staff member. Coverage is limited by how far the tracks
can be followed: each confirmation covers one track (often 1–10 s), and expansion only adds tracks
whose appearance clearly matches.
