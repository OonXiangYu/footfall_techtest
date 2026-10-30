# Staff detection in overhead fisheye video

Finds the staff member (the person wearing the name tag) in an overhead fisheye video and answers,
for every frame, whether they are present and where:

- `staff.csv`: `frame, present, x, y, chain` for every frame (x, y in original-image pixels);
- `staff_ranges.txt`: the frames where the staff member is present, as ranges, with the total,
  and the frames where they are not found (also printed at the end of the run);
- `trajectory.png`: their path over the whole video, coloured by time, with the gaps marked;
- `annotated.mp4`: the video with the staff member boxed and labelled `STAFF (x, y)`;
- `timeline.png`: presence over time.

**Not included:** the sample video and the task brief are company material and are not in this
repository. The hand labels (`data/*.csv`), the trained tag model, the confirmations and the
results below all come from that video. To reproduce them you need the video; without it you can
run the pipeline on your own footage (see below).

## Setup

Python 3.11, from the `staff-detection/` folder:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
```

The last line downloads the detector and pose weights into this folder. Commands below use
`.venv\Scripts\python.exe`; on macOS / Linux use `.venv/bin/python`.

## Run on your own video

The pipeline is built for a ceiling-mounted fisheye camera looking straight down, with one staff
member wearing a name tag. Any video OpenCV can read works:

```
.venv\Scripts\python.exe src\staff_detect.py --video path\to\your_video.mp4 --out outputs\my_run
```

Detection, tracking and linking run on their own (about 4 minutes for the 54 s sample video on
the development machine). Then the review page opens and you confirm the staff member among the ranked candidates (see
[Semi-automatic mode](#semi-automatic-mode)). The results go to `outputs\my_run`.

- The fisheye circle is found automatically. If none is found, the frame centre is used.
  `--unwarp auto` (default) checks the first frames and uses the panorama only if more people
  are detected on it, so ordinary (non-fisheye) footage also runs, though only fisheye footage
  was tested.
- The tag model was trained on the sample video only and has not been tested on other footage;
  its ranking may be weaker there, so you may need to look through more pages.
- The `eval\` scripts need hand labels in the format of `data\gt.csv` for your video. The
  pipeline itself does not need them.

## Reproduce the final result (needs the sample video)

Put the sample video at `data\sample.mp4`. The confirmations from the final review are in
`examples\sample_confirmed.json`, so the final run can be repeated without the review page:

```
.venv\Scripts\python.exe src\staff_detect.py --video data\sample.mp4 --out outputs\final --confirmed examples\sample_confirmed.json --no-review
```

This writes `staff.csv`, `staff_ranges.txt`, `trajectory.png`, `annotated.mp4`, `timeline.png`
and the caches into `outputs\final`.
`staff.csv` is identical to `results\staff.csv`. To reproduce this run and every evaluation number
with one command, see [Evaluation](#evaluation).

## How it works

1. **Detection.** The fisheye frame is unwarped into a panorama (one column per 0.25 degree,
   with the left edge repeated past the right edge so that people on the seam are seen whole).
   YOLOv8n then runs on the panorama (imgsz 1440, conf 0.25). On the original frame the detector
   finds the staff member on 27.3% of the labelled frames; on the panorama it finds them on 85.5%
   (`eval\eval_detection.py`).
2. **Tracking.** A point tracker with a tight gate (0.05 of the fisheye radius per frame)
   follows each person. It breaks tracks rather than swapping people.
3. **Linking.** Track pieces are joined into chains when one ends where another starts. A link
   needs a clear best candidate (ratio test). Torso colour rescues ambiguous links and joins a
   person who sat still while undetected.
4. **Staff identification (semi-automatic).** A tag model ranks the chains. A person confirms the
   staff member on a review page, and chains whose colours match a confirmed one are added.
5. **Output.** A path through the staff chains (one per frame, with a penalty for switching)
   gives the point and box per frame. Gaps of up to 0.5 s are interpolated.

## Semi-automatic mode

A single video is not enough data for the automatic tag detector to work across outfits, so in this
mode a person confirms the staff member once, on a review page, and everything else stays automatic.

1. **Run and review.** Detection, tracking and linking run as usual; then a review window shows the
   people most likely to be wearing the tag, 8 per page (chest crop, and the full frame with the
   person outlined). The window is scaled to fit the screen:

   ```
   .venv\Scripts\python.exe src\staff_detect.py --video data\sample.mp4 --out outputs\run_review
   ```

   - Click a candidate to see its whole track: 5 frames spread from its start to its end, each with
     the person outlined. Then `y` confirms the whole track, `w` confirms only the 5 s either side
     of the clicked frame (for a track that switches to someone else), `Esc` cancels.
   - Click a confirmed candidate (green) to remove it. A warning on a confirmed tile means it is in
     the same frames as another confirmed track; at most one of them can be the staff member.
   - The staff member may change clothes: confirm one candidate per outfit.
   - `n` / `p`: next / previous page. `Enter`: done. (The keys are also printed in the terminal.)
   - `--per-page 8` sets the candidates per page (4 per row); `--ui-height 700` caps the window
     height in pixels if the automatic fit does not suit your screen.

   The confirmations are saved to `outputs\run_review\confirmed.json`, and the run finishes with
   `staff.csv`, `staff_ranges.txt`, `trajectory.png`, `annotated.mp4` and `timeline.png` in the
   same folder.

2. **Continue a review** from earlier confirmations (they open already selected), or add
   `--no-review` to reuse them without the page, e.g. after changing settings:

   ```
   .venv\Scripts\python.exe src\staff_detect.py --video data\sample.mp4 --out outputs\run_review --confirmed outputs\run_review\confirmed.json
   ```

   Confirmations are stored by detection, with their scope (whole track or 5 s window), so they stay
   valid when tracking or linking changes.

3. **Options.**
   - `--confirm-scope chain` (default): the scope for confirmations that do not store their own:
     the person's whole track (all the pieces the tracker linked together), or with `window` only
     the part within `--confirm-window` seconds (default 5) of the confirmed frame. The review page
     stores the scope you choose (`y` / `w`) with each confirmation.
   - `--expand-similarity 0.8` (default): after confirmation, other tracks are added as the staff
     member if their torso colours match a confirmed track this closely, they never overlap in
     time with a staff track, and the staff member could have walked between them.
     Higher is stricter.
   - `--no-expand`: only the confirmed tracks count as staff.
   - `--debug`: draw every chain in `annotated.mp4`, not just the staff member.
   - `--tag oracle`: take the staff chains from the hand tag labels instead of a review (a
     reference for the evaluation).

## Results on the sample video

The final run used 44 confirmations, all whole-track. They are listed in
`examples\sample_confirmed.json`. More detail is in `results\summary.md`.

| | |
|---|---|
| position hits (within 0.10 R = 48 px of the labelled chest point) | **50 of 55** labelled frames |
| median distance on the labelled frames | **15.6 px** |
| frames marked present | 1260 of 1337 (the staff member is in every frame; precision 1.00, recall 0.94) |
| frames on the wrong person (far from the GT, another chain on it) | 56 |
| frames where the point stands still while far from the GT | 6 |
| frames claimed by two staff chains at once | 7 |

The per-frame figures use the hand-labelled chest point, interpolated between the labelled
frames (about one per second).

- `results\staff.csv`: `frame, present, x, y, chain` for every frame (x, y in original pixels).
- `results\staff_ranges.txt`: present in frames 0-245, 268-818, 841-1282 and 1311-1331, which
  is 1260 of 1341 frames (94.0%). The staff member is not found in four gaps of 0.4-1.1 s.
- `results\trajectory.png`: the path over the whole video on a blank background with the fisheye
  circle, coloured by time. The gaps are dotted and numbered.
- `results\timeline.png`: presence over time.
- `results\error_timeline.png`: distance to the labelled point per frame.
- `results\summary.md`: the numbers in one place.

The annotated video and the figures showing video frames are not published, because they show
the people in the sample video. The commands above recreate them from the video.

Simulated users (`eval\eval_review.py`, whole-track confirmation) show how much a short review gives:

| | confirmations | position hits (of 55 labelled frames) | median error | frames with the staff marked | marked frames really the staff |
|---|---|---|---|---|---|
| confirm every staff candidate on page 1 (8 shown) | 5 | 11 | 12 px | 310 | 100% |
| confirm every staff candidate on pages 1-2 (16 shown) | 6 | 11 | 14 px | 399 | 79% |
| one confirmation per outfit, 2 clicks, + expansion | 2 | 11 | 12 px | 285 | 100% |

A hit is a predicted point within 0.10 of the fisheye radius (48 px) of the labelled chest point.
4 of the top 5 review candidates were the staff member (5 of the 8 on page 1). The tag is visible
on only about 14% of the staff frames, so a few clicks cover only part of the video; the final run
needed 44.

## Evaluation

The evaluation compares against the hand labels in `data\`, so it needs the sample video at
`data\sample.mp4`. One command reproduces every number in `results\summary.md`:

```
.venv\Scripts\python.exe eval\reproduce.py
```

It runs the pipeline with the final confirmations (`--no-review`), then the five scripts below.
Each script's output goes to `outputs\reproduce\logs\`. At the end it prints each number next to
the value in `results\summary.md`. It also checks that `staff.csv` and `decisions.csv` match
`results\staff.csv` and `results\decisions.csv`, and exits with an error if anything differs. Add
`--keep-caches` on later runs to reuse the detections and appearance caches.

| step | question it answers | time |
|---|---|---|
| `src\staff_detect.py` | the final run: where is the staff member in each frame? | ~4 min (13 s with caches) |
| `eval\eval_detection.py` | does the detector find the staff member better on the original frame or on the panorama? | ~17 s |
| `eval\eval_linking.py <run>` | which tracker gate and linking settings keep one person per track (purity)? | ~14 s |
| `eval\eval_run.py <run>` | how close is the output to the labels: hits, presence, per-track and frame-by-frame errors? | ~1 s |
| `eval\eval_review.py <run>` | how much does a short review give: candidate ranking, simulated users, expansion threshold? | ~30 s (6 s with caches) |
| `eval\decisions.py --run <run>` | what does each key decision buy, against its alternative with everything else unchanged? | ~30 s |

### Key decisions

`eval\decisions.py` runs each decision as an ablation on the sample video: the current default
against the alternative, with everything else unchanged. It writes one table to
`results\decisions.md` and `results\decisions.csv`, and one bar chart per decision to
`results\decisions.png`:

1. detection view: original frame vs panorama (staff hits, boxes per frame);
2. detection confidence: 0.10 vs 0.25 (staff hits, boxes per frame);
3. tracker gate: 0.20 vs 0.05 of the radius (track purity, staff track changes);
4. linking ratio test: off vs 0.5 (chain purity, staff chain changes);
5. duplicate-box removal: off vs on (chain purity, staff chain changes, and the chain covering
   the seated stretch, frames 0-347);
6. the final result with `examples\sample_confirmed.json` (hits, median error);
7. the oracle reference, `--tag oracle` with `data\tag_gt.csv`: every track with at least 2
   hand-labelled tag sightings counts as staff (hits, median error).

Decisions 1-2 run the detector on the 55 labelled frames. Decisions 3-7 reuse the detection and
appearance caches of a run folder (`--run`, default `outputs\reproduce`) and pass the alternative
as a `staff_detect.py` flag: `--max-dist 0.20`, `--link-ratio 0`, `--keep-duplicates`.

In total it takes about 6 minutes from scratch and about a minute with `--keep-caches`, measured
on the development machine. Each script also runs on its own, for example
`.venv\Scripts\python.exe eval\eval_run.py outputs\final`.

The tag model's per-frame AUC on the held-out half of the video is 0.86 and 0.74 (tag visible vs
other people). That is good enough to rank review candidates, not to pick the staff member
automatically.

## Repository layout

```
src/        the pipeline: staff_detect.py (entry point), fisheye, detections, tracker, linking,
            appearance, pose, crops, tag_model, review, labels (paths and label readers)
eval/       evaluation scripts (above)
data/       hand labels: gt.csv (chest point ~1 per second), tag_gt.csv / tag_gt_dense.csv
            (tag visibility), tag_pos.csv (tag centres) for the sample video
models/     tag_classifier.npz (the tag model that ranks review candidates)
examples/   sample_confirmed.json (the final review's confirmations)
results/    staff.csv, staff_ranges.txt, plots (timeline, trajectory, error) and summary.md of the
            final run; decisions.md / .csv / .png
outputs/    run folders (not in git)
```

The tools used to make the hand labels and to train the tag model are not included.

The earlier experiments are kept in git history under the tag `experiments-archive`
(`git checkout experiments-archive`). They include the fully automatic tag detection (`--tag auto`),
template matching, the ByteTrack comparison and the detector settings sweep.
