import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

'''
d	Go to the next frame (+1).
a	Go to the previous frame (-1).
c	Jump forward 25 frames (1 second).
z	Jump backward 25 frames (1 second).
[	Mark the start of a presence interval (current frame).
]	Mark the end of a presence interval (current frame, inclusive) and save the interval.
x	Delete the presence interval containing the current frame.
i	Mark or unmark the current frame as ignore (bad frame).
s	Save.
q	Save and exit.

for this code is mark the presence of the employee,
and use the csv as a standard answer
'''

WINDOW = "annotate"
JUMP = 25
KEY_STEPS = {ord("d"): 1, ord("a"): -1, ord("c"): JUMP, ord("z"): -JUMP}
STATUS_COLORS = {"present": (0, 255, 0), "absent": (200, 200, 200), "ignore": (0, 0, 255)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

# compress to JPEG and save it, ez for mark the frame
def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames = []
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Loading") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
            progress.update(1)
    cap.release()
    return frames


def frame_status(index, labels):
    if index in labels["ignored"]:
        return "ignore"
    if any(start <= index <= end for start, end in labels["intervals"]):
        return "present"
    return "absent"

# if two section overlap it will merge it
def add_interval(intervals, start, end):
    merged = []
    for s, e in sorted(intervals + [(min(start, end), max(start, end))]):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged

# if csv ady exist it will continue witht he previous progress
def load_labels(path):
    labels = {"intervals": [], "ignored": set(), "points": {}}
    if not Path(path).exists():
        return labels
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    run_start, last_present = None, None
    for row in rows:
        index = int(row["frame"])
        if row["x"] and row["y"]:
            labels["points"][index] = (int(row["x"]), int(row["y"]))
        if row["status"] == "ignore":
            # Ignored frames don't break a present run: their presence is unknown, not absent.
            labels["ignored"].add(index)
        elif row["status"] == "present":
            run_start = index if run_start is None else run_start
            last_present = index
        elif run_start is not None:
            labels["intervals"].append((run_start, last_present))
            run_start = None
    if run_start is not None:
        labels["intervals"].append((run_start, last_present))
    print(f"Resumed from {path}: {len(labels['intervals'])} intervals, "
          f"{len(labels['points'])} points, {len(labels['ignored'])} ignored frames")
    return labels


def save_labels(path, labels, frame_count):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "status", "x", "y"])
        for index in range(frame_count):
            x, y = labels["points"].get(index, ("", ""))
            writer.writerow([index, frame_status(index, labels), x, y])
    print(f"Saved {path}")
    warn_points_outside_presence(labels)


def warn_points_outside_presence(labels):
    stray = sorted(i for i in labels["points"] if frame_status(i, labels) != "present")
    if stray:
        details = ", ".join(f"{i} ({frame_status(i, labels)})" for i in stray)
        print(f"Warning: {len(stray)} frame(s) have a point but are not present: {details}")


def previous_point(index, points):
    earlier = [i for i in points if i < index]
    return points[max(earlier)] if earlier else None


def render(frames, index, labels, pending_start):
    image = cv2.imdecode(frames[index], cv2.IMREAD_COLOR)
    faint = previous_point(index, labels["points"])
    if faint is not None:
        overlay = image.copy()
        cv2.circle(overlay, faint, 6, (0, 255, 255), -1)
        cv2.circle(overlay, faint, 7, (255, 255, 255), 1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    if index in labels["points"]:
        cv2.circle(image, labels["points"][index], 6, (0, 0, 255), -1)
        cv2.circle(image, labels["points"][index], 7, (255, 255, 255), 1)

    status = frame_status(index, labels)
    lines = [(f"frame {index} / {len(frames) - 1}", (255, 255, 255)),
             (status.upper(), STATUS_COLORS[status])]
    if pending_start is not None:
        lines.append((f"start: {pending_start}", (0, 255, 255)))
    for row, (text, color) in enumerate(lines):
        origin = (10, 30 + row * 30)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return image

# if mouse click these the info returned
def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["labels"]["points"][state["index"]] = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["labels"]["points"].pop(state["index"], None)

# handle the key u press from keyboard
def handle_key(key, state, frame_count):
    labels, index = state["labels"], state["index"]
    if key in KEY_STEPS:
        state["index"] = min(max(index + KEY_STEPS[key], 0), frame_count - 1)
    elif key == ord("["):
        state["pending_start"] = index
    elif key == ord("]"):
        if state["pending_start"] is None:
            print("No interval start set; press [ first.")
        else:
            labels["intervals"] = add_interval(labels["intervals"], state["pending_start"], index)
            state["pending_start"] = None
    elif key == ord("x"):
        labels["intervals"] = [(s, e) for s, e in labels["intervals"] if not s <= index <= e]
    elif key == ord("i"):
        labels["ignored"] ^= {index}
    elif key == ord("s"):
        save_labels(state["out"], labels, frame_count)
    elif key == ord("q"):
        return False
    return True

def main():
    args = parse_args()
    frames = load_frames(args.video)
    state = {"index": 0, "pending_start": None, "labels": load_labels(args.out), "out": args.out}

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    running = True
    while running:
        cv2.imshow(WINDOW, render(frames, state["index"], state["labels"], state["pending_start"]))
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        running = handle_key(key, state, len(frames))

    save_labels(args.out, state["labels"], len(frames))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
