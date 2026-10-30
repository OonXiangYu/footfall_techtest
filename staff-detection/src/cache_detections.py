"""Run person detection once over the whole video and cache it to outputs/detections.npz."""

import argparse
import time

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from eval_detection import (DEFAULT_CONF, DEFAULT_IMGSZ, ROOT, fallback_circle, find_fisheye_circle,
                            pad_seam, to_original_point, unwarp)
from eval_tracking import merge_seam_tracks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "detections.npz"))
    return parser.parse_args()


# panorama boxes (seam-merged) and their confidences for one frame
def detect_frame(model, frame, circle):
    pano = unwarp(frame, *circle)
    result = model(pad_seam(pano), classes=[0], imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    # row indices stand in for track IDs, so we learn which original rows survived the merge
    boxes, rows = merge_seam_tracks(boxes, list(range(len(boxes))), pano.shape[1])
    return boxes, confs[rows]


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    model = YOLO(str(ROOT / "yolov8n.pt"))

    start = time.perf_counter()
    circle, frame_ids, points, pano_boxes, confs = None, [], [], [], []
    index = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Detecting") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if circle is None:
                circle = find_fisheye_circle(frame) or fallback_circle(frame)
            boxes, box_confs = detect_frame(model, frame, circle)
            for box, conf in zip(boxes, box_confs):
                frame_ids.append(index)
                points.append(to_original_point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, *circle))
                pano_boxes.append(box)
                confs.append(conf)
            index += 1
            progress.update(1)
    cap.release()
    elapsed = time.perf_counter() - start

    np.savez(args.out,
             frame=np.array(frame_ids, int),
             point=np.array(points, float).reshape(-1, 2),
             pano_box=np.array(pano_boxes, float).reshape(-1, 4),
             conf=np.array(confs, float),
             frame_count=index, fps=fps, center=np.array(circle[0]), radius=circle[1])
    print(f"{len(frame_ids)} boxes over {index} frames ({len(frame_ids) / max(index, 1):.1f} per frame)")
    print(f"Total time {elapsed:.1f} s ({1000 * elapsed / max(index, 1):.0f} ms per frame)")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
