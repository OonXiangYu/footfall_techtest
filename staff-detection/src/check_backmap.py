import math

import cv2
import numpy as np
from ultralytics import YOLO

from eval_detection import (ANGLE_STEPS, DEFAULT_CONF, DEFAULT_IMGSZ, ROOT, detect_view,
                            fallback_circle, find_fisheye_circle, load_gt_points, read_frames,
                            to_original_point, to_panorama_point)

OUT_DIR = ROOT / "outputs" / "backmap"
SAMPLE_COUNT = 8

# check the acc of the point on the unwarp img by reverse it to ori img
def check_round_trip(gt_points, circle):
    worst = 0.0
    for x, y in gt_points.values():
        pano = to_panorama_point(x, y, *circle)
        if pano is None:
            continue
        for pano_x in (pano[0], pano[0] + ANGLE_STEPS):
            back = to_original_point(pano_x, pano[1], *circle)
            worst = max(worst, math.hypot(back[0] - x, back[1] - y))
    return worst

# check the staff box
def staff_box(point, boxes):
    containing = [b for b in boxes for px in (point[0], point[0] + ANGLE_STEPS)
                  if b[0] <= px <= b[2] and b[1] <= point[1] <= b[3]]
    if not containing:
        return None
    return min(containing, key=lambda b: math.hypot((b[0] + b[2]) / 2 - point[0], (b[1] + b[3]) / 2 - point[1]))

# convert the boxes on the unwarp img to the ori img
def box_outline(box, circle, steps=20):
    x1, y1, x2, y2 = box
    xs, ys = np.linspace(x1, x2, steps), np.linspace(y1, y2, steps)
    edge = ([(x, y1) for x in xs] + [(x2, y) for y in ys] +
            [(x, y2) for x in xs[::-1]] + [(x1, y) for y in ys[::-1]])
    return np.array([to_original_point(x, y, *circle) for x, y in edge], np.int32)


def draw(frame, index, gt, centre, outline, distance):
    image = frame.copy()
    cv2.polylines(image, [outline], True, (0, 255, 0), 2)
    cv2.line(image, gt, (round(centre[0]), round(centre[1])), (255, 255, 255), 1)
    cv2.drawMarker(image, gt, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.circle(image, (round(centre[0]), round(centre[1])), 7, (0, 255, 0), -1)
    text = f"frame {index}  GT (red) vs box centre (green): {distance:.0f} px"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    gt_points = load_gt_points(ROOT / "data" / "gt.csv")
    frames = read_frames(str(ROOT / "data" / "sample.mp4"), set(gt_points))
    first = frames[min(frames)]
    circle = find_fisheye_circle(first) or fallback_circle(first)

    print(f"Round trip over {len(gt_points)} GT points (incl. x + {ANGLE_STEPS}): "
          f"max error {check_round_trip(gt_points, circle):.2e} px")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "yolov8n.pt"))
    sample = sorted(gt_points)[::max(1, len(gt_points) // SAMPLE_COUNT)][:SAMPLE_COUNT]
    distances = []
    for index in sample:
        gt = gt_points[index]
        _, boxes, _ = detect_view(model, frames[index], circle, True, DEFAULT_IMGSZ, DEFAULT_CONF)
        box = staff_box(to_panorama_point(*gt, *circle), boxes)
        if box is None:
            print(f"frame {index:4d}: staff not detected")
            continue
        centre = to_original_point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, *circle)
        distance = math.hypot(centre[0] - gt[0], centre[1] - gt[1])
        distances.append(distance)
        print(f"frame {index:4d}: GT ({gt[0]}, {gt[1]})  box centre ({centre[0]:.0f}, {centre[1]:.0f})"
              f"  distance {distance:.1f} px")
        cv2.imwrite(str(OUT_DIR / f"frame_{index:04d}.jpg"),
                    draw(frames[index], index, gt, centre, box_outline(box, circle), distance))
    if distances:
        print(f"Mean distance {np.mean(distances):.1f} px, max {np.max(distances):.1f} px")
    print(f"Images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
