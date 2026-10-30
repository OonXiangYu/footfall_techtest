import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SAMPLE_EVERY = 25
THUMB_WIDTH = 320
SHEET_COLS = 6
SHEET_ROWS = 5
LOW_BRIGHTNESS_K = 3.0

# argument for video path
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="path to the video file")
    return parser.parse_args()

# display
def print_metadata(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolution:            {width}x{height}")
    print(f"FPS:                   {fps:.3f}")
    print(f"Reported frame count:  {reported}")
    return reported

# cal mean brightness
def mean_brightness(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

# resize the img and add a frame index on the left corner
def make_thumbnail(frame, index):
    height, width = frame.shape[:2]
    thumb_height = round(height * THUMB_WIDTH / width)
    thumb = cv2.resize(frame, (THUMB_WIDTH, thumb_height), interpolation=cv2.INTER_AREA)
    label = str(index)

    cv2.putText(thumb, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(thumb, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return thumb

# collect brightness from frame and thumbnail
def read_all_frames(cap, reported_count):
    brightness = []
    thumbnails = []
    with tqdm(total=reported_count or None, unit="frame") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = len(brightness)
            brightness.append(mean_brightness(frame))
            if index % SAMPLE_EVERY == 0:
                thumbnails.append(make_thumbnail(frame, index))
            progress.update(1)
    return np.array(brightness), thumbnails

# get median from brightness
# get the median from the median u get jn with absolute value
def find_low_brightness(brightness):
    median = np.median(brightness)
    robust_std = 1.4826 * np.median(np.abs(brightness - median))
    if robust_std == 0:
        robust_std = 1.0
    threshold = median - LOW_BRIGHTNESS_K * robust_std
    return np.flatnonzero(brightness < threshold), threshold


def plot_brightness(brightness, threshold, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(brightness)), brightness, linewidth=0.8)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8, label="low-brightness threshold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean brightness (0-255)")
    ax.set_title("Mean brightness per frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def save_contact_sheets(thumbnails, output_dir):
    per_sheet = SHEET_COLS * SHEET_ROWS
    blank = np.zeros_like(thumbnails[0])
    paths = []
    for sheet_number, start in enumerate(range(0, len(thumbnails), per_sheet)):
        tiles = thumbnails[start:start + per_sheet]
        tiles += [blank] * (per_sheet - len(tiles))
        rows = [np.hstack(tiles[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
        path = output_dir / f"contact_sheet_{sheet_number:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows))
        paths.append(path)
    return paths


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    reported = print_metadata(cap)
    brightness, thumbnails = read_all_frames(cap, reported)
    cap.release()

    print(f"Actual frames read:    {len(brightness)}")
    if len(brightness) != reported:
        print(f"  (differs from reported count by {len(brightness) - reported:+d})")
    if len(brightness) == 0:
        raise SystemExit("No frames could be read.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    low_frames, threshold = find_low_brightness(brightness)
    plot_brightness(brightness, threshold, OUTPUT_DIR / "brightness.png")
    print(f"Brightness: min {brightness.min():.1f}, median {np.median(brightness):.1f}, "
          f"max {brightness.max():.1f}, threshold {threshold:.1f}")
    print(f"Low-brightness frames ({len(low_frames)}): {low_frames.tolist()}")

    sheets = save_contact_sheets(thumbnails, OUTPUT_DIR)
    print(f"Saved {OUTPUT_DIR / 'brightness.png'}")
    print(f"Saved {len(sheets)} contact sheet(s) of every {SAMPLE_EVERY}th frame to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
