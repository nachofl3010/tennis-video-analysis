"""
Run the whole analysis on CPU from the cached tracking arrays.

This is the quickest way to check that everything still works. It needs no GPU,
no SAM 2 and no source video — only the `rectangles_*.npy` files in this
directory, which hold the tracking output for the sample point.

    python run_analysis.py

Everything it does mirrors the notebook: map detections onto the court through
the homography, clean the player trajectories, measure distance covered, and
detect bounces. The notebook remains the place to look at the intermediate
plots; this script is for checking the numbers come out right.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2

import bounce as bd
import path as cp

COURT_LENGTH = 23.77
COURT_WIDTH = 10.97

# Court corners as picked by hand on frame 130 of the sample clip, in pixels,
# clockwise from the far-left corner. These are specific to this clip: the
# camera angle changes from video to video, so a new clip needs new corners.
IMAGE_CORNERS = np.float32([[592, 362], [1327, 367], [1571, 897], [339, 890]])
COURT_CORNERS = np.float32(
    [[0, 0], [0, COURT_WIDTH], [COURT_LENGTH, COURT_WIDTH], [COURT_LENGTH, 0]]
)

# Trajectory cleaning, tuned for tennis: motion is smoother and jumps smaller
# than the defaults in path.py assume.
CLEAN_KWARGS = dict(
    jump_sigma=4.0,
    min_jump_dist=0.4,
    max_jump_run=4,
    pad_around_runs=1,
    smooth_window=11,
    smooth_poly=2,
)


def court_homography(image_corners: np.ndarray = IMAGE_CORNERS) -> np.ndarray:
    """
    Homography mapping image pixels onto court coordinates in metres.

    Args:
        image_corners (np.ndarray):
            The four court corners in pixels, clockwise from the far-left
            corner. Defaults to the sample clip's.

    Returns:
        np.ndarray: The 3x3 homography matrix.
    """
    matrix, _ = cv2.findHomography(np.float32(image_corners), COURT_CORNERS)
    return matrix


def load_clip(clip_dir: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load cached tracking arrays, and the court corners that go with them.

    A clip produced by `track_new_clip.ipynb` carries a `clip_meta.json` holding
    its own corners and frame offset, since both are specific to the camera
    angle and the tracked range. Without one, the sample clip's values are used.

    Args:
        clip_dir (Path | None):
            Directory holding `rectangles_*.npy` and optionally
            `clip_meta.json`. `None` means the current directory.

    Returns:
        tuple: `(p1_boxes, p2_boxes, ball_boxes, image_corners, metadata)`.
    """
    base = clip_dir or Path(".")
    boxes = tuple(np.load(base / f"rectangles_{n}.npy") for n in ("p1", "p2", "ball"))

    meta_path = base / "clip_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        corners = np.float32(meta["court_corners_px"])
    else:
        meta = {"clip": "sample point", "frame_offset": 147, "fps": 60}
        corners = IMAGE_CORNERS
    return (*boxes, corners, meta)


def to_court(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    Project image points onto the court plane.

    Only valid for points that lie on the court surface. A ball in flight
    projects to a point well beyond where it actually is, which is why bounce
    positions are mapped but whole ball trajectories are not.

    Args:
        points (np.ndarray): Array of shape `(N, 2)` of `(x, y)` pixel positions.
        homography (np.ndarray): The 3x3 matrix from `court_homography`.

    Returns:
        np.ndarray: Array of shape `(N, 2)` of `(x, y)` court positions in metres.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def player_court_positions(boxes: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    Map a player's bounding boxes to court positions via their feet.

    The bottom-centre of the box is used, because that is the point actually in
    contact with the court and therefore the only part of the player the
    ground-plane homography describes correctly.

    Args:
        boxes (np.ndarray): Array of shape `(T, 4)` of `[x1, y1, x2, y2]`.
        homography (np.ndarray): The 3x3 matrix from `court_homography`.

    Returns:
        np.ndarray: Array of shape `(T, 2)` of court positions in metres.
    """
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    return to_court(feet, homography)


def distance_run(path: np.ndarray) -> float:
    """
    Total distance covered along a trajectory.

    Args:
        path (np.ndarray): Array of shape `(T, 2)` of court positions in metres.

    Returns:
        float: Distance in metres.
    """
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory of a clip tracked with track_new_clip.ipynb "
        "(reads its court corners and frame offset from clip_meta.json); "
        "defaults to the sample clip in this directory",
    )
    args = parser.parse_args()

    p1, p2, ball, corners, meta = load_clip(args.clip)
    homography = court_homography(corners)
    print(f"clip: {meta['clip']}  ({len(p1)} frames tracked, {meta['fps']:g} fps)")
    print(f"array index i is video frame i + {meta['frame_offset']}\n")

    raw = np.stack(
        [player_court_positions(b, homography) for b in (p1, p2)], axis=1
    )
    cleaned, edited = cp.clean_paths(raw, **CLEAN_KWARGS)

    print("distance covered")
    for p, name in enumerate(("player 1", "player 2")):
        print(
            f"  {name}: {distance_run(cleaned[:, p]):5.2f} m cleaned"
            f"   ({distance_run(raw[:, p]):5.2f} m raw,"
            f" {edited[:, p].sum()} frames flagged as outliers)"
        )

    # Jitter, as mean absolute frame-to-frame acceleration. Smoothing should cut
    # this sharply while leaving the distance broadly intact.
    jitter = lambda a: float(np.abs(np.diff(a, n=2, axis=0)).mean())
    print("\njitter (mean |acceleration|, m/frame^2)")
    for p, name in enumerate(("player 1", "player 2")):
        before, after = jitter(raw[:, p]), jitter(cleaned[:, p])
        print(f"  {name}: {before:.4f} -> {after:.4f}   ({100 * (1 - after / before):.0f}% reduction)")

    bounces, strokes = bd.detect_bounces(ball)
    smoothed = bd.smoothed_trajectory(ball)
    print(f"\nracket contacts ({len(strokes)}): {strokes.tolist()}")
    print(f"bounces ({len(bounces)}):")
    for frame in bounces:
        x, y = to_court(smoothed[frame], homography)[0]
        inside = 0 <= x <= COURT_LENGTH and 0 <= y <= COURT_WIDTH
        print(f"  frame {frame:>3}  ({x:5.2f} m, {y:5.2f} m)  {'in' if inside else 'OUT'}")


if __name__ == "__main__":
    main()
