"""
Bounce detection from a tracked ball trajectory in image space.

The geometry that makes this work is worth stating up front, because the naive
approach fails badly.

A broadcast camera sits behind and above one baseline, so the ball's image `y`
is driven by *two* things at once: how far away the ball is (depth) and how high
it is off the court (height). Both push `y` in the same direction — a ball that
is farther away and a ball that is higher both appear higher in the frame.

Two consequences follow:

1. **A bounce is usually not a local extremum of `y`.** During a shot the depth
   term dominates, so `y` can run monotonically straight through the bounce. A
   detector built on `find_peaks(y)` misses those bounces entirely.

2. **What `find_peaks(y)` does find is racket contacts.** When the near player
   strikes, the ball is at its lowest in frame (a maximum of `y`); when the far
   player strikes, it is at its highest (a *minimum*, which a peak-finder looking
   for maxima cannot see at all).

The reliable bounce signature is curvature. At a bounce the ball's vertical
velocity flips from downward to upward while its depth motion continues
unchanged, so `dy/dt` drops sharply — a negative spike in `d2y/dt2` — whichever
direction the ball is travelling.

That spike alone is not enough, because racket contacts produce even larger
spikes. The structure of a rally resolves it: contacts and bounces strictly
alternate, so the contacts are located first and exactly one bounce is taken
from each interval between them.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def ball_centres(ball_boxes: np.ndarray) -> np.ndarray:
    """
    Convert per-frame ball bounding boxes to centre points, marking misses.

    Frames where the tracker found nothing are stored as an all-zero box. Those
    become NaN rather than the point (0, 0), which would otherwise read as a
    real position in the top-left corner of the frame.

    Args:
        ball_boxes (np.ndarray):
            Array of shape `(T, 4)` holding `[x1, y1, x2, y2]` per frame.

    Returns:
        np.ndarray:
            Array of shape `(T, 2)` of `(x, y)` centres, with NaN on frames
            where the ball was not detected.
    """
    boxes = ball_boxes.astype(float)
    centres = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
    )
    centres[(boxes == 0).all(axis=1)] = np.nan
    return centres


def _fill_and_smooth(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    """
    Interpolate NaN gaps and apply Savitzky-Golay smoothing.

    Args:
        values (np.ndarray): 1D signal that may contain NaN.
        window (int): Savitzky-Golay window length. Must be odd.
        poly (int): Polynomial order.

    Returns:
        np.ndarray: Gap-free, smoothed signal of the same length.
    """
    filled = (
        pd.Series(values).interpolate(method="linear", limit_direction="both").to_numpy()
    )
    return savgol_filter(filled, window, poly)


def smoothed_trajectory(
    ball_boxes: np.ndarray, smooth_window: int = 9, smooth_poly: int = 3
) -> np.ndarray:
    """
    Interpolated, smoothed ball trajectory in image space.

    Useful on its own for mapping detected bounces to court coordinates: the
    smoothed position is a better estimate of where the ball actually was than
    the raw box centre, which the tracker repeats on roughly a third of frames.

    Args:
        ball_boxes (np.ndarray): Array of shape `(T, 4)`.
        smooth_window (int, default=9): Savitzky-Golay window. Must be odd.
        smooth_poly (int, default=3): Polynomial order.

    Returns:
        np.ndarray: Array of shape `(T, 2)` of smoothed `(x, y)` centres.
    """
    centres = ball_centres(ball_boxes)
    return np.stack(
        [_fill_and_smooth(centres[:, d], smooth_window, smooth_poly) for d in (0, 1)],
        axis=1,
    )


def _unreliable_frames(ball_boxes: np.ndarray, guard: int) -> np.ndarray:
    """
    Flag frames whose curvature cannot be trusted.

    Missing detections are filled by linear interpolation, which replaces a
    curved arc with a straight segment. The joins at either end of that segment
    are corners, and a corner is exactly what the detector is looking for — so
    gaps produce convincing false bounces if left in.

    Args:
        ball_boxes (np.ndarray): Array of shape `(T, 4)`.
        guard (int): Number of frames to exclude either side of each gap.

    Returns:
        np.ndarray: Boolean mask of shape `(T,)`, True where unreliable.
    """
    missing = (ball_boxes.astype(float) == 0).all(axis=1)
    mask = np.zeros(len(ball_boxes), dtype=bool)
    for i in np.flatnonzero(missing):
        mask[max(0, i - guard) : i + guard + 1] = True
    return mask


def detect_strokes(
    smooth_y: np.ndarray,
    curvature: np.ndarray,
    far_prominence: float = 50.0,
    near_curvature: float = 3.0,
    min_separation: int = 12,
) -> np.ndarray:
    """
    Locate racket contacts, which bound the intervals a bounce can fall in.

    The two players need different criteria, because the camera is not
    symmetric with respect to them:

    - **Far player.** The ball is at its greatest distance, so image `y` reaches
      a pronounced minimum. That minimum is large and unambiguous, but the
      associated curvature is weak, because a distant ball moves few pixels.
    - **Near player.** The ball is close, so the reversal is violent in pixel
      terms and shows up as by far the sharpest downward kink in `y`.

    Args:
        smooth_y (np.ndarray): Smoothed image-`y` of the ball, shape `(T,)`.
        curvature (np.ndarray): Second derivative of image-`y`, shape `(T,)`.
        far_prominence (float, default=50.0):
            Minimum prominence, in pixels, for the `y` minimum of a far-court
            contact. Well separated from noise, which sits below ~15 px.
        near_curvature (float, default=3.0):
            Minimum `|d2y/dt2|` for a near-court contact. Bounces in the sample
            clip peak at 2.5, contacts at 3.6 and above, so this sits in a
            genuine gap rather than on a slope.
        min_separation (int, default=12):
            Minimum spacing between contacts, in frames.

    Returns:
        np.ndarray: Sorted frame indices of detected racket contacts.
    """
    far, _ = find_peaks(-smooth_y, prominence=far_prominence)
    near, _ = find_peaks(-curvature, prominence=1.0, distance=min_separation)
    near = [int(i) for i in near if curvature[i] < -near_curvature]
    return np.array(sorted(set(int(i) for i in far) | set(near)), dtype=int)


def detect_bounces(
    ball_boxes: np.ndarray,
    smooth_window: int = 9,
    smooth_poly: int = 3,
    deriv_window: int = 13,
    deriv_poly: int = 2,
    stroke_margin: int = 9,
    gap_guard: int = 3,
    far_prominence: float = 50.0,
    near_curvature: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect ball bounces, and the racket contacts used to find them.

    Racket contacts are located first. Because contacts and bounces alternate
    through a rally, exactly one bounce is then taken from each interval between
    consecutive contacts: the frame of sharpest negative curvature in image `y`.

    Note that this structure fixes the *number* of bounces at one per interval.
    That is what makes the method robust — it cannot emit two bounces 30 ms
    apart — but it also means a shot that is volleyed out of the air, or one
    that bounces twice before being returned, will not be represented correctly.

    Args:
        ball_boxes (np.ndarray):
            Array of shape `(T, 4)` of per-frame ball boxes `[x1, y1, x2, y2]`,
            with all-zero rows for frames where the ball was not detected.

        smooth_window (int, default=9):
            Savitzky-Golay window for the smoothed trajectory. Must be odd.

        smooth_poly (int, default=3):
            Polynomial order for the smoothed trajectory.

        deriv_window (int, default=13):
            Window used for the second derivative. Wider than `smooth_window`
            on purpose: differentiating twice amplifies noise, and the tracker
            repeats an identical box on roughly a third of frames, which puts a
            staircase into the raw signal.

        deriv_poly (int, default=2):
            Polynomial order for the second derivative.

        stroke_margin (int, default=9):
            Frames excluded either side of a racket contact when searching for
            a bounce. Without it the contact's own curvature, which is larger
            than any bounce, wins the search in its own interval.

        gap_guard (int, default=3):
            Frames excluded either side of a missing detection. See
            `_unreliable_frames`.

        far_prominence (float, default=50.0):
            Passed to `detect_strokes`.

        near_curvature (float, default=3.0):
            Passed to `detect_strokes`.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - `bounce_frames`: Sorted frame indices where the ball hit the court.
            - `stroke_frames`: Sorted frame indices of racket contacts.
    """
    centres = ball_centres(ball_boxes)
    smooth_y = smoothed_trajectory(ball_boxes, smooth_window, smooth_poly)[:, 1]

    filled_y = (
        pd.Series(centres[:, 1])
        .interpolate(method="linear", limit_direction="both")
        .to_numpy()
    )
    curvature = savgol_filter(filled_y, deriv_window, deriv_poly, deriv=2)

    strokes = detect_strokes(
        smooth_y, curvature, far_prominence, near_curvature
    )
    unreliable = _unreliable_frames(ball_boxes, gap_guard)

    bounces = []
    for start, end in zip(np.concatenate([[0], strokes[:-1]]), strokes):
        lo = start + stroke_margin if start > 0 else 0
        hi = end - stroke_margin
        if hi <= lo:
            continue
        window = np.arange(lo, hi)
        window = window[~unreliable[window]]
        if window.size:
            bounces.append(int(window[np.argmin(curvature[window])]))

    return np.array(bounces, dtype=int), strokes
