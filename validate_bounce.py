"""
Validate the bounce detector against hand-labelled ground truth.

The labels below were read off the sample clip frame by frame. Indices are into
the cached `rectangles_*.npy` arrays; add 147 to get the frame number in the
source video, which was tracked from its 147th frame onward.

Run with:  python validate_bounce.py
"""

import numpy as np

from bounce import detect_bounces

FPS = 60

# Hand-labelled ball-to-court contacts for the sample point.
GROUND_TRUTH = np.array([16, 86, 169, 239, 315, 386, 502])

# Tolerance in frames. At 60 fps this is 50 ms — under two frames of a
# broadcast at 30 fps, and far below what matters for downstream analysis.
TOLERANCE = 3


def main() -> int:
    bounces, strokes = detect_bounces(np.load("rectangles_ball.npy"))

    print(f"racket contacts : {strokes.tolist()}")
    print(f"bounces detected: {bounces.tolist()}")
    print(f"ground truth    : {GROUND_TRUTH.tolist()}\n")

    if len(bounces) != len(GROUND_TRUTH):
        print(f"FAIL: detected {len(bounces)} bounces, expected {len(GROUND_TRUTH)}")
        return 1

    errors = bounces - GROUND_TRUTH
    for truth, found, err in zip(GROUND_TRUTH, bounces, errors):
        status = "ok" if abs(err) <= TOLERANCE else "OUT OF TOLERANCE"
        print(f"  {truth:>4} -> {found:>4}  {err:+d} frames ({err / FPS * 1000:+.0f} ms)  {status}")

    worst = int(np.abs(errors).max())
    mean = float(np.abs(errors).mean())
    print(
        f"\n{(np.abs(errors) <= TOLERANCE).sum()}/{len(GROUND_TRUTH)} within "
        f"{TOLERANCE} frames | mean error {mean:.1f} frames "
        f"({mean / FPS * 1000:.0f} ms) | worst {worst} frames"
    )

    if worst > TOLERANCE:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
