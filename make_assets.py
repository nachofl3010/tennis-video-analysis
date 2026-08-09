"""
Render the figures used in the README, from the cached tracking arrays.

    python make_assets.py

Everything is drawn from coordinates, so nothing here reproduces any frame of the
source broadcast — only the court-space output the pipeline computes.

Writes into `assets/`:

    point_top_down.gif   the point replayed top-down, with a live distance counter
    trajectories.png     both player paths and every detected bounce
    cleaning.png         raw versus cleaned trajectory, the jitter figure made visible
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

import bounce as bd
import path as cp
import tennis_drawer as td
from run_analysis import (
    CLEAN_KWARGS,
    COURT_LENGTH,
    COURT_WIDTH,
    court_homography,
    player_court_positions,
    to_court,
)

ASSETS = Path("assets")
SCALE, PADDING = 50, 260

P1_COLOUR = sv.Color.from_hex("#FF3B6B")
P2_COLOUR = sv.Color.from_hex("#2FA8FF")
BALL_COLOUR = sv.Color.from_hex("#FFD400")


def landscape(image: np.ndarray, width: int) -> np.ndarray:
    """
    Rotate a court render to landscape and scale it to a target width.

    `tennis_drawer` draws the court portrait, with the length running down the
    image. Rotating reads better in a README and wastes less horizontal space.

    Args:
        image (np.ndarray): The court render.
        width (int): Target width in pixels.

    Returns:
        np.ndarray: The rotated, resized image.
    """
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    height = int(rotated.shape[0] * width / rotated.shape[1])
    return cv2.resize(rotated, (width, height), interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, text: str, origin: tuple[int, int], colour: sv.Color) -> None:
    """Draw text with a dark outline so it stays legible over any court colour."""
    for thickness, shade in ((5, (0, 0, 0)), (2, colour.as_bgr())):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, shade,
                    thickness, cv2.LINE_AA)


def trajectories_figure(cleaned: np.ndarray, bounces: np.ndarray, ball_xy: np.ndarray,
                        homography: np.ndarray, distances: list[float]) -> np.ndarray:
    """Both cleaned player paths, plus every detected bounce, on one court."""
    court = td.draw_court(scale=SCALE, padding=PADDING)
    for player, colour in ((0, P1_COLOUR), (1, P2_COLOUR)):
        td.draw_paths_on_court([cleaned[:, player]], image=court, color=colour,
                               thickness=5, scale=SCALE, padding=PADDING)
    for frame in bounces:
        point = to_court(ball_xy[frame], homography)
        td.draw_points_on_court(point, image=court, color=BALL_COLOUR, radius=18,
                                scale=SCALE, padding=PADDING)
    out = landscape(court, 1100)
    _label(out, f"player 1  -  {distances[0]:.2f} m", (24, 42), P1_COLOUR)
    _label(out, f"player 2  -  {distances[1]:.2f} m", (24, 78), P2_COLOUR)
    _label(out, f"{len(bounces)} bounces", (24, 114), BALL_COLOUR)
    return out


def cleaning_figure(raw: np.ndarray, cleaned: np.ndarray, path: Path) -> None:
    """
    Raw against cleaned trajectory, zoomed until the difference is visible.

    Drawn on the court the two paths sit on top of each other and the figure says
    nothing: an 83% jitter reduction is a centimetre-scale effect on a path
    metres across. These panels zoom to each player's own bounding box.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jitter = lambda a: float(np.abs(np.diff(a, n=2, axis=0)).mean())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Windows chosen where each player moves slowly enough that the tracker's
    # repeated boxes are visible as steps rather than lost in a long sweep.
    for ax, player, colour, name, start, window in (
            (axes[0], 0, "#FF3B6B", "Player 1", 150, 120),
            (axes[1], 1, "#2FA8FF", "Player 2", 345, 70)):
        sl = slice(start, start + window)
        # whichever axis the player moved along most in this window
        axis = int(np.argmax(np.ptp(raw[sl, player], axis=0)))
        frames = np.arange(start, start + window)

        ax.plot(frames, raw[sl, player, axis], color="#AAB3C2", lw=2.4,
                label="raw tracking", solid_capstyle="round")
        ax.plot(frames, cleaned[sl, player, axis], color=colour, lw=2.0,
                label="cleaned (path.py)", solid_capstyle="round")

        drop = 100 * (1 - jitter(cleaned[:, player]) / jitter(raw[:, player]))
        ax.set_title(f"{name}  —  {drop:.0f}% less jitter", fontsize=13, pad=8)
        ax.set_xlabel("frame")
        ax.set_ylabel("court " + ("length" if axis == 0 else "width") + " (m)")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "SAM 2 repeats an identical box on ~40% of frames, so the raw path climbs in steps. "
        "MAD outlier rejection + Savitzky–Golay recovers the motion.",
        fontsize=10.5, y=1.0, color="#55607A")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def replay_gif(cleaned: np.ndarray, bounces: np.ndarray, ball_xy: np.ndarray,
               homography: np.ndarray, step: int = 6, width: int = 640) -> list[Image.Image]:
    """
    Replay the point top-down: trails behind each player, a live distance
    counter, and bounces appearing as they happen.
    """
    steps = np.linalg.norm(np.diff(cleaned, axis=0), axis=2)
    cumulative = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    bounce_points = {int(f): to_court(ball_xy[f], homography)[0] for f in bounces}

    frames = []
    for i in range(0, len(cleaned), step):
        court = td.draw_court(scale=SCALE, padding=PADDING)
        for frame, point in bounce_points.items():          # bounces already played
            if frame <= i:
                td.draw_points_on_court(point[None, :], image=court, color=BALL_COLOUR,
                                        radius=17, scale=SCALE, padding=PADDING)
        for player, colour in ((0, P1_COLOUR), (1, P2_COLOUR)):
            trail = cleaned[max(0, i - 90):i + 1, player]
            td.draw_paths_on_court([trail], image=court, color=colour, thickness=6,
                                   scale=SCALE, padding=PADDING)
            td.draw_points_on_court(cleaned[i, player][None, :], image=court,
                                    color=colour, radius=20, scale=SCALE, padding=PADDING)
        out = landscape(court, width)
        _label(out, f"P1 {cumulative[i, 0]:5.1f} m", (16, 30), P1_COLOUR)
        _label(out, f"P2 {cumulative[i, 1]:5.1f} m", (16, 62), P2_COLOUR)
        frames.append(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
    return frames


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    homography = court_homography()
    p1 = np.load("rectangles_p1.npy")
    p2 = np.load("rectangles_p2.npy")
    ball = np.load("rectangles_ball.npy")

    raw = np.stack([player_court_positions(b, homography) for b in (p1, p2)], axis=1)
    cleaned, _ = cp.clean_paths(raw, **CLEAN_KWARGS)
    distances = [float(np.linalg.norm(np.diff(cleaned[:, p], axis=0), axis=1).sum())
                 for p in (0, 1)]
    bounces, _ = bd.detect_bounces(ball)
    ball_xy = bd.smoothed_trajectory(ball)

    cv2.imwrite(str(ASSETS / "trajectories.png"),
                trajectories_figure(cleaned, bounces, ball_xy, homography, distances))
    cleaning_figure(raw, cleaned, ASSETS / "cleaning.png")

    frames = replay_gif(cleaned, bounces, ball_xy, homography)
    frames[0].save(ASSETS / "point_top_down.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)

    for asset in sorted(ASSETS.iterdir()):
        print(f"  {asset}  {asset.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
