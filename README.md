# Tennis Video Analysis

Turn a broadcast-style tennis clip into court-level analytics: a top-down view of both
players and the ball, distance covered per player, and automatic bounce detection.

The pipeline segments and tracks the two players and the ball with **SAM 2**, then maps
every detection from image space into real-world court coordinates (metres) via a
homography. Once positions live in metres instead of pixels, questions like *"how far did
each player run during this point?"* become straightforward geometry.

## What it does

1. **Track** — SAM 2 tracks three objects (`player-1`, `player-2`, `ball`) across the clip
   from a single annotated reference frame. Per-frame bounding boxes are cached to
   `rectangles_*.npy` so the analysis can be re-run without a GPU.
2. **Map to court coordinates** — four manually picked court corners define a homography
   that projects image points onto a 23.77 × 10.97 m court. Players are mapped by the
   bottom-centre of their box (foot position); the ball by its centre.
3. **Clean trajectories** — raw tracking output jitters and occasionally jumps. Outliers are
   flagged with a MAD-based robust z-score, short bad runs are interpolated away, and the
   result is smoothed with a Savitzky–Golay filter.
4. **Analyse** — cumulative distance run per player, and ball bounces detected from sign
   changes in the vertical velocity of the smoothed ball trajectory.
5. **Render** — a top-down court video with player positions and a live distance counter.

On the sample point, player 1 covered **19.02 m** and player 2 **21.47 m**.

## Stack

Python · SAM 2 · OpenCV · NumPy · pandas · SciPy · supervision · Matplotlib / Plotly

## Repository layout

| File | Purpose |
|---|---|
| `player_ball_tracking.ipynb` | Main notebook — tracking, homography, analysis, rendering |
| `tennis_drawer.py` | Draws a regulation top-down court; plots points and paths on it |
| `path.py` | Trajectory cleaning — MAD outlier detection, jump removal, Savitzky–Golay smoothing |
| `rectangles_*.npy` | Cached per-frame bounding boxes (players and ball) for the sample clip |

## Running it

The notebook is written for **Google Colab with a GPU runtime** — SAM 2 inference over a
video is not practical on CPU.

1. Open `player_ball_tracking.ipynb` in Colab and set the runtime to GPU.
2. Run the setup cells (they install SAM 2 and download the checkpoints).
3. Upload your own clip and point `SOURCE_VIDEO` at it. The sample clip is not included in
   this repository.
4. In the `BBoxWidget` cell, draw one box each around the two players and the ball on the
   reference frame — this is the only manual annotation the tracker needs.
5. Pick the four court corners for the homography. **This step is per-video**, since the
   camera angle changes from clip to clip.

To skip tracking entirely and go straight to the analysis, run the import cells and load the
cached `rectangles_*.npy` arrays.

## Limitations

- Court corners and the initial object boxes are annotated by hand, once per clip.
- Tuned and validated on a single point from one camera angle.
- Bounce detection relies on vertical-velocity reversals, so it can be fooled by occlusion
  or by the ball passing in front of a player.

## Credits

The tracking and setup sections of the notebook are adapted from the
[Roboflow SAM 2 video segmentation notebook](https://github.com/roboflow/notebooks),
which is where this project started. The court mapping, trajectory cleaning, distance and
bounce analysis, and the rendering code are my own.

Built on [SAM 2](https://github.com/facebookresearch/segment-anything-2) (Meta AI,
Apache-2.0) and [supervision](https://github.com/roboflow/supervision) (Roboflow, MIT).
