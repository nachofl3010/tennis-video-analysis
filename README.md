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
4. **Analyse** — cumulative distance run per player, and ball bounces detected from the
   curvature of the ball's image trajectory (see below).
5. **Render** — a top-down court video with player positions and a live distance counter.

On the sample point, player 1 covered **16.74 m** and player 2 **24.78 m**.

## Detecting bounces

The obvious approach — look for peaks in the ball's image `y` — does not work, and it is
worth saying why, because the failure is geometric rather than a matter of tuning.

A broadcast camera sits behind and above one baseline, so image `y` mixes two things:
how far away the ball is, and how high it is off the court. Over a single shot the
distance term dominates. On one segment of the sample clip the ball's `y` runs from 350
to 843 px **monotonically**, straight through a bounce, with no local maximum anywhere.
What peaks of `y` actually mark is *racket contacts* — and only the near player's, since
the far player's contacts are minima.

A bounce instead shows up as curvature. The ball's vertical velocity flips while its
motion in depth carries on, so `d²y/dt²` spikes negative whichever way the ball is
travelling. Racket contacts spike harder, so `bounce.py` locates contacts first and takes
one bounce from each interval between them — the two alternate through a rally.

Against hand-labelled bounces for the sample point: **7/7 detected, no false positives,
mean error 1.4 frames (24 ms) at 60 fps, worst case 3 frames.** Reproduce with:

```bash
python validate_bounce.py
```

## Stack

Python · SAM 2 · OpenCV · NumPy · pandas · SciPy · supervision · Matplotlib / Plotly

## Repository layout

| File | Purpose |
|---|---|
| `player_ball_tracking.ipynb` | Main notebook — tracking, homography, analysis, rendering |
| `track_new_clip.ipynb` | Colab/GPU notebook that produces cached arrays for a **new** clip |
| `run_analysis.py` | Runs the whole analysis on CPU from the cached arrays |
| `tennis_drawer.py` | Draws a regulation top-down court; plots points and paths on it |
| `path.py` | Trajectory cleaning — MAD outlier detection, jump removal, Savitzky–Golay smoothing |
| `bounce.py` | Bounce and racket-contact detection from trajectory curvature |
| `validate_bounce.py` | Checks `bounce.py` against hand-labelled bounces for the sample point |
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

### Analysis only, locally (no GPU)

Everything downstream of tracking — homography, trajectory cleaning, distance and bounce
analysis, rendering — runs on CPU from the cached arrays:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

`run_analysis.py` is the quickest way to confirm everything works. It reads the cached
arrays and prints distance covered, the jitter reduction from cleaning, and every
detected bounce in court coordinates — no GPU, no SAM 2, no source video:

```
distance covered
  player 1: 16.74 m cleaned   (18.90 m raw, 0 frames flagged as outliers)
  player 2: 24.78 m cleaned   (28.43 m raw, 0 frames flagged as outliers)

jitter (mean |acceleration|, m/frame^2)
  player 1: 0.0398 -> 0.0068   (83% reduction)
  ...
```

`requirements.txt` deliberately excludes SAM 2 and torch, which are only needed to
re-run tracking in Colab. In the notebook, **skip the setup cells** — they install SAM 2
and download ~1.5 GB of checkpoints, and they fail without a GPU. Start instead from the
cell that loads `rectangles_*.npy`.

## Tracking a new clip

`track_new_clip.ipynb` runs on Colab with a GPU and produces the cached arrays for a new
video, plus a `clip_meta.json` holding that clip's court corners and frame offset. Then:

```bash
python run_analysis.py --clip tracking_<name>
```

**On the frame offset.** Array index `i` is not video frame `i`. Frames are extracted from
`START_IDX`, and SAM 2 propagates forward from the annotated frame, so index `i` is video
frame `i + start_idx + annotation_frame`. For the sample clip that offset is **147** — the
arrays cover video frames 147–699. Anything that draws tracking output over the source
video has to account for it, or everything lands on empty court. `track_new_clip.ipynb`
records the offset when the arrays are made and checks it visually before you download.

## Limitations

- Court corners and the initial object boxes are annotated by hand, once per clip.
- Tuned and validated on a single point from one camera angle.
- Bounce detection assumes bounces and racket contacts strictly alternate. That is what
  makes it robust — it cannot emit two bounces 30 ms apart — but a ball volleyed out of
  the air, or one that bounces twice before being returned, will not be represented
  correctly. It also fixes the bounce *count* at one per interval, so the count follows
  from the detected contacts; it is the bounce *positions* that the 24 ms figure measures.
- The homography describes the court plane, so it maps the ball correctly only on frames
  where the ball is touching the ground. Bounce positions are mapped; a full ball
  trajectory in court coordinates would not be meaningful.

## Credits

The tracking and setup sections of the notebook are adapted from the
[Roboflow SAM 2 video segmentation notebook](https://github.com/roboflow/notebooks),
which is where this project started. The court mapping, trajectory cleaning, distance and
bounce analysis, and the rendering code are my own.

Built on [SAM 2](https://github.com/facebookresearch/segment-anything-2) (Meta AI,
Apache-2.0) and [supervision](https://github.com/roboflow/supervision) (Roboflow, MIT).
