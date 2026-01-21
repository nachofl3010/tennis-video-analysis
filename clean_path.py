import numpy as np
from scipy.signal import savgol_filter

def _mad(x: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation (MAD) with Gaussian scaling.

    The MAD is a robust measure of statistical dispersion. It represents
    the median of absolute deviations from the data’s median and is scaled
    by 1.4826 to be consistent with the standard deviation for normally
    distributed data.

    Args:
        x (np.ndarray):
            Input array of numeric values. Can contain NaNs, which are ignored
            by `np.median` if the array is pre-filtered.

    Returns:
        float:
            Robust estimate of variability, similar in scale to standard deviation.
    """
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _runs(bool_arr: np.ndarray) -> list[tuple[int, int]]:
    """
    Identify consecutive True regions in a 1D boolean array.

    Returns a list of `(start, end)` index pairs (inclusive) corresponding
    to contiguous runs where the array is True. Useful for grouping
    consecutive detections or events.

    Args:
        bool_arr (np.ndarray):
            1D boolean array representing a binary condition over time or index.
        ```
    """
    idx = np.flatnonzero(bool_arr)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(g[0], g[-1]) for g in groups]


def _linear_interp_1d(y: np.ndarray) -> np.ndarray:
    """
    Perform 1D linear interpolation over NaN gaps.

    Missing (NaN) values inside the array are replaced by linear
    interpolation between neighboring valid points. Leading and trailing
    NaNs are replaced by the nearest valid edge value.

    Args:
        y (np.ndarray):
            1D numeric array that may contain NaN values.

    Returns:
        np.ndarray:
            Array of the same shape with all NaN values filled.
        ```
    """
    n = y.shape[0]
    out = y.copy()
    isnan = np.isnan(out)
    if isnan.all():
        return out
    x = np.arange(n)
    first = np.flatnonzero(~isnan)
    if first.size:
        first_idx = first[0]
        last_idx = first[-1]
        out[:first_idx] = out[first_idx]
        out[last_idx + 1 :] = out[last_idx]
    isnan = np.isnan(out)
    if isnan.any():
        out[isnan] = np.interp(x[isnan], x[~isnan], out[~isnan])
    return out


def _savgol_safe(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """
    Apply Savitzky–Golay smoothing with an automatic fallback.

    When the array is too short for the chosen window and polynomial order,
    the function switches to a simple moving average. This ensures stability
    across short sequences and avoids numerical errors.

    Args:
        y (np.ndarray):
            1D numeric array representing a signal or coordinate sequence.
        window (int):
            Window length for the Savitzky–Golay filter. Must be odd and
            smaller than or equal to the sequence length.
        poly (int):
            Polynomial order for the filter. Defines how flexible the fit is.

    Returns:
        np.ndarray:
            Smoothed version of the input array, same shape as input.
    """
    y = y.astype(float)
    n = y.shape[0]
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2:
        k = min(5, n)
        if k < 2:
            return y
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(k) / k
        return np.convolve(ypad, kernel, mode="valid")
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def clean_tennis_paths(
    video_xy: np.ndarray,
    court_dims: tuple = (10.97, 23.77), # Standard tennis court width/length in meters
    jump_sigma: float = 6.0,            # Higher sigma for explosive tennis starts
    min_jump_dist: float = 0.8,         # Tennis players cover ground faster in bursts
    max_jump_run: int = 5,              # Glitches usually shorter in open-court tennis
    pad_around_runs: int = 1,
    smooth_window: int = 5,             # Smaller window to preserve sharp cuts/slides
    smooth_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adapted trajectory cleaning for Tennis.
    
    Changes from Basketball version:
    1. Tighter smoothing (window=5) to preserve sharp cuts.
    2. Higher jump tolerance (sigma=6.0) for explosive sprints.
    3. 'Hard' boundary clipping: removes points impossibly far from court.
    """
    T, P, _ = video_xy.shape
    cleaned = video_xy.astype(float).copy()
    edited = np.zeros((T, P), dtype=bool)

    # Tennis Court Constraints (in meters, assuming center is 0,0)
    # Adding buffer: Players rarely go >4m wide or >5m behind baseline
    half_w, half_l = court_dims[0]/2, court_dims[1]/2
    x_limit = half_w + 4.0 
    y_limit = half_l + 5.0

    for p in range(P):
        traj = cleaned[:, p, :]  # (T, 2)
        
        # --- NEW: Hard Boundary Check ---
        # If tracking drifts into the stands, kill it immediately.
        # Assumes (0,0) is center court. Adjust if your origin is top-left.
        out_of_bounds = (np.abs(traj[:, 0]) > x_limit) | (np.abs(traj[:, 1]) > y_limit)
        
        # Calculate speed
        diffs = np.diff(traj, axis=0)
        speed = np.linalg.norm(diffs, axis=1)

        # Handle empty/static tracks
        valid_speed = speed[np.isfinite(speed)]
        if valid_speed.size == 0:
            continue

        # --- Statistics ---
        med = np.median(valid_speed)
        scale = _mad(valid_speed)
        scale = max(scale, 1e-6)

        # Detect Outliers
        jump_by_sigma = speed > (med + jump_sigma * scale)
        jump_by_abs = speed > min_jump_dist
        
        # Combine filters: Statistical Jump OR Out of Bounds
        jump_mask = jump_by_sigma & jump_by_abs
        
        # Pad bounds mask to match speed shape (T-1) -> T
        # (We treat a point as bad if it is out of bounds)
        oob_mask = out_of_bounds.copy()
        
        remove = np.zeros(T, dtype=bool)
        
        # 1. Process Out of Bounds "runs"
        if oob_mask.any():
             edited[:, p] |= oob_mask
             traj[oob_mask] = np.nan
             remove |= oob_mask

        # 2. Process Speed Jumps "runs"
        for s, e in _runs(jump_mask):
            length = e - s + 1
            if length <= max_jump_run:
                rs = max(0, s - pad_around_runs)
                re = min(T - 1, e + 1 + pad_around_runs)
                remove[rs : re + 1] = True

        # --- Apply Changes ---
        if not remove.any() and not oob_mask.any():
            # If clean, just smooth
            for d in range(2):
                traj[:, d] = _savgol_safe(traj[:, d], smooth_window, smooth_poly)
            cleaned[:, p, :] = traj
            continue

        # Mark edited frames
        edited[:, p] |= remove
        traj_nan = traj.copy()
        traj_nan[remove, :] = np.nan

        # Interpolate & Smooth
        for d in range(2):
            traj_nan[:, d] = _linear_interp_1d(traj_nan[:, d])
            traj_nan[:, d] = _savgol_safe(traj_nan[:, d], smooth_window, smooth_poly)

        cleaned[:, p, :] = traj_nan

    return cleaned, edited
