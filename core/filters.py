# core/filters.py
from __future__ import annotations
import numpy as np
from functools import lru_cache

try:
    from scipy.signal import butter, sosfiltfilt
except Exception as e:
    raise RuntimeError(
        "SciPy is required for Butterworth filtering. Install with: pip install scipy"
    ) from e


def mean_subtract(y: np.ndarray, mean_val: float | None) -> np.ndarray:
    return y - (0.0 if mean_val is None else float(mean_val))


def zscore_normalize(y: np.ndarray, global_mean: float | None = None, global_std: float | None = None) -> np.ndarray:
    """
    Z-score normalization: (y - mean) / std.

    Normalizes the signal to have zero mean and unit standard deviation.
    Handles NaN values by computing statistics only on valid data.

    Args:
        y: Input signal array
        global_mean: Pre-computed global mean (optional). If provided, uses this instead of computing from y.
        global_std: Pre-computed global std (optional). If provided, uses this instead of computing from y.

    Returns:
        Z-score normalized signal (same shape as input)
    """
    y = np.asarray(y, dtype=float)

    # Handle all-NaN case
    if np.all(np.isnan(y)):
        return y

    # If global statistics are provided, use them
    if global_mean is not None and global_std is not None:
        mean_val = float(global_mean)
        std_val = float(global_std)
    else:
        # Compute mean and std only on valid (non-NaN) values
        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            return y

        mean_val = np.mean(y[valid_mask])
        std_val = np.std(y[valid_mask], ddof=1)  # Use sample std (ddof=1)

    # Avoid division by zero
    if std_val == 0 or not np.isfinite(std_val):
        # Signal is constant - just subtract mean
        return y - mean_val

    # Apply z-score normalization
    return (y - mean_val) / std_val


try:
    from scipy.ndimage import uniform_filter1d
    def _movmean(x: np.ndarray, n: int) -> np.ndarray:
        # Fast O(N) moving average with good edge handling
        return uniform_filter1d(x, size=n, mode='nearest')
except Exception:
    def _movmean(x: np.ndarray, n: int) -> np.ndarray:
        # Fallback: simple convolution
        if n <= 1:
            return x
        k = np.ones(int(n), dtype=float) / float(n)
        return np.convolve(x, k, mode="same")


def rolling_mean_subtract_1d(y: np.ndarray, sr_hz: float, window_seconds: float) -> np.ndarray:
    """
    Subtract a rolling average (centered approximate) from y.
    window_seconds: rolling window length in seconds (e.g., 5.0)
    """
    y = np.asarray(y, dtype=float)
    if window_seconds is None or window_seconds <= 0 or not np.isfinite(window_seconds):
        return y

    n = max(3, int(round(window_seconds * float(sr_hz))))
    # enforce odd size for nicer centering (optional)
    if n % 2 == 0:
        n += 1

    baseline = _movmean(y, n)
    return y - baseline




def invert(y: np.ndarray) -> np.ndarray:
    return -y


def _sanitize_cutoff(c: float, nyq: float) -> float:
    # keep within (0, nyq); also avoid exactly 0 or >= nyq which break butter()
    eps = 1e-9
    c = max(eps, min(float(c), nyq * (1 - 2e-3)))
    return c


@lru_cache(maxsize=128)
def _design_low(sr_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    nyq = 0.5 * sr_hz
    c = _sanitize_cutoff(cutoff_hz, nyq)
    wn = c / nyq
    return butter(order, wn, btype="lowpass", output="sos")


@lru_cache(maxsize=128)
def _design_high(sr_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    nyq = 0.5 * sr_hz
    c = _sanitize_cutoff(cutoff_hz, nyq)
    wn = c / nyq
    return butter(order, wn, btype="highpass", output="sos")


@lru_cache(maxsize=128)
def _design_band(sr_hz: float, low_cut_hz: float, high_cut_hz: float, order: int) -> np.ndarray:
    """
    Band-pass between low_cut_hz (HP edge) and high_cut_hz (LP edge).
    Requires: 0 < low_cut < high_cut < Nyquist
    """
    nyq = 0.5 * sr_hz
    lo = _sanitize_cutoff(low_cut_hz, nyq)
    hi = _sanitize_cutoff(high_cut_hz, nyq)
    if not (lo < hi):
        # Fallback: if invalid, return a low-pass at hi (safer default)
        return _design_low(sr_hz, hi, order)
    wn = (lo / nyq, hi / nyq)
    return butter(order, wn, btype="bandpass", output="sos")


def _sosfiltfilt_safe(sos: np.ndarray, y: np.ndarray, is_highpass: bool = False) -> np.ndarray:
    """
    sosfiltfilt with improved edge handling to minimize artifacts.

    For high-pass filters especially, edge effects can be significant because
    removing low frequencies creates transients at boundaries.

    Strategies used:
    1. Longer padding (padlen) to allow filter to settle
    2. 'odd' padding (reflection) which works well for most signals
    3. Fallback to shorter padding if signal is too short
    """
    n = y.shape[0]

    # Calculate appropriate padding length
    # For high-pass filters, use longer padding to reduce edge effects
    # Rule of thumb: pad by at least 3x the filter's impulse response length
    # For IIR filters, use a multiple of the signal length or a fixed minimum
    if is_highpass:
        # High-pass filters need more padding to handle DC offset removal
        padlen = min(n - 1, max(500, n // 4))
    else:
        # For low-pass, standard padding is usually sufficient
        padlen = min(n - 1, max(100, n // 8))

    try:
        # Use 'odd' padding (default) which reflects signal with odd symmetry
        # This generally works well for continuous signals
        return sosfiltfilt(sos, y, axis=0, padlen=padlen, padtype='odd')
    except ValueError:
        # Signal too short for requested padding - try smaller padding
        try:
            padlen = max(0, min(99, n // 3))
            return sosfiltfilt(sos, y, axis=0, padlen=padlen, padtype='odd')
        except ValueError:
            # Still too short - try minimal padding
            try:
                padlen = max(0, n // 4)
                return sosfiltfilt(sos, y, axis=0, padlen=padlen, padtype='constant')
            except Exception:
                # Give up and return original
                return y


def low_pass_1d(y: np.ndarray, sr_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = _design_low(sr_hz, cutoff_hz, order)
    return _sosfiltfilt_safe(sos, y, is_highpass=False)


def high_pass_1d(y: np.ndarray, sr_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = _design_high(sr_hz, cutoff_hz, order)
    return _sosfiltfilt_safe(sos, y, is_highpass=True)


def band_pass_1d(y: np.ndarray, sr_hz: float,
                 low_cut_hz: float, high_cut_hz: float, order: int = 4) -> np.ndarray:
    """
    Band-pass where low_cut_hz is the HIGH-PASS edge (lower bound),
    and high_cut_hz is the LOW-PASS edge (upper bound).
    """
    sos = _design_band(sr_hz, low_cut_hz, high_cut_hz, order)
    # Band-pass includes high-pass component, so use longer padding
    return _sosfiltfilt_safe(sos, y, is_highpass=True)


#                  use_low: bool,  low_hz: float | None,
#                  use_high: bool, high_hz: float | None,
#                  use_mean: bool, mean_val: float | None,
#                  use_inv: bool,
#     """
#     Applies mean subtraction, Butterworth HP/LP (or band-pass if both are enabled),
#     and optional inversion. Uses cached SOS for speed.
#     """
# def apply_all_1d(y: np.ndarray, sr_hz: float,
#                  order: int = 4) -> np.ndarray:
#     out = y

#     # Mean subtraction first (DC/baseline)
#     if use_mean and mean_val is not None:
#         out = mean_subtract(out, mean_val)

#     # Combine HP+LP into one band-pass when both are valid
#     has_lp = use_low and (low_hz is not None)
#     has_hp = use_high and (high_hz is not None)

#     if has_lp and has_hp:
#         # high_hz is lower bound (HP), low_hz is upper bound (LP)
#     elif has_hp:
#     elif has_lp:
#         out = band_pass_1d(out, sr_hz, low_cut_hz=high_hz, high_cut_hz=low_hz, order=order)
#         out = high_pass_1d(out, sr_hz, cutoff_hz=high_hz, order=order)
#         out = low_pass_1d(out, sr_hz, cutoff_hz=low_hz, order=order)

#     if use_inv:
#         out = invert(out)

#     return out

#     y: np.ndarray, sr_hz: float,
#     use_low: bool,  low_hz: float | None,
#     use_high: bool, high_hz: float | None,
#     use_mean_sub: bool, mean_param: float | None,   # NOW interpreted as window_seconds
#     use_invert: bool
# ) -> np.ndarray:
# def apply_all_1d(

#     x = np.asarray(y, dtype=float)

#     # Low-pass first (optional)
#     if use_low and low_hz is not None and low_hz > 0:
#         x = butter_lowpass_1d(x, sr_hz, low_hz)

#     # High-pass (optional)
#     if use_high and high_hz is not None and high_hz > 0:
#         x = butter_highpass_1d(x, sr_hz, high_hz)

#     # Rolling baseline subtraction (optional)
#     if use_mean_sub and mean_param is not None and mean_param > 0:
#         # mean_param is interpreted as window_seconds
#         x = rolling_mean_subtract_1d(x, sr_hz, mean_param)

#     if use_invert:
#         x = -x

#     return x

def notch_filter_1d(y: np.ndarray, sr_hz: float, lower_freq: float, upper_freq: float, order: int = 4) -> np.ndarray:
    """Apply a band-stop (notch) filter to remove frequencies between lower_freq and upper_freq.

    Pure function — no side effects. Replaces MainWindow._apply_notch_filter.
    """
    from scipy import signal as _sig

    nyquist = sr_hz / 2.0
    low = np.clip(lower_freq / nyquist, 0.001, 0.999)
    high = np.clip(upper_freq / nyquist, 0.001, 0.999)

    if low >= high:
        return y

    try:
        sos = _sig.butter(order, [low, high], btype="bandstop", output="sos")
        return _sig.sosfiltfilt(sos, y)
    except Exception:
        return y


def apply_all_1d(
    y: np.ndarray,
    sr_hz: float,
    use_low: bool,  low_hz: float | None,
    use_high: bool, high_hz: float | None,
    use_mean: bool, mean_param: float | None,
    use_inv: bool,
    order: int = 4,
    mean_mode: str = "window",   # "window" (rolling) or "const" (subtract a number)
) -> np.ndarray:
    """
    Pipeline:
    - (optional) constant mean subtraction  (if mean_mode == 'const')
    - Butterworth HP/LP or band-pass using existing helpers: high_pass_1d / low_pass_1d / band_pass_1d
    - (optional) rolling baseline subtraction (if mean_mode == 'window', mean_param = window_seconds)
    - (optional) invert

    NOTE: Handles NaN-padded arrays (from multi-file concatenation) by filtering only the valid portion.
    """
    x = np.asarray(y, dtype=float)

    # Check for NaN values (from padded sweeps in multi-file loading)
    has_nan = np.isnan(x).any()

    if has_nan:
        # Find the valid (non-NaN) portion
        valid_mask = ~np.isnan(x)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # All NaN - return as is
            return x

        # Get the contiguous valid region (assuming NaN padding is at the end)
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]

        # Extract valid portion
        x_valid = x[first_valid:last_valid + 1]

        # Apply filters to valid portion only
        x_filtered = _apply_filters_to_valid(
            x_valid, sr_hz, use_low, low_hz, use_high, high_hz,
            use_mean, mean_param, use_inv, order, mean_mode
        )

        # Reconstruct full array with NaN padding
        result = np.full_like(x, np.nan)
        result[first_valid:last_valid + 1] = x_filtered
        return result
    else:
        # No NaN values - apply filters normally
        return _apply_filters_to_valid(
            x, sr_hz, use_low, low_hz, use_high, high_hz,
            use_mean, mean_param, use_inv, order, mean_mode
        )


def _apply_filters_to_valid(
    x: np.ndarray,
    sr_hz: float,
    use_low: bool, low_hz: float | None,
    use_high: bool, high_hz: float | None,
    use_mean: bool, mean_param: float | None,
    use_inv: bool,
    order: int,
    mean_mode: str
) -> np.ndarray:
    """Helper function to apply filters to valid (non-NaN) data."""
    # 1) optional constant mean subtraction
    if use_mean and mean_param is not None and mean_mode == "const":
        x = mean_subtract(x, float(mean_param))

    # 2) HP/LP/BP
    has_lp = use_low  and (low_hz  is not None)
    has_hp = use_high and (high_hz is not None)

    if has_lp and has_hp:
        # Band-pass when HP cutoff (high_hz) < LP cutoff (low_hz)
        if float(high_hz) < float(low_hz):
            x = band_pass_1d(x, sr_hz, low_cut_hz=float(high_hz), high_cut_hz=float(low_hz), order=order)
        else:
            # Graceful fallback if user enters overlapping/invalid cutoffs
            x = high_pass_1d(x, sr_hz, cutoff_hz=float(high_hz), order=order)
            x = low_pass_1d(x,  sr_hz, cutoff_hz=float(low_hz),  order=order)
    elif has_hp:
        x = high_pass_1d(x, sr_hz, cutoff_hz=float(high_hz), order=order)
    elif has_lp:
        x = low_pass_1d(x,  sr_hz, cutoff_hz=float(low_hz),  order=order)

    # 3) optional rolling baseline subtraction (window in seconds)
    if use_mean and mean_param is not None and mean_mode == "window":
        x = rolling_mean_subtract_1d(x, sr_hz, window_seconds=float(mean_param))

    # 4) optional invert
    if use_inv:
        x = invert(x)

    return x
