"""
Hargreaves thermal sensitivity detector.

Implements the Hargreaves onset detection algorithm which finds the true
onset point where the signal first leaves the baseline noise level,
rather than just the threshold crossing point.
"""

from typing import List, Tuple, Optional
import numpy as np

from .base import EventDetector, DetectorRegistry, ParamSpec, ParamType


class HargreavesDetector(EventDetector):
    """
    Hargreaves thermal sensitivity detector.

    This detector is designed for thermal sensitivity testing where the true
    onset of a response may occur before the signal crosses a detection threshold.

    Algorithm:
    1. Find initial threshold crossings (rising edges)
    2. For each crossing, search backward to find where signal first leaves noise level
    3. Calculate baseline noise from early portion of search window
    4. Find transition point using 3-sigma above baseline

    This provides more accurate onset times than simple threshold detection,
    especially for gradually rising signals.
    """

    name = "Hargreaves Thermal"
    description = "Detects onset where signal first leaves baseline noise (for thermal sensitivity)"

    def __init__(self):
        """Initialize the detector with default parameters."""
        super().__init__()
        self._params = {
            'threshold': 0.5,
            'lookback_time': 1.0,  # seconds to search backward for onset
            'sigma_multiplier': 3.0,  # noise threshold = baseline + N*sigma
            'baseline_fraction': 0.2,  # fraction of lookback window for baseline
            'direction': 'rising',  # rising, falling, or both
            'min_duration': 0.2,  # filter out brief noise spikes
            'min_gap': 1.0,  # prevent detecting too many events in noisy data
            # Withdrawal detection parameters
            'min_withdrawal_delay': 1.0,  # min time after onset to look for withdrawal
            'max_withdrawal_time': 8.0,  # max time after onset to look for withdrawal
            'withdrawal_smooth': 51,  # smoothing window for derivative calculation
            'withdrawal_deriv_sigma': 1.0,  # derivative must exceed N*sigma of baseline derivative
        }

    def get_param_specs(self) -> List[ParamSpec]:
        """Return parameter specifications for UI generation."""
        return [
            ParamSpec(
                name='threshold',
                label='Detection Threshold',
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=-100.0,
                max_value=100.0,
                step=0.1,
                unit='V',
                tooltip=(
                    "Primary threshold for initial event detection.\n\n"
                    "The algorithm first finds where the signal crosses this\n"
                    "threshold, then searches BACKWARD to find the true onset\n"
                    "where the signal first left baseline noise.\n\n"
                    "This threshold should be set high enough to reliably\n"
                    "detect your events, even if it's past the true onset.\n\n"
                    "Tip: Drag the orange line on the preview plot to adjust."
                ),
            ),
            ParamSpec(
                name='lookback_time',
                label='Lookback Window',
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                unit='s',
                tooltip=(
                    "How far back (in seconds) to search for true onset.\n\n"
                    "After finding a threshold crossing, the algorithm searches\n"
                    "backward through this time window to find where the signal\n"
                    "first left the baseline noise level.\n\n"
                    "• 1.0s = Search up to 1 second before threshold crossing\n"
                    "• 2.0s = Search up to 2 seconds before\n\n"
                    "Tip: Set based on how gradual your signal rises.\n"
                    "Shown as BLUE region in preview."
                ),
            ),
            ParamSpec(
                name='sigma_multiplier',
                label='Noise Sigma',
                param_type=ParamType.FLOAT,
                default=3.0,
                min_value=1.0,
                max_value=6.0,
                step=0.5,
                tooltip=(
                    "How many standard deviations above baseline = onset.\n\n"
                    "The onset is detected where signal first exceeds:\n"
                    "  baseline_mean + (sigma × baseline_std)\n\n"
                    "• 2σ = 95% confidence (more sensitive, earlier onset)\n"
                    "• 3σ = 99.7% confidence (default, balanced)\n"
                    "• 4σ = 99.99% confidence (less sensitive, later onset)\n\n"
                    "Tip: Lower values detect onset earlier but may be\n"
                    "triggered by noise. Higher values are more conservative."
                ),
            ),
            ParamSpec(
                name='baseline_fraction',
                label='Baseline Fraction',
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.05,
                max_value=0.5,
                step=0.05,
                tooltip=(
                    "Fraction of lookback window used for baseline calculation.\n\n"
                    "The earliest portion of the lookback window is assumed to\n"
                    "be baseline (before any response started). This is used to\n"
                    "calculate the noise level (mean and standard deviation).\n\n"
                    "• 0.2 = Use first 20% of lookback window as baseline\n"
                    "• 0.1 = Use first 10% (if responses are slow)\n"
                    "• 0.3 = Use first 30% (if baseline is stable)\n\n"
                    "Shown as YELLOW region in preview."
                ),
            ),
            ParamSpec(
                name='direction',
                label='Signal Direction',
                param_type=ParamType.CHOICE,
                default='rising',
                choices=[
                    ('Rising (positive)', 'rising'),
                    ('Falling (negative)', 'falling'),
                    ('Both', 'both'),
                ],
                tooltip=(
                    "Which signal direction triggers an event.\n\n"
                    "• Rising: Detect responses that increase (e.g., paw withdrawal)\n"
                    "• Falling: Detect responses that decrease\n"
                    "• Both: Detect both rising and falling responses"
                ),
            ),
            ParamSpec(
                name='min_duration',
                label='Min Duration',
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                max_value=60.0,
                step=0.05,
                unit='s',
                tooltip=(
                    "Minimum event duration - shorter events are discarded.\n\n"
                    "Use this to filter out brief noise spikes that cross the\n"
                    "threshold but don't represent real responses.\n\n"
                    "• 0.0s = Keep all events regardless of duration\n"
                    "• 0.2s = Ignore events shorter than 200ms\n"
                    "• 1.0s = Only keep events lasting at least 1 second\n\n"
                    "Tip: Set based on the shortest real response you expect."
                ),
            ),
            ParamSpec(
                name='min_gap',
                label='Min Gap',
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.0,
                max_value=60.0,
                step=0.1,
                unit='s',
                tooltip=(
                    "Minimum time between events - closer events are merged.\n\n"
                    "If two events occur within this time window, they are\n"
                    "combined into a single event spanning both.\n\n"
                    "• 0.0s = Never merge events\n"
                    "• 1.0s = Merge events less than 1 second apart\n"
                    "• 5.0s = Only separate events if >5 seconds apart\n\n"
                    "Tip: Increase this if detecting too many events.\n"
                    "For thermal tests, trials are usually >30s apart."
                ),
            ),
            # Withdrawal detection parameters
            ParamSpec(
                name='min_withdrawal_delay',
                label='Min Withdrawal Delay',
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                unit='s',
                tooltip=(
                    "Minimum time after onset before looking for withdrawal.\n\n"
                    "The withdrawal typically occurs after the peak of the\n"
                    "heat response. This delay prevents detecting the initial\n"
                    "rising phase as a withdrawal.\n\n"
                    "• 1.0s = Start looking for withdrawal 1s after onset\n"
                    "• 2.0s = Wait longer for slower responses"
                ),
            ),
            ParamSpec(
                name='max_withdrawal_time',
                label='Max Withdrawal Time',
                param_type=ParamType.FLOAT,
                default=8.0,
                min_value=2.0,
                max_value=60.0,
                step=1.0,
                unit='s',
                tooltip=(
                    "Maximum time after onset to search for withdrawal.\n\n"
                    "If no withdrawal is detected within this window,\n"
                    "the event end will be placed at this time.\n\n"
                    "• 8s = Default cutoff for Hargreaves test\n"
                    "• 20s = Extended search window"
                ),
            ),
            ParamSpec(
                name='withdrawal_smooth',
                label='Withdrawal Smoothing',
                param_type=ParamType.INT,
                default=51,
                min_value=11,
                max_value=201,
                step=10,
                tooltip=(
                    "Smoothing window for derivative calculation.\n\n"
                    "The withdrawal is detected by finding a major deflection\n"
                    "in the smoothed derivative. Higher values = smoother.\n\n"
                    "• 21-31: Less smoothing, more sensitive\n"
                    "• 51: Default (balanced)\n"
                    "• 101-151: Heavy smoothing for noisy signals"
                ),
            ),
            ParamSpec(
                name='withdrawal_deriv_sigma',
                label='Withdrawal Sensitivity',
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.5,
                max_value=6.0,
                step=0.5,
                tooltip=(
                    "Sensitivity for withdrawal detection (in sigma).\n\n"
                    "The derivative must exceed N standard deviations from\n"
                    "the baseline derivative to be considered a withdrawal.\n\n"
                    "• 1σ = Default (sensitive, detects smaller movements)\n"
                    "• 2σ = Moderate sensitivity\n"
                    "• 3σ = Less sensitive (only large withdrawals)"
                ),
            ),
        ]

    def _detect_raw(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float,
    ) -> List[Tuple[float, float]]:
        """
        Detect events using Hargreaves onset algorithm.

        Args:
            signal: Signal data
            time: Time array
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_time, end_time) tuples
        """
        threshold = self._params['threshold']
        lookback_time = self._params['lookback_time']
        sigma_mult = self._params['sigma_multiplier']
        baseline_frac = self._params['baseline_fraction']
        direction = self._params['direction']

        events = []

        # Process rising edges
        if direction in ('rising', 'both'):
            rising_events = self._detect_direction(
                signal, time, sample_rate,
                threshold, lookback_time, sigma_mult, baseline_frac,
                is_rising=True
            )
            events.extend(rising_events)

        # Process falling edges
        if direction in ('falling', 'both'):
            falling_events = self._detect_direction(
                -signal, time, sample_rate,  # Invert signal for falling detection
                -threshold, lookback_time, sigma_mult, baseline_frac,
                is_rising=True  # After inversion, falling becomes rising
            )
            events.extend(falling_events)

        # Sort by start time
        events.sort(key=lambda x: x[0])

        return events

    def detect(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float,
        min_duration: float = None,
        min_gap: float = None,
    ):
        """
        Detect events with filtering.

        Overrides base class to use instance parameters if not specified.
        """
        # Use instance params if not explicitly provided
        if min_duration is None:
            min_duration = self.get_param('min_duration') or 0.0
        if min_gap is None:
            min_gap = self.get_param('min_gap') or 0.0

        return super().detect(signal, time, sample_rate, min_duration, min_gap)

    def _detect_direction(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float,
        threshold: float,
        lookback_time: float,
        sigma_mult: float,
        baseline_frac: float,
        is_rising: bool,
    ) -> List[Tuple[float, float]]:
        """Detect events in one direction."""
        events = []

        # Find threshold crossings
        above = signal > threshold
        crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1

        if len(crossings) == 0:
            return events

        lookback_samples = int(lookback_time * sample_rate)

        # Get withdrawal parameters
        min_withdrawal_delay = self._params.get('min_withdrawal_delay', 1.0)
        max_withdrawal_time = self._params.get('max_withdrawal_time', 20.0)
        withdrawal_smooth = self._params.get('withdrawal_smooth', 51)
        withdrawal_deriv_sigma = self._params.get('withdrawal_deriv_sigma', 3.0)

        for crossing_idx in crossings:
            # Find true onset using backward search
            onset_idx = self._find_true_onset(
                signal, crossing_idx, lookback_samples,
                threshold, sigma_mult, baseline_frac
            )

            # Find withdrawal using derivative-based detection
            withdrawal_idx = self._find_withdrawal(
                signal, time, sample_rate, onset_idx,
                min_withdrawal_delay, max_withdrawal_time,
                withdrawal_smooth, withdrawal_deriv_sigma
            )

            if withdrawal_idx is not None and withdrawal_idx > onset_idx:
                start_time = time[onset_idx]
                end_time = time[withdrawal_idx]
                events.append((start_time, end_time))

        return events

    def _find_true_onset(
        self,
        signal: np.ndarray,
        crossing_idx: int,
        lookback_samples: int,
        threshold: float,
        sigma_mult: float,
        baseline_frac: float,
    ) -> int:
        """
        Find true onset by searching backward from threshold crossing.

        Args:
            signal: Signal data
            crossing_idx: Index of threshold crossing
            lookback_samples: Number of samples to search backward
            threshold: Detection threshold
            sigma_mult: Sigma multiplier for noise threshold
            baseline_frac: Fraction of window to use for baseline

        Returns:
            Index of true onset
        """
        # Define search window
        search_start = max(0, crossing_idx - lookback_samples)
        search_end = crossing_idx

        if search_end <= search_start:
            return crossing_idx  # Fallback

        signal_window = signal[search_start:search_end]
        window_len = len(signal_window)

        # Estimate baseline noise level
        baseline_samples = max(10, min(int(baseline_frac * window_len), window_len // 2))

        if baseline_samples >= 10:
            baseline_region = signal_window[:baseline_samples]
            baseline_mean = np.mean(baseline_region)
            baseline_std = np.std(baseline_region)

            if baseline_std > 0:
                noise_threshold = baseline_mean + sigma_mult * baseline_std
            else:
                # No variance - use small offset from mean
                noise_threshold = baseline_mean + 0.01 * abs(baseline_mean)
        else:
            # Fallback: use MAD-based estimate
            baseline_mean = np.median(signal_window)
            mad = np.median(np.abs(signal_window - baseline_mean))
            noise_threshold = baseline_mean + sigma_mult * mad * 1.4826

        # Search backward from crossing to find where signal first leaves noise
        onset_idx_abs = None

        for i in range(window_len - 1, -1, -1):
            idx_abs = search_start + i

            if signal[idx_abs] > noise_threshold:
                # Signal is elevated - keep track of this point
                onset_idx_abs = idx_abs
            else:
                # Signal is at noise level
                if onset_idx_abs is not None:
                    # We've found the transition from baseline to elevated
                    break

        # Validate and return onset
        if onset_idx_abs is not None and onset_idx_abs < crossing_idx:
            if signal[onset_idx_abs] < threshold:
                return onset_idx_abs

        # Fallback: find minimum in search window
        min_idx_rel = np.argmin(signal_window)
        min_idx_abs = search_start + min_idx_rel

        if signal[min_idx_abs] < threshold:
            return min_idx_abs

        return crossing_idx  # Final fallback

    def _find_withdrawal(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float,
        onset_idx: int,
        min_delay: float,
        max_time: float,
        smooth_window: int,
        deriv_sigma: float,
    ) -> Optional[int]:
        """
        Find withdrawal point using derivative-based detection.

        The withdrawal is detected as the first major deflection in the
        derivative AFTER the signal has peaked and the derivative has
        decayed back toward zero.

        Algorithm:
        1. Calculate smoothed derivative for entire search region
        2. Find the signal peak (maximum)
        3. Wait for derivative to settle near zero (abs(deriv) < threshold)
        4. Then find first major deflection after that settling point
        5. Apply time constraints (min_delay, max_time)

        Args:
            signal: Signal data
            time: Time array
            sample_rate: Sample rate in Hz
            onset_idx: Index of detected onset
            min_delay: Minimum time after onset to start looking
            max_time: Maximum time after onset to search
            smooth_window: Smoothing window for derivative
            deriv_sigma: Number of sigma for derivative threshold

        Returns:
            Index of withdrawal, or None if not found
        """
        from scipy.signal import savgol_filter

        onset_time = time[onset_idx]

        # Calculate search bounds
        search_end_time = onset_time + max_time
        search_end_idx = min(np.searchsorted(time, search_end_time), len(signal) - 1)

        if onset_idx >= search_end_idx - 100:
            # Not enough data, use max_time as fallback
            return search_end_idx

        # Ensure smooth_window is odd and doesn't exceed signal length
        window = smooth_window
        if window % 2 == 0:
            window += 1
        search_len = search_end_idx - onset_idx
        if window >= search_len:
            window = search_len - 1 if search_len % 2 == 0 else search_len - 2
            if window < 5:
                return search_end_idx

        # Calculate smoothed derivative for the region from onset to search_end
        region_start = onset_idx
        region_end = search_end_idx
        region_signal = signal[region_start:region_end]
        region_time = time[region_start:region_end]
        dt = 1.0 / sample_rate

        try:
            derivative = savgol_filter(region_signal, window, polyorder=3, deriv=1, delta=dt)
        except Exception:
            derivative = np.gradient(region_signal, dt)

        # Find the signal peak within the search region
        peak_region_idx = np.argmax(region_signal)

        # Calculate baseline derivative statistics using the plateau region
        # Look at the middle portion where derivative should be near zero (just noise)
        mid_start = len(derivative) // 3
        mid_end = 2 * len(derivative) // 3
        if mid_end > mid_start + 20:
            mid_deriv = derivative[mid_start:mid_end]
            abs_mid_deriv = np.abs(mid_deriv)
            # Use 90th percentile as baseline - this captures noise level
            baseline_noise = np.percentile(abs_mid_deriv, 90)
        else:
            # Fallback
            abs_deriv = np.abs(derivative)
            baseline_noise = np.percentile(abs_deriv, 75)

        # Threshold for "settled" derivative - must be below noise level
        # Use a generous multiplier to account for noise fluctuations
        settle_threshold = baseline_noise * 1.5

        # Threshold for withdrawal detection - must be significantly above noise
        # A real withdrawal should have a derivative much larger than baseline noise
        withdrawal_threshold = baseline_noise * deriv_sigma

        # Start searching after the peak
        search_start = max(peak_region_idx, int(min_delay * sample_rate))

        # Phase 1: Wait for derivative to settle for a SUSTAINED period
        # (abs(deriv) must stay below settle_threshold for at least 0.5s)
        sustained_settle_samples = int(0.5 * sample_rate)  # 0.5 seconds
        settled_idx = None
        settle_count = 0

        for i in range(search_start, len(derivative)):
            if abs(derivative[i]) < settle_threshold:
                settle_count += 1
                if settle_count >= sustained_settle_samples:
                    # Derivative has been settled for sustained period
                    settled_idx = i
                    break
            else:
                # Reset counter if derivative exceeds threshold
                settle_count = 0

        if settled_idx is None:
            # Derivative never settled for sustained period, use max_time
            return search_end_idx

        # Phase 2: After sustained settling, find first major deflection
        for i in range(settled_idx, len(derivative)):
            if abs(derivative[i]) > withdrawal_threshold:
                # Found withdrawal - return the index in original signal
                withdrawal_idx = region_start + i
                # Verify it's after min_delay
                if time[withdrawal_idx] >= onset_time + min_delay:
                    return withdrawal_idx

        # No withdrawal detected within window, return max_time point
        return search_end_idx

    def _find_end(
        self,
        signal: np.ndarray,
        start_idx: int,
        threshold: float,
    ) -> Optional[int]:
        """
        Legacy method - find where signal returns below threshold.
        Kept for backwards compatibility.
        """
        for i in range(start_idx + 1, len(signal)):
            if signal[i] < threshold:
                return i
        return None


# Register the detector
DetectorRegistry.register(HargreavesDetector)
