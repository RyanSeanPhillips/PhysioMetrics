"""
Threshold crossing event detector.

Detects events based on signal crossing a threshold value,
with support for direction selection and hysteresis.
"""

from typing import List, Tuple
import numpy as np

from .base import EventDetector, DetectorRegistry, ParamSpec, ParamType


@DetectorRegistry.register
class ThresholdCrossingDetector(EventDetector):
    """
    Detect events based on threshold crossings.

    Features:
    - Configurable threshold level
    - Direction selection (rising, falling, both)
    - Hysteresis to prevent chatter around threshold
    - Minimum duration filtering
    - Minimum gap / event merging
    """

    name = "Threshold Crossing"
    description = "Detect events when signal crosses a threshold value"

    def get_param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="threshold",
                label="Threshold",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=-1000.0,
                max_value=1000.0,
                step=0.1,
                unit="V",
                tooltip=(
                    "Signal level that triggers event detection.\n\n"
                    "For RISING edge: event starts when signal goes ABOVE this value.\n"
                    "For FALLING edge: event starts when signal goes BELOW this value.\n\n"
                    "Tip: Drag the orange line on the preview plot to adjust visually."
                )
            ),
            ParamSpec(
                name="direction",
                label="Direction",
                param_type=ParamType.CHOICE,
                default="rising",
                choices=[
                    ("Rising Edge", "rising"),
                    ("Falling Edge", "falling"),
                    ("Both Edges", "both"),
                ],
                tooltip=(
                    "Which signal direction triggers an event.\n\n"
                    "• Rising Edge: Detect when signal increases above threshold\n"
                    "• Falling Edge: Detect when signal decreases below threshold\n"
                    "• Both Edges: Detect both rising and falling crossings"
                )
            ),
            ParamSpec(
                name="hysteresis",
                label="Hysteresis",
                param_type=ParamType.FLOAT,
                default=0.05,
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                unit="V",
                tooltip=(
                    "Prevents false triggers when signal oscillates around threshold.\n\n"
                    "Creates a buffer zone between ON and OFF thresholds:\n"
                    "• Rising: ON at threshold, OFF at (threshold - hysteresis)\n"
                    "• Falling: ON at threshold, OFF at (threshold + hysteresis)\n\n"
                    "Example: threshold=0.5V, hysteresis=0.1V\n"
                    "→ Turns ON at 0.5V, won't turn OFF until below 0.4V\n\n"
                    "Tip: Set to ~10-20% of threshold for noisy signals."
                )
            ),
            ParamSpec(
                name="min_duration",
                label="Min Duration",
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                max_value=60.0,
                step=0.05,
                unit="s",
                tooltip=(
                    "Minimum event duration - shorter events are discarded.\n\n"
                    "Use this to filter out brief noise spikes that cross the\n"
                    "threshold but don't represent real events.\n\n"
                    "• 0.0s = Keep all events regardless of duration\n"
                    "• 0.2s = Ignore events shorter than 200ms\n"
                    "• 1.0s = Only keep events lasting at least 1 second\n\n"
                    "Tip: Set based on the shortest real event you expect."
                )
            ),
            ParamSpec(
                name="min_gap",
                label="Min Gap",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.0,
                max_value=60.0,
                step=0.1,
                unit="s",
                tooltip=(
                    "Minimum time between events - closer events are merged.\n\n"
                    "If two events occur within this time window, they are\n"
                    "combined into a single event spanning both.\n\n"
                    "• 0.0s = Never merge events\n"
                    "• 1.0s = Merge events less than 1 second apart\n"
                    "• 5.0s = Only separate events if >5 seconds apart\n\n"
                    "Tip: Increase this if detecting too many events.\n"
                    "Set based on how far apart real events should be."
                )
            ),
        ]

    def _detect_raw(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float
    ) -> List[Tuple[float, float]]:
        """
        Detect threshold crossings in the signal.

        Returns list of (start_time, end_time) for each detected event.
        """
        threshold = self.get_param("threshold")
        direction = self.get_param("direction")
        hysteresis = self.get_param("hysteresis")

        # Calculate on/off thresholds with hysteresis
        if direction == "rising":
            thresh_on = threshold
            thresh_off = threshold - hysteresis
        elif direction == "falling":
            thresh_on = threshold
            thresh_off = threshold + hysteresis
        else:  # both
            thresh_on = threshold
            thresh_off = threshold  # No hysteresis for "both" mode

        events = []

        if direction == "rising":
            events = self._detect_rising(signal, time, thresh_on, thresh_off)
        elif direction == "falling":
            events = self._detect_falling(signal, time, thresh_on, thresh_off)
        else:  # both
            rising = self._detect_rising(signal, time, thresh_on, thresh_off)
            falling = self._detect_falling(signal, time, thresh_on, thresh_off)
            events = sorted(rising + falling, key=lambda x: x[0])

        return events

    def _detect_rising(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        thresh_on: float,
        thresh_off: float
    ) -> List[Tuple[float, float]]:
        """Detect rising edge events (signal goes above threshold)."""
        events = []
        in_event = False
        event_start = 0.0

        for i in range(len(signal)):
            if not in_event and signal[i] >= thresh_on:
                # Start of event
                in_event = True
                event_start = time[i]
            elif in_event and signal[i] < thresh_off:
                # End of event
                in_event = False
                events.append((event_start, time[i]))

        # Handle event that extends to end of signal
        if in_event:
            events.append((event_start, time[-1]))

        return events

    def _detect_falling(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        thresh_on: float,
        thresh_off: float
    ) -> List[Tuple[float, float]]:
        """Detect falling edge events (signal goes below threshold)."""
        events = []
        in_event = False
        event_start = 0.0

        for i in range(len(signal)):
            if not in_event and signal[i] <= thresh_on:
                # Start of event (signal dropped below threshold)
                in_event = True
                event_start = time[i]
            elif in_event and signal[i] > thresh_off:
                # End of event (signal rose back above)
                in_event = False
                events.append((event_start, time[i]))

        # Handle event that extends to end of signal
        if in_event:
            events.append((event_start, time[-1]))

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
            min_duration = self.get_param("min_duration")
        if min_gap is None:
            min_gap = self.get_param("min_gap")

        return super().detect(signal, time, sample_rate, min_duration, min_gap)
