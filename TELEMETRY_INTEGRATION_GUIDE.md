# Telemetry Integration Guide

Complete guide for tracking user behavior, timing data, and crashes in PlethApp.

## Table of Contents
1. [Available Tracking Functions](#available-tracking-functions)
2. [Integration Examples](#integration-examples)
3. [Viewing Data in GA4](#viewing-data-in-ga4)
4. [Privacy Considerations](#privacy-considerations)

---

## Available Tracking Functions

### 1. Button/Feature Clicks
```python
from core import telemetry

telemetry.log_button_click('detect_peaks', threshold=0.5)
telemetry.log_button_click('apply_filter', filter_type='butterworth', cutoff=2.0)
telemetry.log_button_click('run_gmm', num_components=2)
```

### 2. Timing Data
```python
import time
start = time.time()
# ... perform operation ...
duration = time.time() - start

telemetry.log_timing('peak_detection', duration, num_breaths=247)
telemetry.log_timing('file_load', duration, file_size_mb=15, num_sweeps=10)
telemetry.log_timing('channel_selection_to_save', duration)
```

### 3. Manual Edits
```python
telemetry.log_edit('add_peak', num_peaks_after=248)
telemetry.log_edit('delete_peak', num_peaks_after=246)
telemetry.log_edit('move_peak', distance_moved_samples=50)
telemetry.log_edit('mark_sniff')
```

### 4. Breath Statistics
```python
telemetry.log_breath_statistics(
    num_breaths=247,
    mean_frequency=1.2,
    regularity_score=0.85,
    eupnea_percentage=75,
    apnea_count=3
)
```

### 5. Crash Tracking
```python
try:
    # ... risky operation ...
except Exception as e:
    telemetry.log_crash(
        error_message=f"{type(e).__name__} in peak detection",
        operation='detect_peaks',
        num_breaths=state.num_breaths if state.breaths else 0
    )
    # ... handle error ...
```

### 6. Existing Functions
```python
# Already implemented:
telemetry.log_file_loaded('abf', num_sweeps=10, num_breaths=247)
telemetry.log_feature_used('gmm_clustering')
telemetry.log_export('breaths_csv')
```

---

## Integration Examples

### Example 1: Peak Detection Button

**Location:** `main.py` - in the peak detection button handler

```python
def on_detect_peaks_clicked(self):
    """Handle peak detection button click."""
    # Log button click
    telemetry.log_button_click('detect_peaks',
                               threshold=self.threshold_spinbox.value())

    try:
        # Start timing
        import time
        start_time = time.time()

        # Detect peaks
        peaks = detect_peaks(self.state.data, threshold=...)

        # Log timing
        duration = time.time() - start_time
        telemetry.log_timing('peak_detection', duration,
                            num_peaks=len(peaks),
                            threshold=self.threshold_spinbox.value())

        # Log statistics
        if peaks:
            telemetry.log_breath_statistics(
                num_breaths=len(peaks),
                mean_frequency=calculate_mean_frequency(peaks)
            )

    except Exception as e:
        telemetry.log_crash(f"{type(e).__name__} in peak detection",
                           threshold=self.threshold_spinbox.value())
        raise
```

### Example 2: Manual Edit Operations

**Location:** `editing/editing_modes.py` or wherever edit handlers are

```python
def add_peak_at_cursor(self, x_position):
    """Add a peak at the cursor position."""
    # Add the peak
    self.state.breaths.append(x_position)
    self.state.breaths.sort()

    # Log the edit
    telemetry.log_edit('add_peak',
                      num_peaks_after=len(self.state.breaths),
                      x_position=int(x_position))  # Anonymized position

    # Redraw
    self.redraw()

def delete_peak(self, peak_index):
    """Delete a peak."""
    self.state.breaths.pop(peak_index)

    telemetry.log_edit('delete_peak',
                      num_peaks_after=len(self.state.breaths))

    self.redraw()
```

### Example 3: Filter Application

**Location:** `main.py` - filter button handler

```python
def on_apply_filter_clicked(self):
    """Apply Butterworth filter."""
    import time

    # Log button click with parameters
    telemetry.log_button_click('apply_filter',
                               filter_type='butterworth',
                               highpass=self.hp_spinbox.value(),
                               lowpass=self.lp_spinbox.value(),
                               order=self.order_spinbox.value())

    # Time the operation
    start = time.time()

    # Apply filter
    self.state.filtered_data = apply_butterworth_filter(...)

    # Log timing
    duration = time.time() - start
    telemetry.log_timing('apply_filter', duration,
                        data_points=len(self.state.data))
```

### Example 4: GMM Clustering

**Location:** `dialogs/gmm_clustering_dialog.py`

```python
def run_gmm_clustering(self):
    """Run GMM clustering."""
    import time

    # Log button click
    telemetry.log_button_click('run_gmm',
                               n_components=self.n_components,
                               feature_set=self.selected_features)

    # Time the operation
    start = time.time()

    try:
        # Run GMM
        labels, scores = fit_gmm(...)

        # Log timing and results
        duration = time.time() - start
        telemetry.log_timing('gmm_clustering', duration,
                            n_components=self.n_components,
                            num_breaths=len(labels),
                            silhouette_score=scores['silhouette'])

        # Log feature usage
        telemetry.log_feature_used('gmm_clustering')

    except Exception as e:
        telemetry.log_crash(f"GMM clustering failed: {type(e).__name__}",
                           n_components=self.n_components,
                           num_breaths=len(self.breaths))
        raise
```

### Example 5: Export Operations

**Location:** `export/export_manager.py` or wherever exports happen

```python
def export_to_csv(self, filename):
    """Export breath data to CSV."""
    import time

    # Log button click
    telemetry.log_button_click('export_csv',
                               num_rows=len(self.state.breaths))

    # Time the export
    start = time.time()

    try:
        # Perform export
        df.to_csv(filename)

        # Log export
        duration = time.time() - start
        telemetry.log_export('breaths_csv')
        telemetry.log_timing('export_csv', duration,
                            num_rows=len(self.state.breaths))

    except Exception as e:
        telemetry.log_crash(f"Export failed: {type(e).__name__}",
                           export_type='csv',
                           num_rows=len(self.state.breaths))
        raise
```

### Example 6: Channel Selection to Save Workflow

**Track time from channel selection to first save:**

```python
class PlethApp:
    def __init__(self):
        self.channel_selection_time = None

    def on_channel_selected(self, channel_name):
        """Called when user selects analysis channel."""
        import time
        self.channel_selection_time = time.time()

        telemetry.log_button_click('select_channel',
                                   channel=channel_name)

    def on_save_data(self):
        """Called when user saves/exports data."""
        import time

        # Calculate time from channel selection to save
        if self.channel_selection_time:
            duration = time.time() - self.channel_selection_time
            telemetry.log_timing('channel_selection_to_save', duration)
            self.channel_selection_time = None  # Reset

        telemetry.log_button_click('save_data')
```

### Example 7: Global Exception Handler

**Location:** `main.py` or `run_debug.py` - for catching unhandled crashes

```python
import sys
from core import telemetry

def exception_hook(exctype, value, tb):
    """Global exception handler to log crashes."""
    import traceback

    # Log crash to telemetry
    telemetry.log_crash(
        error_message=f"{exctype.__name__}: {str(value)[:100]}",
        traceback_lines=len(traceback.extract_tb(tb))
    )

    # Call default handler
    sys.__excepthook__(exctype, value, tb)

# Install global exception handler
sys.excepthook = exception_hook
```

---

## Viewing Data in GA4

### 1. Geographic Location
- **Location:** Reports → User → Demographics → Geographic
- Shows: Country, Region, City of users (derived from IP, anonymous)

### 2. App Version Usage
- **Location:** Reports → Tech → App version
- Shows: Which versions of PlethApp are being used
- **Note:** Version is already sent with every event automatically

### 3. Button Click Tracking
- **Location:** Reports → Engagement → Events
- Filter by: `event_name = button_click`
- Shows: Which buttons/features are used most
- **Parameters:** Click event name to see button names, parameters

### 4. Timing Data
- **Location:** Reports → Engagement → Events
- Filter by: `event_name = timing`
- Shows: Average duration of operations
- **Custom Report:**
  - Go to: Explore → Create new exploration
  - Dimensions: `operation` parameter
  - Metrics: Average `duration_seconds`
  - Shows: Which operations are slow for users

### 5. Edit Statistics
- **Location:** Reports → Engagement → Events
- Filter by: `event_name = manual_edit`
- Shows: How much manual editing users do
- **Session Summary:** Check `session_end` event → `edits_made` parameter

### 6. Breath Statistics
- **Location:** Reports → Engagement → Events
- Filter by: `event_name = breath_statistics`
- **Custom Report:**
  - Metric: Average `num_breaths`
  - Metric: Average `mean_frequency_hz`
  - Metric: Average `regularity_score`

### 7. Crash Tracking
- **Location:** Reports → Engagement → Events
- Filter by: `event_name = crash`
- Shows: Error types, frequency, last action before crash
- **Parameters:**
  - `error_type`: Type of error
  - `last_action`: Last button/feature used before crash

### 8. Workflow Timing (Channel Selection to Save)
- **Location:** Explore → Create custom report
- Filter events: `event_name = timing` AND `operation = channel_selection_to_save`
- Metric: Average `duration_seconds`
- Shows: How long users take from selecting channel to saving

### 9. Creating Custom Dashboards

**Example: Operations Performance Dashboard**

1. Go to: **Explore** → **Create new exploration**
2. **Technique:** Free form
3. **Dimensions:**
   - `operation` (from timing events)
   - `event_date`
4. **Metrics:**
   - Event count
   - Average `duration_seconds`
   - Min/Max `duration_seconds`
5. **Visualization:** Table or Line chart
6. **Result:** See which operations are slow and getting slower/faster over time

**Example: User Engagement Dashboard**

1. **Dimensions:** `event_name`, `button` parameter
2. **Metrics:** Event count, Users
3. **Filters:** Last 30 days
4. **Result:** Most used features, user retention

---

## Privacy Considerations

### ✅ What We Track (Anonymous & Safe)
- Button clicks and feature usage
- Operation timing (duration only)
- Number of breaths detected
- Edit counts
- Error types (no stack traces in GA4)
- Anonymous user ID (random UUID)
- Geographic location (country/region from IP)
- App version

### ❌ What We NEVER Track
- ❌ File names or paths
- ❌ Animal metadata (strain, virus, etc.)
- ❌ Actual breathing data values
- ❌ User's name, email, institution
- ❌ Computer name or username
- ❌ IP addresses (stored by GA4)
- ❌ Specific x/y coordinates (only anonymized positions)

### Sanitization Examples

**❌ BAD (contains PII):**
```python
telemetry.log_timing('file_load', duration,
                     filename='/Users/john/experiment_mouse_123.abf')
```

**✅ GOOD (anonymous):**
```python
telemetry.log_timing('file_load', duration,
                     file_size_mb=15.2,
                     num_sweeps=10)
```

**❌ BAD (contains experimental data):**
```python
telemetry.log_breath_statistics(breath_frequencies=[1.2, 1.3, 1.1, ...])
```

**✅ GOOD (aggregate only):**
```python
telemetry.log_breath_statistics(num_breaths=247,
                               mean_frequency=1.2,
                               regularity_score=0.85)
```

---

## Implementation Checklist

### High Priority (Core Metrics)
- [ ] Log all button clicks (`log_button_click`)
- [ ] Track peak detection timing (`log_timing`)
- [ ] Track file load timing (`log_timing`)
- [ ] Track all manual edits (`log_edit`)
- [ ] Track breath statistics after analysis (`log_breath_statistics`)
- [ ] Track channel selection to save workflow (`log_timing`)
- [ ] Add global exception handler (`log_crash`)

### Medium Priority (Enhanced Insights)
- [ ] Track GMM clustering timing and results
- [ ] Track filter application
- [ ] Track export operations timing
- [ ] Track spectral analysis usage
- [ ] Track session save/load

### Low Priority (Nice to Have)
- [ ] Track navigation patterns (sweep changes)
- [ ] Track zoom level changes
- [ ] Track window resize/layout changes

---

## Testing Telemetry

### Validation Test
```bash
python test_ga4_validation.py
```
Should show: `✅ SUCCESS! Event is VALID`

### Check Events in GA4
1. Go to: **Reports → Realtime**
2. Perform actions in PlethApp
3. Events should appear within 10-30 seconds in Realtime view
4. Full processing takes 24-48 hours for historical reports

---

## Questions?

See `TELEMETRY_SETUP.md` for initial setup instructions.

For GA4 dashboard help: https://support.google.com/analytics/
