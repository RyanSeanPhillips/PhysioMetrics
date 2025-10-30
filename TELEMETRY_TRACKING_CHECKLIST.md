# Comprehensive Telemetry Tracking Checklist

Everything that should be tracked in PlethApp for comprehensive usage analytics.

---

## ✅ Already Implemented

- [x] Session start/end with duration
- [x] File loaded (basic)
- [x] GMM clustering usage
- [x] Exports

---

## 📋 File Loading (Enhanced)

**Function:** `telemetry.log_file_loaded(file_type, num_sweeps, num_breaths, **extra)`

### Track:
- [x] File type (abf, smrx, edf)
- [x] Number of sweeps
- [ ] **File size (MB)**
- [ ] **Sampling rate (Hz)**
- [ ] **Recording duration (minutes)**
- [ ] **Number of channels available**
- [ ] **Selected channel name/index**
- [ ] **Time to load file** (use `log_timing('file_load', duration)`)

### Where to add:
```python
# In main.py - load_file() function
file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
duration_min = len(data) / sampling_rate / 60

telemetry.log_file_loaded(
    file_type=file_ext,
    num_sweeps=n_sweeps,
    file_size_mb=round(file_size_mb, 2),
    sampling_rate_hz=sampling_rate,
    duration_minutes=round(duration_min, 1),
    num_channels=len(channel_names),
    selected_channel=selected_channel_name
)
```

---

## 🔍 Peak Detection

**Functions:**
- `telemetry.log_peak_detection(method, num_peaks, **params)`
- `telemetry.log_timing('peak_detection', duration, **params)`

### Track:
- [ ] **Detection method** ('auto_threshold', 'manual_threshold')
- [ ] **Number of peaks detected**
- [ ] **Threshold value used**
- [ ] **Min distance parameter**
- [ ] **Detection duration** (how long it took)
- [ ] **Success/failure** (0 peaks = warning)

### Where to add:
```python
# In peak detection handler
import time
start = time.time()

try:
    peaks = detect_peaks(data, threshold=thresh, min_distance=min_dist)

    duration = time.time() - start

    # Log the detection
    telemetry.log_peak_detection(
        method='auto_threshold' if auto else 'manual_threshold',
        num_peaks=len(peaks),
        threshold=thresh,
        min_distance=min_dist
    )

    # Log timing
    telemetry.log_timing('peak_detection', duration,
                        num_peaks=len(peaks),
                        data_points=len(data))

    # Warning if no peaks
    if len(peaks) == 0:
        telemetry.log_warning('No peaks detected',
                             threshold=thresh,
                             data_range=data.max() - data.min())

except Exception as e:
    telemetry.log_crash(f"Peak detection failed: {type(e).__name__}",
                       threshold=thresh)
```

---

## 🎨 Filtering

**Function:** `telemetry.log_filter_applied(filter_type, **params)`

### Track:
- [ ] **Filter type** ('butterworth', 'notch', 'mean_subtract')
- [ ] **Highpass cutoff** (Hz)
- [ ] **Lowpass cutoff** (Hz)
- [ ] **Filter order**
- [ ] **Data size** (number of points)
- [ ] **Filter application duration**

### Where to add:
```python
# In filter button handler
import time
start = time.time()

filtered_data = apply_butterworth_filter(data, hp, lp, order, fs)

duration = time.time() - start

telemetry.log_filter_applied(
    filter_type='butterworth',
    highpass_hz=hp,
    lowpass_hz=lp,
    order=order,
    sampling_rate_hz=fs,
    data_points=len(data)
)

telemetry.log_timing('filter_apply', duration,
                    data_points=len(data))
```

---

## ✏️ Manual Editing

**Function:** `telemetry.log_edit(edit_type, **params)`

### Track:
- [ ] **Edit type** ('add_peak', 'delete_peak', 'move_peak', 'mark_sniff')
- [ ] **Number of peaks after edit**
- [ ] **Total edits in session** (already tracked)

### Where to add:
```python
# In editing mode handlers
def on_shift_click_add_peak(self, x_pos):
    """Add peak with Shift+Click."""
    self.state.breaths.append(x_pos)
    self.state.breaths.sort()

    # Log the edit
    telemetry.log_edit('add_peak',
                      num_peaks_after=len(self.state.breaths))

    # Also log keyboard shortcut usage
    telemetry.log_keyboard_shortcut('shift_click_add_peak')

    self.redraw()

def on_ctrl_click_delete_peak(self, peak_index):
    """Delete peak with Ctrl+Click."""
    self.state.breaths.pop(peak_index)

    telemetry.log_edit('delete_peak',
                      num_peaks_after=len(self.state.breaths))

    telemetry.log_keyboard_shortcut('ctrl_click_delete_peak')

    self.redraw()
```

---

## 🧬 GMM Clustering

**Functions:**
- `telemetry.log_button_click('run_gmm', **params)`
- `telemetry.log_timing('gmm_clustering', duration, **params)`
- `telemetry.log_feature_used('gmm_clustering')`

### Track:
- [ ] **Number of components** (usually 2)
- [ ] **Features used** (list of feature names)
- [ ] **Number of breaths clustered**
- [ ] **Clustering duration**
- [ ] **Silhouette score** (quality metric)
- [ ] **Cluster sizes** (how many in each cluster)
- [ ] **User accepted/rejected results**

### Where to add:
```python
# In GMM dialog
import time
start = time.time()

labels, metrics = fit_gmm(features, n_components=2)

duration = time.time() - start

telemetry.log_button_click('run_gmm',
                          n_components=2,
                          num_features=len(feature_names),
                          num_breaths=len(labels))

telemetry.log_timing('gmm_clustering', duration,
                    num_breaths=len(labels),
                    n_components=2)

# If user clicks Apply
if user_accepted:
    telemetry.log_feature_used('gmm_clustering')
    telemetry.log_breath_statistics(
        num_breaths=len(labels),
        cluster_0_size=np.sum(labels == 0),
        cluster_1_size=np.sum(labels == 1),
        silhouette_score=metrics['silhouette']
    )
```

---

## 💾 Exports

**Functions:**
- `telemetry.log_export(export_type)`
- `telemetry.log_timing('export', duration, **params)`

### Track:
- [ ] **Export type** ('breaths_csv', 'timeseries_csv', 'summary_pdf', 'npz_session')
- [ ] **Number of rows/breaths exported**
- [ ] **Export duration**
- [ ] **File size** (if available)

### Where to add:
```python
# In export handler
import time
start = time.time()

df.to_csv(filename)

duration = time.time() - start
file_size_mb = os.path.getsize(filename) / (1024 * 1024)

telemetry.log_export('breaths_csv')
telemetry.log_timing('export_csv', duration,
                    num_rows=len(df),
                    file_size_mb=round(file_size_mb, 2))
```

---

## 📊 Breath Statistics

**Function:** `telemetry.log_breath_statistics(num_breaths, **params)`

### Track (after analysis complete):
- [ ] **Total breaths detected**
- [ ] **Mean breathing frequency** (Hz)
- [ ] **Breathing regularity score**
- [ ] **Eupnea percentage** (% of time in normal breathing)
- [ ] **Apnea count** (number of apnea events)
- [ ] **Sniff count** (if using GMM)

### Where to add:
```python
# After completing breath analysis
telemetry.log_breath_statistics(
    num_breaths=len(state.breaths),
    mean_frequency_hz=calculate_mean_freq(state.breaths, state.fs),
    regularity_score=calculate_regularity(state.breaths),
    eupnea_percentage=calculate_eupnea_percent(state.eupnea_mask),
    apnea_count=len(state.apnea_events),
    sniff_count=np.sum(state.gmm_labels == 1) if state.gmm_labels else None
)
```

---

## 🧭 Navigation

**Function:** `telemetry.log_navigation(action, **params)`

### Track:
- [ ] **Sweep changes** (how often users change sweeps)
- [ ] **Zoom in/out**
- [ ] **Pan left/right**
- [ ] **Reset view**

### Where to add:
```python
# In navigation handlers
def on_sweep_changed(self, new_sweep):
    """Handle sweep spinbox change."""
    telemetry.log_navigation('change_sweep',
                            sweep_number=new_sweep,
                            total_sweeps=self.state.n_sweeps)

    self.update_plot()

def on_zoom_in(self):
    telemetry.log_navigation('zoom_in',
                            zoom_level=self.current_zoom)

def on_zoom_out(self):
    telemetry.log_navigation('zoom_out',
                            zoom_level=self.current_zoom)
```

---

## ⌨️ Keyboard Shortcuts

**Function:** `telemetry.log_keyboard_shortcut(shortcut_name, **params)`

### Track:
- [ ] **Shift+Click** (add peak)
- [ ] **Ctrl+Click** (delete peak)
- [ ] **F1** (help dialog)
- [ ] **Ctrl+S** (save session)
- [ ] **Ctrl+O** (open file)
- [ ] **Ctrl+E** (export)
- [ ] Any other shortcuts

### Where to add:
```python
# In keyboard event handlers
def keyPressEvent(self, event):
    """Handle keyboard shortcuts."""
    if event.key() == Qt.Key.Key_F1:
        telemetry.log_keyboard_shortcut('f1_help')
        self.show_help_dialog()

    elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
        telemetry.log_keyboard_shortcut('ctrl_s_save')
        self.save_session()
```

---

## ⏱️ Workflow Timing

**Function:** `telemetry.log_timing(operation_name, duration, **params)`

### Track:
- [ ] **File load time**
- [ ] **Peak detection time**
- [ ] **GMM clustering time**
- [ ] **Filter application time**
- [ ] **Export time**
- [ ] **Channel selection to save** (workflow duration)

### Channel Selection to Save Example:
```python
class PlethApp:
    def __init__(self):
        self.channel_selection_time = None

    def on_channel_selected(self, channel_index):
        """User selects analysis channel."""
        import time
        self.channel_selection_time = time.time()

        telemetry.log_button_click('select_channel',
                                   channel_index=channel_index)

    def on_first_save_after_analysis(self):
        """User saves/exports for first time after selecting channel."""
        import time

        if self.channel_selection_time:
            duration = time.time() - self.channel_selection_time
            telemetry.log_timing('channel_selection_to_save', duration)
            self.channel_selection_time = None

        # ... continue with save
```

---

## ⚠️ Warnings (Non-Critical Issues)

**Function:** `telemetry.log_warning(warning_message, **params)`

### Track:
- [ ] **No peaks detected** (threshold too high?)
- [ ] **Filter unstable** (order too high?)
- [ ] **File corrupt/incomplete**
- [ ] **Missing channels**
- [ ] **Sampling rate mismatch**

### Where to add:
```python
# In various error-handling sections
if len(peaks) == 0:
    telemetry.log_warning('No peaks detected',
                         threshold=threshold,
                         data_min=data.min(),
                         data_max=data.max())

    QMessageBox.warning(self, "No peaks", "No peaks detected...")

if filter_order > 10:
    telemetry.log_warning('High filter order',
                         order=filter_order,
                         recommended_max=10)
```

---

## 💥 Crashes

**Function:** `telemetry.log_crash(error_message, **params)`

### Track:
- [ ] **Exception type** (IndexError, ValueError, etc.)
- [ ] **Operation where crash occurred**
- [ ] **Last button/action before crash**
- [ ] **Relevant state** (num_breaths, num_sweeps, etc.)

### Where to add:
```python
# Global exception handler (in main.py or run_debug.py)
import sys
from core import telemetry

def exception_hook(exctype, value, tb):
    """Catch unhandled exceptions."""
    import traceback

    telemetry.log_crash(
        error_message=f"{exctype.__name__}: {str(value)[:100]}",
        traceback_depth=len(traceback.extract_tb(tb)),
        last_action=telemetry._session_data.get('last_action', 'unknown')
    )

    # Call default handler
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = exception_hook

# Also wrap critical functions
try:
    result = risky_operation()
except Exception as e:
    telemetry.log_crash(f"{type(e).__name__} in risky_operation",
                       operation='risky_operation',
                       param_value=some_param)
    raise  # Re-raise after logging
```

---

## 🎯 Button Clicks

**Function:** `telemetry.log_button_click(button_name, **params)`

### Track EVERY button:
- [ ] Detect Peaks
- [ ] Apply Filter
- [ ] Run GMM
- [ ] Mark Sniff
- [ ] Reset View
- [ ] Save Session
- [ ] Load Session
- [ ] Export (all types)
- [ ] Spectral Analysis
- [ ] Calculate Metrics
- [ ] Help/About
- [ ] Settings

### Where to add:
```python
# In EVERY button click handler
def on_detect_peaks_clicked(self):
    telemetry.log_button_click('detect_peaks',
                               threshold=self.threshold_spinbox.value())
    # ... rest of handler

def on_apply_filter_clicked(self):
    telemetry.log_button_click('apply_filter',
                               highpass=self.hp_spinbox.value(),
                               lowpass=self.lp_spinbox.value())
    # ... rest of handler

def on_export_clicked(self):
    telemetry.log_button_click('export_dialog_open')
    # ... show export dialog
```

---

## 📈 What You'll Learn From This Data

### File Usage Patterns:
- What file types are most common?
- How big are typical files?
- What sampling rates do users use?

### Performance Metrics:
- How long does peak detection take for different file sizes?
- Are filters slow for some users?
- Which operations need optimization?

### User Workflows:
- Do users prefer auto-threshold or manual?
- How much manual editing happens?
- Do users use GMM clustering?
- Keyboard shortcuts vs mouse clicks?

### Problem Areas:
- Which operations crash most often?
- What warnings appear frequently?
- Where do users get stuck? (long times between steps)

### Feature Adoption:
- Which features are never used? (candidates for removal)
- Which features are used every session? (critical to maintain)
- Which shortcuts are discovered vs unknown?

### **🎯 ML Performance Tracking (Critical for Publication!)**

**Track detection accuracy improvements over time:**

#### **Session-End Metrics:**
- **`edit_percentage`**: % of breaths that needed manual correction
  - **Before ML**: 15-20% typical
  - **After ML**: Target 5-10%
  - **Shows:** Overall correction burden

- **`false_negative_rate`**: % of missed breaths (had to manually add)
  - **Shows:** Detection is too conservative (misses real breaths)

- **`false_positive_rate`**: % of wrong detections (had to manually delete)
  - **Shows:** Detection is too sensitive (false alarms)

- **`edits_per_file`**: Average # of corrections per file
  - **Shows:** User effort per analysis

#### **How to Use for ML Evaluation:**

1. **Baseline (current detection):**
   - Filter `session_end` events by `app_version = 1.0.9`
   - Average `edit_percentage` across all users
   - Example: "Current detection requires 18% manual correction"

2. **After ML classifier (future):**
   - Filter `session_end` events by `app_version = 1.1.0` (with ML)
   - Average `edit_percentage`
   - Example: "ML classifier reduced corrections to 6%"

3. **Publication claim:**
   - "Our ML breath classifier reduced manual editing by 67% (18% → 6%)"
   - "False positive rate decreased from 12% to 3%"
   - "False negative rate decreased from 6% to 3%"

4. **GA4 Custom Report:**
   - **Dimensions:** `app_version`
   - **Metrics:** Average `edit_percentage`, `false_negative_rate`, `false_positive_rate`
   - **Comparison:** v1.0.9 (no ML) vs v1.1.0 (with ML)
   - **Visualization:** Bar chart showing improvement

#### **Why This Is Perfect:**
- ✅ Quantifiable improvement metric
- ✅ Direct measure of user effort saved
- ✅ Shows both precision (false positives) and recall (false negatives)
- ✅ Tracks improvement over app versions
- ✅ Great for publication and grant applications

---

## 🔧 Implementation Priority

### Phase 1: Critical (Do Now)
1. ✅ Enhanced file loading (size, sampling rate, duration)
2. ✅ Peak detection (method, threshold, timing)
3. ✅ All button clicks
4. ✅ Manual edits
5. ✅ Crashes with global exception handler

### Phase 2: Important (Do Soon)
6. ✅ Filter application
7. ✅ GMM clustering timing and results
8. ✅ Export timing
9. ✅ Breath statistics
10. ✅ Warnings

### Phase 3: Nice to Have (Do Later)
11. ✅ Navigation tracking
12. ✅ Keyboard shortcuts
13. ✅ Workflow timing (channel → save)

---

## 📊 Registering Custom Dimensions in GA4

To filter/group by version and other parameters, register them as custom dimensions:

1. Go to: **GA4 → Admin → Data display → Custom definitions**
2. Click: **Create custom dimension**
3. Add these:

| Dimension name | Event parameter | Scope |
|---------------|----------------|-------|
| App Version | app_version | Event |
| Platform | platform | Event |
| Python Version | python_version | Event |
| Detection Method | detection_method | Event |
| Filter Type | filter_type | Event |
| Edit Type | edit_type | Event |
| Button Name | button | Event |

Once registered, you can filter reports by these dimensions!

---

## ✅ Final Checklist

Before considering telemetry "complete":

- [ ] All button clicks tracked
- [ ] All timing data tracked (file load, peak detect, GMM, export)
- [ ] Manual edits tracked
- [ ] Warnings tracked (no peaks, etc.)
- [ ] Crashes tracked with global handler
- [ ] Custom dimensions registered in GA4
- [ ] Test with real usage to verify events appear
- [ ] Create custom GA4 dashboard for key metrics

---

This comprehensive tracking will give you incredible insights into how PlethApp is actually used in the wild!
