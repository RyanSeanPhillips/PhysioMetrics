# PlethApp - Feature Backlog & Development Roadmap

This document tracks planned features, enhancements, and long-term development goals.

**📍 Quick Reference**: See checklist below for current status of all features.

---

## Current Status Checklist

### ✅ Completed Features (v1.0.8)
- [x] High-resolution splash screen replacement
- [x] Cache eupnea masks for 4× export speedup (changed approach: calculate means during GMM)
- [x] Add optional auto-GMM checkbox to eliminate editing lag
- [x] Multi-file ABF concatenation support
- [x] Move point editing mode (drag peaks to reposition)
- [x] Enhanced eupnea threshold controls dialog
- [x] Enhanced outlier threshold controls dialog
- [x] Sniffing bout detection and annotation
- [x] Higher-level modularization (editing_modes, export_manager)
- [x] Status bar message window with timing and warnings
- [x] Setup line_profiler with @profile decorators and run_profiling.bat
- [x] Export optimization: searchsorted + mask_to_intervals - 49% speedup!
- [x] Add checkboxes to SaveMetaDialog for optional file exports
- [x] **NPZ file save/load with full state restoration** (.pleth.npz sessions)
- [x] **Auto-prominence threshold detection using Otsu's method** (interactive histogram dialog)
- [x] **Help button with comprehensive usage guide** (3 tabs: workflow, exported files, about) - *NOTE: Update as features are added*

### 🎯 High Priority (Next Up)
- [ ] **Anonymous usage tracking and telemetry** (4-6 hours) ← USAGE STATISTICS
- [ ] **Relative metrics (z-scores) for ML features** (2-3 hours) ← ML FOUNDATION
- [ ] **Auto-threshold selection with ML auto-labeling** (3-4 hours) ← ML FOUNDATION
- [ ] Display mode toggles for breathing states - background vs line (4-5 hours)

### 📋 Medium Priority (Scientific Features)
- [ ] **Project Builder - Batch processing workflow** (6-8 hours) ← PRODUCTIVITY BOOST
- [ ] 25ms stim analysis: phase response curves, airflow CTA, firing rate (10-15 hours)
- [ ] Omit region mode - exclude artifact sections within sweep (2-3 hours)
- [ ] Idealized breath template overlay for move point mode (3-4 hours)
- [ ] CSV/text time-series import with preview dialog (6-7 hours)
- [ ] Statistical significance in consolidated data (Cohen's d, p-values) (4-5 hours)
- [ ] Expiratory onset detection (separate from insp offset) (3-4 hours)

### 🔮 Long-Term Features (Post v1.0)
- [ ] Dark mode toggle for main plot (2-3 hours)
- [ ] ML-ready data export format (CSV, HDF5, JSON) (3-4 hours)
- [ ] ML breath classifier (Random Forest + XGBoost) (12-20 hours in phases)
- [ ] **LLM-assisted metadata extraction from experiment notes** (8-12 hours) ← AI ENHANCEMENT

---

## Current Development Phase

**Active**: v0.9.x → v1.0.0 (Pre-release to Public Release)

**Focus Areas**:
1. Performance optimization (see PERFORMANCE_OPTIMIZATION_PLAN.md) ✅ Phase 2 Complete
2. ML breath classifier implementation (see PUBLICATION_ROADMAP.md)
3. Documentation and examples for JOSS publication

---

## High Priority Features (Next Implementation Phase)

### 1. High-Resolution Splash Screen
**Description**: Replace current splash screen with higher resolution image for better visual presentation

**Motivation**: Current splash screen appears blurry on high-DPI displays

**Implementation**:
- Create new splash screen image (2× or 3× resolution)
- Update `run_debug.py` to use high-res image
- Ensure proper scaling on different display resolutions

**Files to modify**:
- `run_debug.py`
- Image assets in `images/` directory

**Effort**: 30 minutes

---

### 2. NPZ File Save/Load with Full State Restoration
**Description**: Save all analysis data (traces, peaks, metrics, annotations) to NPZ file for later review

**Motivation**:
- Quality control: Review analysis after initial processing
- Collaborative review: Share analyzed data with colleagues
- Incremental analysis: Continue work across multiple sessions

**Features**:
- **Save**: Raw data, processed traces, detected peaks/events, calculated metrics, manual edits, filter settings
- **Load**: Restore complete analysis state - user can review, verify, or modify previous work
- **Auto-populate filename**: When re-saving, suggest original NPZ filename
- **Versioned format**: Backward compatibility checking for future format changes

**File Structure**:
```python
npz_data = {
    # Raw data
    'raw_sweeps': dict,          # Original channel data by channel name
    'sr_hz': float,              # Sample rate
    'channel_names': list,       # Ordered channel names

    # Detection results
    'peaks_by_sweep': dict,      # All detected features (peaks, onsets, offsets)

    # Analysis results
    'breath_metrics': dict,      # Computed metrics (IF, Ti, Te, amp, etc.)
    'gmm_probabilities': dict,   # GMM classification results

    # User annotations
    'manual_edits': dict,        # Added/deleted peaks, marked sighs
    'user_notes': str,           # Optional text notes

    # Processing settings
    'filter_params': dict,       # HP, LP, order, notch, invert, etc.
    'detection_params': dict,    # Threshold, prominence, distance

    # Metadata
    'version': str,              # Format version (e.g., '1.0')
    'save_timestamp': str,       # ISO format timestamp
    'source_file': str           # Original ABF/SMRX/EDF file path
}
```

**UI Components**:
- "Save Analysis" button → file dialog with .npz extension
- "Load Analysis" button → file dialog, version check, full state restoration
- Status bar message: "Analysis saved to analysis_2025-10-24.npz"

**Files to modify**:
- `core/export.py` (add `save_npz()` and `load_npz()` functions)
- `main.py` (add buttons, file dialogs, state restoration logic)

**Effort**: 5-6 hours

---

### 3. Multi-File ABF Concatenation
**Description**: Load multiple ABF files as concatenated sweeps

**Motivation**:
- Long experiments split across multiple ABF files
- Analyze multiple sessions as a single continuous dataset
- Simplify workflow when recordings are naturally segmented

**Features**:
- **Multi-select in file dialog**: User selects multiple ABF files
- **Files treated as sequential sweeps**: Each file's sweeps appended to previous
- **Validation checks**:
  - Same number of channels across all files
  - Same channel names (in same order)
  - Same sample rate
  - Same date (extracted from filename format: `YYYYMMDD####.abf`)
- **Warning dialog**: If validation fails, show issues with option to proceed anyway
- **Status bar display**: "File 1 of 3: 20241024_0001.abf (Sweeps 0-9)"

**Validation Logic**:
```python
def validate_multi_file_abf(file_paths: list[Path]) -> tuple[bool, list[str]]:
    """
    Validate that multiple ABF files can be concatenated.
    Returns (is_valid, error_messages).
    """
    errors = []

    # Extract metadata from first file
    ref_metadata = get_abf_metadata(file_paths[0])

    # Check each subsequent file
    for path in file_paths[1:]:
        meta = get_abf_metadata(path)

        if meta.num_channels != ref_metadata.num_channels:
            errors.append(f"{path.name}: Channel count mismatch")

        if meta.channel_names != ref_metadata.channel_names:
            errors.append(f"{path.name}: Channel names differ")

        if meta.sample_rate != ref_metadata.sample_rate:
            errors.append(f"{path.name}: Sample rate mismatch")

        if meta.date != ref_metadata.date:
            errors.append(f"{path.name}: Date mismatch")

    return (len(errors) == 0, errors)
```

**UI Components**:
- Modify file dialog to allow multi-selection
- Validation warning dialog with checkbox "Proceed anyway"
- File info display in status bar or dedicated label

**Files to modify**:
- `main.py` (file dialog, validation, display)
- `core/abf_io.py` (concatenation logic, metadata extraction)

**Effort**: 4-5 hours

---

### 4. Auto-Threshold Selection with ML Auto-Labeling
**Description**: Automatically detect optimal peak detection threshold using elbow detection, then auto-label all peaks for ML training

**Motivation**:
- **Zero manual annotation**: Auto-label 4,500 peaks (3,100 breaths + 1,400 noise) without user clicking
- **ML training foundation**: Generate labeled dataset automatically for breath classifier
- **Adaptive to noise**: Different recordings have different noise levels - auto-adapt
- **Future-proof**: Detect all peaks (threshold=0) so better ML can reclassify later

**Key Insight** (from user testing):
- Threshold=0 with min_distance=0.05s: 4,500 peaks (10 min recording)
- Optimal threshold: 3,100 peaks (real breaths)
- **Only 31% overhead** (1,400 extra peaks) - totally manageable!
- Min distance prevents explosion (no 20Hz breathing physically possible)

**Algorithm**:
```python
# Step 1: Detect ALL peaks (threshold=0, min_distance=0.05s)
all_peaks = detect_peaks(signal, prominence=0, distance=samples_per_50ms)

# Step 2: Compute peaks vs threshold curve
thresholds = np.linspace(0.01, 0.5, 100)
n_peaks = []
for thresh in thresholds:
    peaks_at_thresh = all_peaks[prominence >= thresh]
    n_peaks.append(len(peaks_at_thresh))

# Step 3: Find elbow (plateau before noise pickup)
optimal_threshold = find_knee_point(thresholds, n_peaks)
# Methods: second derivative, curvature, or Kneedle algorithm

# Step 4: AUTO-LABEL all peaks (no manual work!)
auto_labels = []
for peak in all_peaks:
    peak_prom = compute_prominence(signal, peak)
    if peak_prom >= optimal_threshold:
        auto_labels.append("breath")      # 3100 peaks (69%)
    else:
        auto_labels.append("not_breath")  # 1400 peaks (31%)
```

**UI Implementation**:
```
┌─────────────────────────────────────────────┐
│ Peak Detection Settings                     │
├─────────────────────────────────────────────┤
│ ☑ Auto-detect threshold (elbow method)     │
│   Detected: 0.12 prominence                 │
│   → 3,100 peaks (1,400 below threshold)     │
│                                             │
│ ☐ Manual threshold: [____] 0.15            │
│                                             │
│ ☐ Show all peaks (including below thresh)  │
│                                             │
│ [📊 View Threshold Curve]                  │
└─────────────────────────────────────────────┘
```

**Threshold Curve Visualization**:
```
Number of Peaks
    │
4500├─────────────────  ← Noise starts
    │                ╱
3500│               ╱
    │              ╱
3100├─────────────●    ← Elbow (auto-selected)
    │            ╱
2000│          ╱
    │        ╱
    │      ╱
    │    ╱
  0 └────────────────
    0.0  0.1  0.2  0.3  Prominence Threshold
              ↑
         Auto-selected
```

**ML Training Data Export** (Deep Learning Ready):
```python
# Saved in NPZ file - INCLUDES RAW WAVEFORMS for future deep learning!
ml_training_data = {
    # Peak locations
    "peaks": all_peaks,                  # 4500 peak indices
    "onsets": onset_indices,             # Onset indices
    "offsets": offset_indices,           # Offset indices
    "auto_threshold": 0.12,              # Detected elbow

    # Labels
    "labels": auto_labels,               # ["breath", "not_breath", ...]
    "label_sources": label_sources,      # ["auto", "user_corrected", ...]

    # Basic features (8) - for current Random Forest ML
    "features": {
        "ti": ti_values,                 # Inspiratory time
        "te": te_values,                 # Expiratory time
        "amp_insp": amp_insp_values,     # Inspiratory amplitude
        "amp_exp": amp_exp_values,       # Expiratory amplitude
        "max_dinsp": max_dinsp_values,   # Max inspiratory derivative
        "max_dexp": max_dexp_values,     # Max expiratory derivative
        "prominence": prominence_values, # Peak prominence
        "freq": freq_values,             # Instantaneous frequency

        # Half-width features (4) - Phase 2.1
        "fwhm": fwhm_values,             # Full width at half max (robust!)
        "width_25": width_25_values,     # Width at 25% of peak
        "width_75": width_75_values,     # Width at 75% of peak
        "width_ratio": width_ratio_vals, # Shape ratio (75%/25%)

        # Sigh detection features (9) - Phase 2.2
        "n_inflections": n_inflect_vals, # Inflection point count (double-hump)
        "rise_variability": rise_var,    # Derivative variability
        "n_shoulder_peaks": n_shoulder,  # Secondary peak count
        "shoulder_prominence": sh_prom,  # Shoulder peak size
        "rise_autocorr": rise_autocorr,  # Oscillation detection
        "peak_sharpness": peak_sharp,    # Curvature at peak
        "trough_sharpness": trough_sharp,# Curvature at trough
        "skewness": skewness_vals,       # Statistical asymmetry
        "kurtosis": kurtosis_vals        # Statistical peakedness
    },
    # Total: 21 features for Random Forest

    # RAW WAVEFORMS (for future deep learning) - Phase 3
    "waveforms": {
        "breath_segments": breath_waveforms,   # (4500, 300) array
        "context_segments": context_waveforms, # (4500, 500) array (wider context)
        "sample_rate": 1000.0,                 # Hz
        "normalization": "minmax"              # How waveforms were normalized
    },
    # Storage: ~15 MB per file (very reasonable!)

    # Metadata
    "file_id": "20241024_mouse05.abf",
    "timestamp": "2025-01-15 14:32:01",
    "plethapp_version": "1.0.8"
}
```

**Feature Evolution Roadmap**:
- **Phase 1** (Now): 8 basic features → 85-90% accuracy
- **Phase 2.1** (+4 half-width): 12 features → 90-92% accuracy (more robust)
- **Phase 2.2** (+9 sigh detection): 21 features → 93-95% accuracy (excellent!)
- **Phase 3** (Future): Deep learning on waveforms → 96-98% accuracy

**Why Save Waveforms Now**:
- ✅ Future-proof for deep learning (when you have 50+ files)
- ✅ Only 15 MB per file (totally reasonable)
- ✅ Can always revisit with better models later
- ✅ No regrets - waveforms already extracted during analysis
- ✅ Enables CNN/LSTM approaches without re-analyzing files

**User Workflow**:
1. Load file → Run peak detection with auto-threshold
2. App shows 3,100 "breath" peaks (clean display)
3. User reviews - looks good!
4. User toggles "Show all peaks" → sees 1,400 faint peaks below threshold
5. User manually corrects 5 mistakes:
   - Promotes 3 real breaths that were below threshold
   - Demotes 2 noise peaks that were above threshold
6. Corrections automatically update labels and label_source
7. Export saves all 4,500 peaks with corrected labels

**Benefits**:
- ✅ **Zero annotation burden**: 4,500 peaks labeled automatically
- ✅ **Good class balance**: 69% positive, 31% negative (no special ML techniques needed)
- ✅ **Clean UI**: Only shows "breath" peaks by default
- ✅ **Fast**: Only 31% more peaks to process (not 10× like raw threshold=0)
- ✅ **Active learning ready**: User corrections improve future ML models
- ✅ **Future-proof**: All peaks saved, better ML can reclassify

**Files to modify**:
- `dialogs/event_detection_dialog.py` (add auto-threshold checkbox, view curve button)
- `core/peaks.py` (add `find_optimal_threshold()`, `compute_threshold_curve()`)
- `main.py` (add auto-labeling logic, "show all peaks" toggle)
- `core/state.py` (save auto-labels and label sources)
- `export/export_manager.py` (export ML training data)

**Dependencies**:
- NumPy (existing)
- SciPy (existing, for knee detection)
- Matplotlib (existing, for curve visualization)

**Effort**: 3-4 hours

**Follow-up Features (Phased Implementation)**:

**Phase 1: Basic ML** (3-4 hours)
- Implement auto-threshold with elbow detection
- Auto-label all peaks (breath vs not_breath)
- Save 8 basic features + waveforms to NPZ
- Export ML training data format
- Expected accuracy: 85-90%

**Phase 2.1: Half-Width Features** (2-3 hours)
- Add FWHM (full width at half maximum) - robust to noise
- Add width at 25%, 75% of peak amplitude
- Add width ratio (shape descriptor)
- Total features: 8 + 4 = 12
- Expected accuracy: 90-92%

**Phase 2.2: Sigh Detection Features** (3-4 hours)
- Add inflection point counting (detects double-hump)
- Add shoulder peak detection (local maxima in rising phase)
- Add rise-phase autocorrelation (detects oscillations)
- Add curvature features (peak/trough sharpness)
- Add statistical shape features (skewness, kurtosis)
- Total features: 12 + 9 = 21
- Expected accuracy: 93-95%

**Phase 3: ML Classifier** (12-20 hours)
- Train Random Forest on 21 features
- Implement XGBoost for comparison
- Active learning integration
- Expected accuracy: 95-97%

**Phase 4: Deep Learning** (Optional, 20-30 hours, when you have 50+ files)
- Load saved waveforms from NPZ
- Train CNN on raw breath segments
- Expected accuracy: 96-98%

**Training Data Requirements** (see below for details):
- **Minimum viable**: 5-10 files (~20,000-40,000 peaks, ~15,000-30,000 breaths)
- **Good performance**: 20-30 files (~80,000-120,000 peaks)
- **Excellent performance**: 50-100 files (~200,000-400,000 peaks)
- **Key insight**: Diversity matters more than quantity (different animals, conditions, noise levels)

---

### 5. CSV/Text Time-Series Import
**Description**: Load arbitrary time-series data from CSV/text files

**Motivation**:
- Import data from custom acquisition systems
- Load exported data from other analysis software
- Test with synthetic data
- Support users without ABF/SMRX/EDF formats

**Features**:
- **File preview dialog**: Shows first 20 rows of file
- **Column selection UI**: User picks time column + data columns
- **Auto-detection**:
  - Headers (first row is header or not)
  - Delimiter (comma, tab, space, semicolon)
  - Decimal separator (. or ,)
- **Sample rate detection**:
  - Auto-calculate from time column (median of successive differences)
  - User-specified override if time column not available
- **Column to sweep mapping**: Each selected data column becomes a sweep

**UI Components**:
```
┌─ CSV Import Dialog ─────────────────────────────┐
│                                                  │
│ File: recording.csv                              │
│                                                  │
│ Preview (first 20 rows):                         │
│ ┌──────────────────────────────────────────────┐ │
│ │ Time,Pleth,ECG,Temp                          │ │
│ │ 0.000,0.123,-0.045,37.2                      │ │
│ │ 0.001,0.134,-0.052,37.2                      │ │
│ │ ...                                          │ │
│ └──────────────────────────────────────────────┘ │
│                                                  │
│ ☑ First row is header                            │
│ Delimiter: [Comma ▼]                             │
│ Decimal separator: [Period ▼]                    │
│                                                  │
│ Time column: [Time ▼]                            │
│ ☑ Auto-detect sample rate (from time column)     │
│ Sample rate (Hz): [1000.0        ]               │
│                                                  │
│ Data columns:                                    │
│ ☑ Pleth                                          │
│ ☑ ECG                                            │
│ ☐ Temp                                           │
│                                                  │
│           [Cancel]  [Import]                     │
└──────────────────────────────────────────────────┘
```

**Implementation Details**:
```python
# core/io/csv_loader.py

def load_csv(file_path: str, config: CSVImportConfig):
    """
    Load CSV/text file as time-series data.

    Args:
        file_path: Path to CSV file
        config: User-selected import configuration

    Returns:
        sr_hz, sweeps_by_channel, channel_names, t
    """
    # Read file with pandas
    df = pd.read_csv(
        file_path,
        header=0 if config.has_header else None,
        delimiter=config.delimiter,
        decimal=config.decimal_sep
    )

    # Extract time vector
    if config.time_column:
        t = df[config.time_column].values
        sr_hz = 1.0 / np.median(np.diff(t))  # Auto-detect sample rate
    else:
        sr_hz = config.sample_rate
        t = np.arange(len(df)) / sr_hz

    # Extract data columns
    sweeps_by_channel = {}
    for col in config.selected_columns:
        data = df[col].values
        sweeps_by_channel[col] = data.reshape(-1, 1)  # (n_samples, 1) for continuous

    channel_names = list(config.selected_columns)

    return sr_hz, sweeps_by_channel, channel_names, t
```

**Files to create**:
- `core/io/csv_loader.py` (loader logic)

**Files to modify**:
- `main.py` (add `CSVImportDialog` class, integrate with file loading)
- `core/abf_io.py` (dispatcher to route .csv files to csv_loader)

**Effort**: 6-7 hours

---

### 5. ✅ Spike2 .smrx File Support (COMPLETED)
**Status**: ✅ **IMPLEMENTED** (2025-10-05)

**See FILE_FORMATS.md for complete documentation.**

---

### 6. Move Point Editing Mode
**Description**: Add button/mode to manually drag and reposition detected peaks

**Motivation**:
- Fine-tune automated detection results
- Correct peaks that are slightly misplaced
- Adjust timing of onsets/offsets for better accuracy

**Features**:
- New editing mode: "Move Peak"
- Click and drag peak markers to new positions
- Real-time preview of new position
- Snap to local maximum/minimum option
- Updates dependent features (onsets, offsets, metrics) automatically

**UI Components**:
- "Move Peak" button in editing controls
- Cursor changes to move icon when hovering over peak
- Visual feedback: Selected peak highlighted, drag preview shown

**Implementation**:
- Detect mouse click near peak marker (within threshold)
- Enter drag mode: Track mouse motion
- Update peak position on mouse release
- Recompute breath events for affected peak
- Redraw plot with updated peak position

**Files to modify**:
- `main.py` (editing modes, mouse event handlers)
- `editing/editing_modes.py` (move peak mode logic)

**Effort**: 3-4 hours

---

### 7. Enhanced Eupnea Threshold Controls
**Description**: Convert "Eupnea Thresh (Hz)" label to clickable button/link

**Motivation**:
- Expose all eupnea detection parameters in one dialog
- Allow manual annotation of eupnic regions
- Provide visual feedback for parameter tuning

**Features**:
- **Parameter dialog** with all eupnea detection settings:
  - Frequency threshold (Hz) - currently exposed
  - Duration threshold (s) - currently hardcoded to 2s
  - Regularity criteria - currently not exposed
  - Minimum gap between regions
- **Manual mode**: User can click-drag to highlight/annotate eupnic regions
- **Visual feedback**: Real-time preview of detected eupnic regions as parameters change

**UI Components**:
```
┌─ Eupnea Detection Parameters ───────────────┐
│                                              │
│ Frequency threshold: [5.0     ] Hz          │
│ Duration threshold:  [2.0     ] s           │
│ Regularity (RMSSD):  [0.1     ] s           │
│ Minimum gap:         [1.0     ] s           │
│                                              │
│ ☐ Enable manual annotation mode              │
│                                              │
│ Preview: [Update Preview]                    │
│                                              │
│         [Cancel]  [Apply]                    │
└──────────────────────────────────────────────┘
```

**Files to modify**:
- `main.py` (create `EupneaControlDialog` class)
- `core/metrics.py` (expose additional parameters in `detect_eupnic_regions()`)

**Effort**: 4-5 hours

---

### 8. Enhanced Outlier Threshold Controls
**Description**: Convert "Outlier Thresh (SD)" label to clickable button/link

**Motivation**:
- Allow selection of which metrics to use for outlier detection
- Provide individual thresholds for different metrics
- Preview flagged breaths before applying

**Features**:
- **Multi-metric selection**: Choose which metrics to use (Ti, Te, IF, amp_insp, amp_exp, area_insp, area_exp)
- **Individual SD thresholds**: Different threshold for each metric
- **Preview mode**: Show which breaths would be flagged before applying
- **Logic options**: Flag if ANY metric exceeds threshold vs ALL metrics

**UI Components**:
```
┌─ Outlier Detection Parameters ───────────────┐
│                                               │
│ Metrics to check:                             │
│ ☑ Instantaneous Frequency  [3.0 ] SD         │
│ ☑ Inspiratory Time (Ti)    [3.0 ] SD         │
│ ☑ Expiratory Time (Te)     [3.0 ] SD         │
│ ☑ Inspiratory Amplitude    [3.0 ] SD         │
│ ☐ Expiratory Amplitude     [3.0 ] SD         │
│ ☐ Inspiratory Area         [3.0 ] SD         │
│ ☐ Expiratory Area          [3.0 ] SD         │
│                                               │
│ Logic: ⦿ Flag if ANY metric exceeds threshold │
│        ○ Flag if ALL metrics exceed threshold │
│                                               │
│ Preview: [Show Flagged Breaths]               │
│                                               │
│         [Cancel]  [Apply]                     │
└───────────────────────────────────────────────┘
```

**Files to modify**:
- `main.py` (create `OutlierControlDialog` class)
- `core/metrics.py` (add multi-metric outlier detection function)

**Effort**: 4-5 hours

---

### 9. Statistical Significance in Consolidated Data
**Description**: Add statistical testing to identify when stim response differs significantly from baseline

**Motivation**:
- Identify statistically significant responses to stimulation
- Quantify effect sizes for publication
- Reduce subjective interpretation of consolidated plots

**Features**:
- **Three new columns in consolidated CSV**:
  - `cohens_d`: Effect size at each timepoint (mean - baseline_mean) / baseline_sd
  - `p_value`: Uncorrected paired t-test p-value (timepoint sweeps vs baseline sweeps)
  - `sig_corrected`: Boolean flag after Bonferroni correction (p < 0.05/n_timepoints)

- **Visual enhancements in consolidated plot**:
  - Shaded gray background for significant regions
  - Asterisks for significance levels: `*` p<0.05, `**` p<0.01, `***` p<0.001
  - Horizontal dashed lines at Cohen's d = ±0.5 (medium effect size)

- **User-configurable options**:
  - Baseline window (default: -2 to 0 sec pre-stim)
  - Significance threshold (default: 0.05)
  - Correction method: Bonferroni (conservative) or None

**Alternative Methods** (future consideration):
- Cluster-based permutation testing for sustained effects
- Confidence interval non-overlap flagging
- Mixed-effects models for multi-animal data

**Implementation Example**:
```python
def compute_statistical_significance(
    consolidated_data: pd.DataFrame,
    baseline_window: tuple[float, float] = (-2.0, 0.0),
    alpha: float = 0.05,
    correction: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Add statistical significance columns to consolidated data.
    """
    # Extract baseline data
    baseline_mask = (consolidated_data['time'] >= baseline_window[0]) & \
                    (consolidated_data['time'] <= baseline_window[1])
    baseline_mean = consolidated_data.loc[baseline_mask, 'mean'].mean()
    baseline_sd = consolidated_data.loc[baseline_mask, 'mean'].std()

    # Compute Cohen's d
    consolidated_data['cohens_d'] = \
        (consolidated_data['mean'] - baseline_mean) / baseline_sd

    # Compute p-values (paired t-test at each timepoint)
    # ... (compare sweep values at each timepoint to baseline)

    # Apply Bonferroni correction
    if correction == 'bonferroni':
        n_tests = len(consolidated_data)
        consolidated_data['sig_corrected'] = \
            consolidated_data['p_value'] < (alpha / n_tests)

    return consolidated_data
```

**Files to modify**:
- `main.py` (consolidation dialog, plotting)
- `core/metrics.py` (statistical helpers)
- `export/export_manager.py` (CSV export with new columns)

**Dependencies**: `scipy.stats` (already included)

**Effort**: 4-5 hours

---

### 10. Anonymous Usage Tracking and Telemetry
**Description**: Privacy-first telemetry system to track adoption, feature usage, and errors

**Motivation**:
- **Track adoption**: How many users are actively using PlethApp?
- **Prioritize features**: Which features are most/least used?
- **Find bugs faster**: Which operations crash most frequently?
- **Publication metrics**: Report usage statistics in future papers
- **Grant justification**: Demonstrate research impact with usage data

**Implementation Strategy: Opt-Out by Default (Academic Software Model)**

**First-Launch Dialog**:
```
PlethApp - First Time Setup

Help improve PlethApp by sharing anonymous usage statistics.

☑ Share anonymous usage data
  • Number of files analyzed (no file names or paths)
  • Features used (GMM, manual editing, etc.)
  • App version, OS, and Python version
  • Anonymous user ID (random UUID)

☑ Send crash reports
  • Error stack traces to help fix bugs
  • No personal or experimental data

[Learn More...]  [Continue]
```

**What Gets Collected**:
```python
telemetry_event = {
    "user_id": "a3f2e8c9-4b7d-...",  # Random UUID, generated once
    "version": "1.0.8",
    "os": "Windows 10",
    "python_version": "3.11.5",
    "timestamp": "2025-10-28T14:32:01Z",

    "session_data": {
        "files_analyzed": 3,
        "file_types": {"abf": 2, "smrx": 1, "edf": 0},
        "total_breaths": 1247,
        "total_sweeps": 30,
        "session_duration_minutes": 45,

        "features_used": [
            "gmm_clustering",
            "manual_editing_add_peak",
            "manual_editing_delete_peak",
            "mark_sniff",
            "spectral_analysis"
        ],

        "exports": {
            "summary_pdf": 1,
            "breaths_csv": 1,
            "timeseries_csv": 1,
            "npz_session": 1
        }
    }
}
```

**What NEVER Gets Collected**:
- File names, paths, or directory structure
- Animal metadata (strain, virus, injection site, etc.)
- Actual breathing data (frequencies, amplitudes, metrics)
- User's name, email, or institution
- Computer name or network information

**Technical Implementation**:

**Option 1: Sentry (Recommended for v1.0)**
```python
import sentry_sdk

# On app startup (if user has opted in)
if config.get('telemetry_enabled', True):  # Default True = opt-out
    sentry_sdk.init(
        dsn="https://your-project@sentry.io/...",
        traces_sample_rate=0.0,  # No performance tracing
        send_default_pii=False,  # Never send personal info
        before_send=sanitize_event  # Strip any paths/filenames
    )

    # Set custom user ID
    sentry_sdk.set_user({"id": config.get('user_id')})
```

**Option 2: Custom Simple Server**
```python
import requests
import json
from datetime import datetime

def send_telemetry(event_data):
    """Send telemetry event to server."""
    if not config.get('telemetry_enabled', True):
        return

    try:
        response = requests.post(
            'https://your-server.com/api/telemetry',
            json=event_data,
            timeout=5  # Don't block UI
        )
        response.raise_for_status()
    except Exception:
        # Silently fail - never interrupt user workflow
        pass
```

**Settings UI** (in Help dialog About tab):
```python
# Add to help_dialog.py About tab:
telemetry_group = QGroupBox("Usage Statistics")
telemetry_layout = QVBoxLayout()

self.telemetry_checkbox = QCheckBox("Share anonymous usage statistics")
self.telemetry_checkbox.setChecked(config.get('telemetry_enabled', True))
self.telemetry_checkbox.toggled.connect(self.on_telemetry_changed)

crash_checkbox = QCheckBox("Send crash reports")
crash_checkbox.setChecked(config.get('crash_reports_enabled', True))

learn_more_link = QLabel('<a href="#details">What data is collected?</a>')
learn_more_link.linkActivated.connect(self.show_telemetry_details)

telemetry_layout.addWidget(self.telemetry_checkbox)
telemetry_layout.addWidget(crash_checkbox)
telemetry_layout.addWidget(learn_more_link)
telemetry_group.setLayout(telemetry_layout)
```

**Files to modify**:
- `main.py` (first-launch dialog, telemetry event tracking)
- `dialogs/help_dialog.py` (add opt-out checkbox to About tab)
- `core/state.py` or new `core/telemetry.py` (telemetry manager)
- `core/config.py` (new file - persistent config with user_id, telemetry settings)

**Dependencies**:
- `sentry-sdk` (Option 1, recommended)
- OR `requests` (Option 2, already included in requirements)

**Effort**: 4-6 hours
- UUID generation and config storage: 30 min
- First-launch dialog: 1.5 hours
- Telemetry event tracking: 2 hours
- Settings UI in Help dialog: 1 hour
- Sentry setup or custom server: 1-2 hours

**Ethical Considerations**:
✅ **Opt-out default is acceptable because**:
- Only academic/research software (not commercial)
- Clear disclosure on first launch (not hidden)
- No personal/identifying data collected
- Easy to disable (one checkbox)
- Benefits the research community (better software through usage data)

✅ **Precedents**: Many academic tools use opt-out telemetry:
- VS Code (Microsoft)
- Anaconda Navigator
- JupyterLab
- Many PyPI packages with `pip` install analytics

⚠️ **Alternative: Opt-in** if you prefer maximum privacy:
- Set `telemetry_enabled` default to `False`
- First-launch dialog has checkboxes unchecked
- User must actively choose to share data
- **Trade-off**: ~10-20% participation rate vs. 80-90% with opt-out

**Recommendation for PlethApp**: **Opt-out** with clear disclosure
- You're releasing academic software, not commercial product
- Usage data will help justify future grants/publications
- Easy for privacy-conscious users to disable
- Follows VS Code model (trusted by researchers)

---

### 11. Relative Metrics (Z-scores) for ML Features
**Description**: Calculate z-score normalized metrics for each breath relative to baseline population

**Motivation**:
- **Normalize across animals**: Different mice have different baseline breathing rates
- **Robust ML features**: Machine learning models perform better with normalized data
- **Automatic outlier detection**: High z-scores (>2.5 SD) indicate unusual breaths (sighs, gasps)
- **Interpretable results**: "This breath is 3.2 standard deviations above baseline"
- **Publication-ready**: Z-scores are standard in neuroscience papers

**Example Workflow**:
```
Normal breath: IF=2.1 Hz, baseline μ=2.0 Hz, σ=0.3 Hz
→ IF_zscore = (2.1 - 2.0) / 0.3 = 0.33

Sigh: IF=0.8 Hz, baseline μ=2.0 Hz, σ=0.3 Hz
→ IF_zscore = (0.8 - 2.0) / 0.3 = -4.0  (4 SD below normal)

Gasp: amp=8.2 mV, baseline μ=3.1 mV, σ=0.9 mV
→ amp_zscore = (8.2 - 3.1) / 0.9 = 5.67  (5.67 SD above normal)
```

**Features to Add**:

For each existing breath metric, calculate z-score:
- `if_zscore`: Instantaneous frequency relative to baseline
- `amp_insp_zscore`: Inspiratory amplitude
- `amp_exp_zscore`: Expiratory amplitude
- `ti_zscore`: Inspiratory time
- `te_zscore`: Expiratory time
- `ttot_zscore`: Total breath duration
- `vent_proxy_zscore`: Ventilation proxy (amp × freq)

**Baseline Calculation Options**:

**Option 1: Eupnea Baseline (Recommended)**
```python
def calculate_zscores_eupnea_baseline(breaths_df):
    """Calculate z-scores relative to eupnea breaths only."""

    # Filter to eupnea breaths only (exclude sighs, sniffing)
    eupnea_mask = (breaths_df['is_eupnea'] == 1) & (breaths_df['is_sigh'] == 0)
    eupnea_breaths = breaths_df[eupnea_mask]

    # Calculate baseline statistics from eupnea
    baseline_stats = {
        'if_mean': eupnea_breaths['if'].mean(),
        'if_std': eupnea_breaths['if'].std(),
        'amp_insp_mean': eupnea_breaths['amp_insp'].mean(),
        'amp_insp_std': eupnea_breaths['amp_insp'].std(),
        # ... for each metric
    }

    # Calculate z-scores for ALL breaths (including sighs, sniffs)
    breaths_df['if_zscore'] = (
        (breaths_df['if'] - baseline_stats['if_mean']) /
        baseline_stats['if_std']
    )
    breaths_df['amp_insp_zscore'] = (
        (breaths_df['amp_insp'] - baseline_stats['amp_insp_mean']) /
        baseline_stats['amp_insp_std']
    )
    # ... for each metric

    return breaths_df, baseline_stats
```

**Option 2: Pre-Stimulus Baseline**
```python
def calculate_zscores_prestim_baseline(breaths_df, stim_start_time):
    """Calculate z-scores relative to pre-stimulus period."""

    # Filter to pre-stimulus breaths only
    prestim_mask = breaths_df['t'] < stim_start_time
    prestim_breaths = breaths_df[prestim_mask]

    # Calculate baseline from pre-stim period
    baseline_stats = {
        'if_mean': prestim_breaths['if'].mean(),
        'if_std': prestim_breaths['if'].std(),
        # ... etc.
    }

    # Calculate z-scores for all breaths
    # (same as Option 1)
```

**Option 3: Rolling Window Baseline**
```python
def calculate_zscores_rolling(breaths_df, window_size=30):
    """Calculate z-scores relative to rolling window (e.g., last 30 breaths)."""

    for metric in ['if', 'amp_insp', 'ti', 'te']:
        rolling_mean = breaths_df[metric].rolling(window=window_size).mean()
        rolling_std = breaths_df[metric].rolling(window=window_size).std()

        breaths_df[f'{metric}_zscore'] = (
            (breaths_df[metric] - rolling_mean) / rolling_std
        )

    return breaths_df
```

**Recommended Implementation**: **Option 1 (Eupnea Baseline)**
- Most robust (excludes transient events like sighs)
- Biologically meaningful (normal breathing is the reference)
- Stable baseline (not affected by brief perturbations)

**UI Integration**:
- No UI changes needed - automatically calculated during metrics computation
- Exported in breaths CSV as additional columns
- Used as features for ML classifier

**CSV Export Format**:
```csv
sweep,breath,t,region,if,amp_insp,ti,te,if_zscore,amp_insp_zscore,ti_zscore,te_zscore,...
1,1,-26.09,all,7.27,3.21,0.053,0.085,2.41,0.87,-0.45,1.32,...
1,2,-25.95,all,2.13,3.05,0.048,0.092,-0.21,0.15,-1.12,1.88,...
```

**Performance Considerations**:
- Minimal overhead (just numpy mean/std calculation)
- ~1ms per 1000 breaths
- No impact on UI responsiveness

**Files to modify**:
- `core/metrics.py` (add `calculate_breath_zscores()` function)
- `export/export_manager.py` (include z-score columns in breaths CSV)
- `core/robust_metrics.py` (call z-score calculation after metrics)

**Dependencies**: None (just numpy, already included)

**Effort**: 2-3 hours
- Implement z-score calculation: 1 hour
- Integrate into metrics pipeline: 30 min
- Add to CSV export: 30 min
- Testing with example data: 1 hour

**Future ML Integration**:
```python
# In ML breath classifier:
feature_columns = [
    'if', 'amp_insp', 'ti', 'te', 'vent_proxy',  # Raw features
    'if_zscore', 'amp_insp_zscore', 'ti_zscore', 'te_zscore',  # Z-score features
]

X = breaths_df[feature_columns]
y = breaths_df['breath_type']  # eupnea, sigh, sniff, gasp

# Train model with both raw and normalized features
rf_model.fit(X, y)
```

**Benefits for ML**:
- Improves model generalization across animals
- Reduces overfitting to specific animals in training set
- Enables detection of outliers (z-score > 2.5 = likely sigh/gasp)
- Standard practice in neuroscience ML

---

## Medium Priority Features

### 10. Sniffing Bout Detection and Annotation
**Description**: Automated and manual detection of high-frequency sniffing bouts

**Motivation**:
- Distinguish transient sniffing from sustained sniffing
- Quantify sniffing bout frequency and duration
- Correlate sniffing with behavioral or experimental events

**Features**:
- **Algorithmic detection** based on:
  - Rapid breathing (>7 Hz)
  - Shallow amplitudes (<50% of median eupneic amplitude)
  - Duration >0.5s
- **Manual annotation mode**: Click-and-drag to mark sniffing regions
- **Visual indicators**: Color-coded overlays (e.g., light purple for sniffing bouts)
- **Export**: Sniffing bout timestamps and durations in CSV

**Files to modify**:
- `core/metrics.py` (detection algorithm)
- `main.py` (manual annotation mode, plotting)

**Effort**: 5-6 hours

---

### 11. Expiratory Onset Detection
**Description**: Add separate expiratory onset point (distinct from inspiratory offset)

**Motivation**:
- In rare cases, gap exists between inspiratory offset and expiratory onset
- More accurate breath phase timing
- Better support for unusual breath patterns (post-sigh, breath-holds)

**Implementation**:
- Extend `compute_breath_events()` in `core/peaks.py`
- Detect expiratory onset as first downward zero crossing after inspiratory offset
- Add expiratory onset markers to plots (new color, e.g., magenta)

**UI Changes**:
- Add expiratory onset markers to main plot
- Include in breath event export CSV

**Files to modify**:
- `core/peaks.py` (detection)
- `core/metrics.py` (use expiratory onset in metrics)
- `main.py` (plotting)

**Effort**: 3-4 hours

---

### 12. Dark Mode for Main Plot
**Description**: Toggle dark theme for matplotlib plot area

**Motivation**:
- Consistency with dark UI theme
- Reduced eye strain during long analysis sessions
- Better for presentations in dark rooms

**Features**:
- Checkbox or button: "Dark Plot Theme"
- Changes matplotlib style:
  - Background: black
  - Grid lines: dark gray
  - Text: white
  - Data colors: Adjust for dark background
- Persistent setting (saved to config file)

**Implementation**:
```python
def set_dark_plot_theme(ax, dark_mode: bool):
    if dark_mode:
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    else:
        # Light theme (default matplotlib)
        ax.set_facecolor('white')
        # ... etc.
```

**Files to modify**:
- `main.py` (add checkbox, theme toggle)
- `core/plotting.py` (theme application function)

**Effort**: 2-3 hours

---

## Long-Term / Major Features

### 13. Universal Data Loader Framework (Cross-App Infrastructure)
**Description**: Create modular, reusable file loading system for all neuroscience apps

**Motivation**:
- Unified interface for PlethApp, photometry, Spike2 viewer, Neuropixels
- Write file loader once, use in all apps
- Consistent error handling and validation

**See MULTI_APP_STRATEGY.md for complete implementation plan.**

**Effort**: 12-15 hours (saves 3-5 hours per future app)

---

### 14. ML-Ready Data Export
**Description**: Export format optimized for machine learning training and analysis

**See PUBLICATION_ROADMAP.md for complete implementation plan (Week 1-2 of v1.0 timeline).**

**Effort**: 4-5 hours

---

### 15. Machine Learning Integration
**Description**: Train models on exported labeled data to improve automated detection

**See PUBLICATION_ROADMAP.md for complete implementation plan (Week 3-6 of v1.0 timeline).**

**Effort**: 18-24 hours (phased implementation)

---

## Advanced ML Features (Phase 2)

### Half-Width Features - Robust to Noise

**Problem**: Onset/offset detection can be noisy at breath boundaries
**Solution**: Measure breath width at fixed amplitude percentages (more stable)

```python
def compute_half_width_features(signal, onset_idx, peak_idx, offset_idx):
    """Compute width at various amplitude percentages."""

    baseline = signal[onset_idx]
    peak_amp = signal[peak_idx]
    amplitude_range = peak_amp - baseline

    # Full Width at Half Maximum (FWHM) - most robust!
    half_height = baseline + 0.5 * amplitude_range
    rising = signal[onset_idx:peak_idx]
    falling = signal[peak_idx:offset_idx]

    # Find crossings
    cross_rising = np.where(rising >= half_height)[0]
    cross_falling = np.where(falling >= half_height)[0]

    t_half_rising = onset_idx + cross_rising[0] if len(cross_rising) > 0 else onset_idx
    t_half_falling = peak_idx + cross_falling[-1] if len(cross_falling) > 0 else offset_idx

    fwhm = t_half_falling - t_half_rising

    # Also at 25% and 75% for shape description
    width_25 = compute_width_at_percent(signal, onset_idx, peak_idx, offset_idx, 0.25)
    width_75 = compute_width_at_percent(signal, onset_idx, peak_idx, offset_idx, 0.75)

    return {
        "fwhm": fwhm,                    # Key feature - robust!
        "width_25": width_25,            # Wider, captures base
        "width_75": width_75,            # Narrower, captures peak
        "width_ratio": width_75 / width_25  # Shape descriptor
    }
```

**Why Better Than On/Offset**:
- ✅ Less sensitive to baseline drift
- ✅ Robust to noise at breath boundaries
- ✅ Threshold-independent (uses relative amplitude)
- ✅ Works even if onset/offset detection is imperfect
- ✅ Standard metric in signal processing (used in neuroscience, cardiology)

**Expected Impact**: +2-3% accuracy improvement (87-90% → 90-92%)

---

### Sigh Detection Features - Capturing the Double-Hump

**Problem**: Sighs have characteristic "shoulder" on inspiratory upstroke - how to quantify?
**Solution**: Multiple complementary approaches detect waveform irregularities

#### Method 1: Inflection Point Counting

```python
def detect_inflection_points(signal, onset_idx, peak_idx):
    """Count inflection points in rising phase (sigh signature)."""

    rising = signal[onset_idx:peak_idx]

    # First derivative (velocity)
    dy = np.gradient(rising)

    # Second derivative (acceleration/curvature)
    d2y = np.gradient(dy)

    # Find zero crossings of d2y (inflection points)
    sign_changes = np.diff(np.sign(d2y))
    inflection_points = np.where(sign_changes != 0)[0]

    n_inflections = len(inflection_points)

    # Derivative variability
    dy_variability = np.std(dy) / (np.mean(dy) + 1e-9)

    return {
        "n_inflections": n_inflections,      # Normal: 0-1, Sigh: 2-3
        "rise_variability": dy_variability   # Higher for double-hump
    }
```

**Typical Values**:
- Normal breath: 0-1 inflection points, low variability
- Sigh with shoulder: 2-3 inflection points, high variability
- Artifact: 4+ inflection points (reject as noise)

#### Method 2: Shoulder Peak Detection

```python
def detect_shoulder_peaks(signal, onset_idx, peak_idx, min_prominence=0.1):
    """Detect secondary peaks (shoulders) in rising phase."""

    from scipy.signal import find_peaks

    rising = signal[onset_idx:peak_idx]

    # Find local maxima
    peaks, properties = find_peaks(
        rising,
        prominence=min_prominence * np.ptp(rising),  # 10% of range
        distance=10  # At least 10 samples apart
    )

    # Exclude main peak at end
    shoulder_peaks = peaks[peaks < len(rising) - 5]

    return {
        "n_shoulder_peaks": len(shoulder_peaks),  # 0 for normal, 1+ for sighs
        "shoulder_prominence": np.max(properties["prominences"]) if len(peaks) > 0 else 0
    }
```

#### Method 3: Rise-Phase Autocorrelation

```python
def compute_rise_autocorr(signal, onset_idx, peak_idx):
    """Detect oscillations using autocorrelation."""

    rising = signal[onset_idx:peak_idx]
    rising_norm = (rising - np.mean(rising)) / (np.std(rising) + 1e-9)

    # Autocorrelation
    autocorr = np.correlate(rising_norm, rising_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize

    # Secondary peak indicates periodicity (double-hump oscillation)
    secondary_peak = np.max(autocorr[5:20]) if len(autocorr) > 20 else 0

    return {
        "rise_autocorr_peak": secondary_peak  # High for oscillating sighs
    }
```

#### Method 4: Curvature Features

```python
def compute_curvature_features(signal, onset_idx, peak_idx, offset_idx):
    """Measure sharpness at peak and trough."""

    # Second derivative at peak
    window = 5
    peak_segment = signal[peak_idx-window:peak_idx+window]
    d2y_peak = np.gradient(np.gradient(peak_segment))
    peak_sharpness = np.abs(d2y_peak[window])  # At center

    # Second derivative at trough (expiratory minimum)
    trough_idx = np.argmin(signal[peak_idx:offset_idx]) + peak_idx
    trough_segment = signal[trough_idx-window:trough_idx+window]
    d2y_trough = np.gradient(np.gradient(trough_segment))
    trough_sharpness = np.abs(d2y_trough[window])

    return {
        "peak_sharpness": peak_sharpness,    # Sharp peaks = normal
        "trough_sharpness": trough_sharpness # Sharp troughs = normal
    }
```

#### Method 5: Statistical Shape Features

```python
def compute_statistical_shape(signal, onset_idx, offset_idx):
    """Statistical descriptors of waveform shape."""

    from scipy.stats import skew, kurtosis

    breath = signal[onset_idx:offset_idx]
    breath_norm = (breath - np.mean(breath)) / (np.std(breath) + 1e-9)

    return {
        "skewness": skew(breath_norm),    # Asymmetry
        "kurtosis": kurtosis(breath_norm) # Peakedness
    }
```

**Combined Sigh Detection**: 9 features total
1. n_inflections
2. rise_variability
3. n_shoulder_peaks
4. shoulder_prominence
5. rise_autocorr_peak
6. peak_sharpness
7. trough_sharpness
8. skewness
9. kurtosis

**Expected Impact**: +3-5% accuracy improvement (90-92% → 93-95%)

**Files to Add/Modify**:
- Create `core/advanced_features.py` - all advanced feature functions
- Modify `core/metrics.py` - integrate advanced features
- Modify `export/export_manager.py` - export all 21 features

---

## ML Training Data Requirements & Continuous Learning

### How Many Breaths Do You Need to Train a Good Model?

**TL;DR**: Start with 5-10 files (~15,000-30,000 breaths), diversity matters more than quantity.

#### Minimum Viable Model (MVP)
- **Files**: 5-10 recordings (~10 min each)
- **Total peaks**: ~20,000-40,000 (auto-labeled via elbow detection)
  - ~15,000-30,000 breaths (positive class)
  - ~5,000-10,000 noise (negative class)
- **Diversity needed**:
  - 2-3 different animals
  - Multiple recording sessions
  - Different noise levels
- **Expected performance**: 85-90% accuracy
- **Good enough for**: Initial testing, catching obvious errors

#### Good Performance Model
- **Files**: 20-30 recordings
- **Total peaks**: ~80,000-120,000 peaks
  - ~60,000-90,000 breaths
  - ~20,000-30,000 noise
- **Diversity needed**:
  - 5-10 different animals
  - Multiple experimental conditions (baseline, stim, drugs)
  - Variety of noise types (motion artifacts, electrical noise, etc.)
- **Expected performance**: 92-95% accuracy
- **Good enough for**: Production use, reliable analysis

#### Excellent Performance Model
- **Files**: 50-100 recordings
- **Total peaks**: ~200,000-400,000 peaks
  - ~150,000-300,000 breaths
  - ~50,000-100,000 noise
- **Diversity needed**:
  - 20+ animals
  - Multiple genotypes/strains
  - Different recording setups (hardware, sampling rates)
  - Wide range of experimental manipulations
- **Expected performance**: 96-98% accuracy
- **Good enough for**: Publication-quality automated analysis

#### Key Insights

**Diversity Matters More Than Quantity**:
- 10 files from 10 different animals > 10 files from 1 animal
- Different noise types are crucial (train on what you'll see in practice)
- Edge cases are valuable (sighs, apneas, motion artifacts)

**Class Balance**:
- Your auto-labeling gives ~70% breaths / ~30% noise (ideal!)
- No need for SMOTE or special balancing techniques
- Natural distribution from elbow detection

**Feature Quality**:
- 8 features (ti, te, amp_insp, amp_exp, max_dinsp, max_dexp, prominence, freq)
- Well-tested, biologically meaningful
- Low correlation (each adds unique information)

### How Does Continuous Learning Work?

**Concept**: The model improves as users correct mistakes during normal analysis.

#### Active Learning Workflow

**Phase 1: Initial Training (One-Time)**
```python
# Use auto-labeled data from 10 files
initial_dataset = load_auto_labeled_files(file_list)
# 40,000 peaks with auto-labels

model = RandomForestClassifier()
model.fit(initial_dataset.features, initial_dataset.labels)
# Accuracy: ~85-90% on test set

# Save model
save_model(model, "breath_classifier_v1.pkl")
```

**Phase 2: User Corrections (During Normal Use)**
```python
# User loads file, runs peak detection with ML
all_peaks = detect_peaks(signal, threshold=0, min_distance=0.05)
ml_predictions = model.predict(compute_features(all_peaks))

# UI shows ML predictions
# User reviews: most are correct, but finds 3 mistakes

# USER ACTION: Manually corrects 3 peaks
# - Peak #245: ML said "not_breath", user corrects to "breath"
# - Peak #1821: ML said "breath", user corrects to "not_breath"
# - Peak #3012: ML said "breath", user corrects to "not_breath"

# Corrections are AUTOMATICALLY saved
corrections = {
    "peaks": [245, 1821, 3012],
    "ml_labels": ["not_breath", "breath", "breath"],
    "user_labels": ["breath", "not_breath", "not_breath"],
    "label_source": ["user_corrected", "user_corrected", "user_corrected"],
    "file_id": "20241024_mouse05.abf",
    "timestamp": "2025-01-15 14:32:01"
}

# Saved in NPZ export
npz.save(corrections)
```

**Phase 3: Periodic Retraining (Weekly/Monthly)**
```python
# Collect all user corrections from multiple files
all_corrections = load_corrections_from_all_files()
# 200 user corrections from 20 analyzed files

# Combine with original auto-labeled data
combined_dataset = {
    "auto_labeled": 40,000 peaks,
    "user_corrected": 200 peaks  # GOLD STANDARD!
}

# Weight user corrections higher (they're more reliable)
sample_weights = []
for label_source in combined_dataset.label_sources:
    if label_source == "user_corrected":
        sample_weights.append(10.0)  # 10× more important
    else:
        sample_weights.append(1.0)

# Retrain model
model_v2 = RandomForestClassifier()
model_v2.fit(
    combined_dataset.features,
    combined_dataset.labels,
    sample_weight=sample_weights
)

# Accuracy: ~93-95% (improved!)

# Save updated model
save_model(model_v2, "breath_classifier_v2.pkl")
```

**Phase 4: Model Improvement Over Time**
```
Version 1: 40,000 auto-labeled → 85-90% accuracy
   ↓ (User analyzes 20 files, makes 200 corrections)
Version 2: 40,000 auto + 200 corrected → 93-95% accuracy
   ↓ (User analyzes 50 more files, makes 150 corrections)
Version 3: 40,000 auto + 350 corrected → 95-97% accuracy
   ↓ (User analyzes 100 more files, makes 100 corrections)
Version 4: 40,000 auto + 450 corrected → 96-98% accuracy
   ↓ (Fewer corrections needed as model improves!)
```

#### Implementation Details

**Tracking Label Sources**:
```python
label_sources = [
    "auto",              # From elbow detection
    "auto",
    "user_corrected",    # User manually changed
    "auto",
    "user_added",        # User added missing peak
    "user_removed"       # User deleted false positive
]
```

**Export Format** (saved in NPZ):
```python
ml_training_data = {
    "peaks": [...],
    "labels": ["breath", "not_breath", ...],
    "label_sources": ["auto", "user_corrected", ...],
    "features": {...},
    "model_version_used": "v2",  # Which model made predictions
    "timestamp": "2025-01-15 14:32:01",
    "file_id": "20241024_mouse05.abf"
}
```

**Retraining Script** (run monthly):
```python
# collect_corrections.py
corrections_db = []
for npz_file in Path("analyzed_data").glob("*.npz"):
    data = np.load(npz_file)
    if "label_sources" in data:
        corrections = extract_corrections(data)
        corrections_db.extend(corrections)

print(f"Found {len(corrections_db)} user corrections")
# Found 450 user corrections from 150 files

# retrain_model.py
retrain_model(
    auto_labeled_data="initial_dataset.npz",
    user_corrections=corrections_db,
    output_model="breath_classifier_v4.pkl"
)
```

#### Benefits of Continuous Learning

✅ **Adapts to your data**: Learns patterns specific to your recording setup
✅ **Improves over time**: Every correction makes the model better
✅ **Minimal overhead**: Corrections happen during normal analysis (no extra work)
✅ **Personalized**: Model learns your labeling preferences
✅ **Transparent**: User always has final control (ML is assistive, not autonomous)

#### When to Retrain

**Recommended schedule**:
- **First month**: Retrain weekly (rapid improvement phase)
- **Months 2-3**: Retrain bi-weekly (stabilization phase)
- **After 3 months**: Retrain monthly (maintenance phase)

**Triggers for immediate retraining**:
- Accumulated 50+ new corrections
- New experimental condition (different drug, genotype, etc.)
- Model accuracy drops noticeably

---

### 16. Core Modularization (Breathtools Package)
**Description**: Refactor core analysis functions into standalone, reusable library

**See MULTI_APP_STRATEGY.md for complete extraction plan.**

**Effort**: 8-10 hours

---

### 17. PyPI Publication
**Description**: Publish app to Python Package Index for public `pip install`

**Prerequisites**:
- Choose professional package name (check PyPI availability)
- Add licensing (MIT planned)
- Create proper package structure

**Steps**:
1. Check name availability on PyPI
2. Create `pyproject.toml` with proper metadata
3. Add console script entry point
4. Test with `pip install -e .`
5. Build distribution: `python -m build`
6. Upload to PyPI: `twine upload dist/*`

**Alternative**: Install from GitHub (`pip install git+https://github.com/user/plethapp.git`)

**Effort**: 4-6 hours (first-time setup)

---

## LLM-Assisted Metadata Extraction from Experiment Notes

**Status**: 🔮 Long-term (Post v1.0) - Not high priority

**Description**: Integrate LLM (Claude, ChatGPT, or local model) to intelligently extract experimental metadata from lab notes, Excel files, Word documents, and text files located near data files.

**Motivation**:
- Researchers often keep experiment notes in various formats alongside data files
- Manually entering metadata (animal ID, sex, experimental parameters) is tedious and error-prone
- Notes contain rich contextual information that could auto-populate save dialogs
- Would significantly speed up data organization and reduce manual data entry

**Use Cases**:
1. **Auto-populate Save Metadata Dialog**:
   - LLM scans folder for `.xlsx`, `.docx`, `.txt` files near the `.abf`/`.smrx` file
   - Extracts: Animal ID, sex, genotype, experimental conditions (e.g., "30Hz stim"), channel assignments
   - Pre-fills SaveMetaDialog fields with extracted information
   - User reviews and confirms/edits before saving

2. **Build Internal Metadata Database**:
   - Maintain a JSON/SQLite database mapping each data file to extracted metadata
   - Cache LLM responses to avoid re-processing same notes
   - Enable search/filter by experiment parameters across all recordings

3. **Smart Channel Detection**:
   - Parse notes like "Ch0 = Mouse1 Pleth, Ch1 = Mouse2 Pleth, Ch2 = Stim"
   - Auto-assign channel names based on experiment notes
   - Detect multi-animal recordings and suggest channel groupings

**Implementation Approach**:

**Phase 1: Local File Discovery** (2-3 hours)
- Scan directory and parent directories for common note formats
- Priority order: `.xlsx` > `.docx` > `.txt` (most structured to least)
- File name matching heuristics (e.g., notes with same date/animal ID as data file)

**Phase 2: LLM Integration** (3-4 hours)
- Support multiple backends:
  - **Anthropic Claude API** (recommended - excellent at structured extraction)
  - **OpenAI ChatGPT API** (alternative)
  - **Local LLM** (llama.cpp, Ollama) for offline/privacy-sensitive use
- Structured prompt engineering:
  ```
  Extract experimental metadata from these notes:
  - Animal ID(s)
  - Sex (M/F/Unknown)
  - Genotype (WT, KO, Cre, etc.)
  - Experimental condition (frequency, drug, etc.)
  - Channel assignments (which channel = which animal/signal)

  Notes: [extracted text from files]

  Return JSON: {"animal_id": "...", "sex": "...", ...}
  ```

**Phase 3: UI Integration** (2-3 hours)
- Add "Extract from Notes..." button to SaveMetaDialog
- Progress indicator during LLM processing
- Review dialog showing extracted metadata with confidence scores
- Option to cache/save associations for future use

**Phase 4: Database & Search** (3-4 hours)
- SQLite database: `plethapp_metadata.db`
- Schema: `file_path, animal_id, sex, genotype, condition, channel_map, notes_source, timestamp`
- Search interface: Filter recordings by any metadata field
- Export filtered dataset for batch analysis

**Technical Considerations**:
- **API Costs**: Claude/ChatGPT charges per token - need cost controls and caching
- **Privacy**: Option to use local LLM for sensitive animal data
- **Accuracy**: LLM may hallucinate - always require user review
- **Rate Limits**: Implement queuing for batch processing
- **File Parsing**: Use `openpyxl` (Excel), `python-docx` (Word), built-in for text

**Configuration**:
```python
# In settings or config file
LLM_CONFIG = {
    "backend": "claude",  # or "chatgpt", "local"
    "api_key": "sk-...",  # stored securely
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 500,
    "cache_responses": True,
    "auto_extract": False,  # Manual trigger only by default
}
```

**Example Workflow**:
1. User loads `VgatCre_M25121004_20Hz10s.abf`
2. User clicks "Save Data..." → SaveMetaDialog opens
3. Click "Extract from Notes..." button
4. App finds `Experiment_Notes_Dec2024.xlsx` in same folder
5. LLM extracts: `{"animal_id": "M25121004", "sex": "M", "genotype": "VgatCre", "condition": "20Hz_10s_8mW"}`
6. Fields auto-populate, user confirms and saves

**Risks & Mitigations**:
- **Risk**: LLM extracts wrong information → **Mitigation**: Always show review dialog, never auto-save
- **Risk**: API costs accumulate → **Mitigation**: Aggressive caching, user opt-in, cost tracking
- **Risk**: Privacy concerns → **Mitigation**: Support local LLM, document data handling
- **Risk**: LLM unavailable/slow → **Mitigation**: Timeout, fallback to manual entry

**Dependencies**:
- Anthropic or OpenAI Python SDK
- Optional: `llama-cpp-python` for local models
- `openpyxl`, `python-docx` for file parsing

**Effort Estimate**: 8-12 hours total
- Phase 1: 2-3 hours
- Phase 2: 3-4 hours
- Phase 3: 2-3 hours
- Phase 4 (optional): 3-4 hours

**Priority**: Low (nice-to-have enhancement, not critical for publication)

**Related Features**:
- Could integrate with ML training data preparation
- Metadata database enables better experiment organization
- Foundation for future collaborative/cloud features

---

## Project Builder - Batch Processing Workflow

### Overview
**Description**: Auto-discover and batch process related ABF files based on experiment metadata filters

**Motivation**:
- Speed up repetitive workflows (analyzing dozens of similar experiments)
- Reduce manual file browsing and loading time
- Enable systematic processing of complete datasets
- Auto-advance to next file after saving analysis

**User Story**:
> "I have 47 VgatCre+RVM+ChR2 recordings from the past 3 months. Instead of manually finding and loading each file, I want to define the project once, then step through all files sequentially with auto-advance."

---

### Core Features

#### 1. Project Definition
Define a project with metadata filters to auto-discover matching files:

```python
project_config = {
    "name": "VgatCre_RVM_ChR2_2024",
    "description": "ChR2 activation in RVM Vgat neurons",

    "filters": {
        "strain": "VgatCre",           # Match mouse strain
        "virus": "ChR2",               # Match virus type
        "location": "RVM",             # Match injection site
        "date_range": ("2024-01-01", "2024-12-31"),  # Date range
        "stim_pattern": "30Hz_15s"     # Match stimulus protocol (optional)
    },

    "search_paths": [
        "D:/Experiments/2024/",
        "E:/Backup/VgatCre/",
        "//NetworkDrive/LabData/"     # Network paths supported
    ],

    "file_pattern": "*.abf",           # File extension
    "recursive": True                   # Search subdirectories
}
```

#### 2. File Discovery & Validation

**Auto-discover files**:
- Scan all search paths recursively
- Parse ABF metadata and filenames
- Match against filter criteria
- Validate file integrity (readable, not corrupted)

**Example discovered files**:
```
Found 47 matching files:
✓ 2024_03_15_0003.abf  VgatCre, ChR2, RVM, 30Hz_15s
✓ 2024_03_15_0004.abf  VgatCre, ChR2, RVM, 30Hz_15s
✓ 2024_03_22_0001.abf  VgatCre, ChR2, RVM, 30Hz_15s
...
✓ 2024_11_10_0012.abf  VgatCre, ChR2, RVM, 30Hz_15s
```

**Metadata extraction sources**:
1. **Filename parsing**: Extract date, animal ID from standard formats
   - `YYYYMMDD_####.abf` → Date extraction
   - `VgatCre_M25121004_*.abf` → Strain, animal ID
2. **ABF file metadata**: Sample rate, channels, protocol name
3. **Companion files**: Look for `.txt`, `.xlsx` notes files with metadata
4. **Previous analysis**: Load metadata from existing `.pleth.npz` session files

#### 3. Project Browser UI

**Main project window**:
```
┌────────────────────────────────────────────────────┐
│ Project: VgatCre_RVM_ChR2_2024                     │
│ Description: ChR2 activation in RVM Vgat neurons   │
├────────────────────────────────────────────────────┤
│ Files: 47 total  │  Analyzed: 12 (26%)  │  Pending: 35  │
│                                                    │
│ Current File (13/47):                              │
│ 📄 2024_05_18_0002.abf                            │
│ └─ IN_0: 10 sweeps, 20kHz, 15min                 │
│                                                    │
│ Status: ✓ Analyzed, saved to .pleth.npz          │
├────────────────────────────────────────────────────┤
│ [◀ Previous]  [Next ▶]  [Jump to #...]  [Filter] │
│                                                    │
│ Auto-advance after save: ☑                        │
│ Skip already analyzed:   ☑                        │
└────────────────────────────────────────────────────┘
```

**File list with filtering**:
```
┌────────────────────────────────────────────────────┐
│ File List - Sort by: Date ▼                       │
├────────────────────────────────────────────────────┤
│ ☑ 2024_03_15_0003.abf    ✓ Analyzed   [Jump]     │
│ ☑ 2024_03_15_0004.abf    ✓ Analyzed   [Jump]     │
│ ☐ 2024_03_22_0001.abf    ⚠ Pending    [Jump]     │
│ ☐ 2024_04_05_0007.abf    ⚠ Pending    [Jump]     │
│ ...                                                │
└────────────────────────────────────────────────────┘

Filters:
☑ Show analyzed    ☑ Show pending    ☐ Show errors
```

#### 4. Auto-Advance Workflow

**Standard workflow**:
1. User defines project → files discovered
2. Load first pending file
3. User analyzes (detect peaks, edit, review)
4. Press Ctrl+S → Save analysis dialog
5. After save completes → **Auto-load next pending file**
6. Repeat until all files processed

**Smart features**:
- **Skip analyzed files**: Only show files without `.pleth.npz` sessions
- **Resume from last**: Remember last analyzed file, continue from there
- **Bookmark files**: Mark specific files for later review
- **Batch export**: Export all analyzed files at once (CSV, summary PDF)

#### 5. Progress Tracking & Bookmarks

**Project state file** (`.plethproject.json`):
```json
{
  "project_name": "VgatCre_RVM_ChR2_2024",
  "created": "2024-10-28T10:30:00",
  "last_opened": "2024-10-29T14:22:00",
  "total_files": 47,
  "analyzed_files": 12,
  "current_file_index": 12,

  "file_status": {
    "2024_03_15_0003.abf": {"status": "analyzed", "timestamp": "2024-10-28T11:05:23"},
    "2024_03_15_0004.abf": {"status": "analyzed", "timestamp": "2024-10-28T11:32:18"},
    "2024_03_22_0001.abf": {"status": "pending"},
    ...
  },

  "bookmarks": [
    {"file": "2024_05_10_0003.abf", "note": "Unusual breathing pattern - review"},
    {"file": "2024_08_15_0001.abf", "note": "High noise - may need reanalysis"}
  ]
}
```

**Visual progress indicators**:
- Progress bar: Files analyzed vs total
- Color coding: Green (analyzed), Yellow (in progress), Gray (pending), Red (errors)
- Statistics: Average analysis time per file, estimated time remaining

---

### Implementation Plan

**Phase 1: Project Definition & Discovery** (2-3 hours)
- Create `ProjectConfig` dataclass
- Implement file discovery with metadata parsing
- Filter matching based on criteria
- Save/load project state to JSON

**Files to create**:
- `core/project_builder.py` - Core logic
- `dialogs/project_config_dialog.py` - UI for defining projects

**Phase 2: Project Browser UI** (2-3 hours)
- Main project window with file list
- Navigation controls (prev/next/jump)
- Progress tracking display
- Bookmark management

**Files to modify**:
- `main.py` - Add "Open Project" menu item
- Create `dialogs/project_browser_dialog.py`

**Phase 3: Auto-Advance Integration** (2 hours)
- Hook into save workflow
- Auto-load next file after save
- Skip already analyzed files
- Smart resumption (continue from last file)

**Files to modify**:
- `main.py` - Integrate auto-advance in save callback
- `core/project_builder.py` - Next file logic

**Phase 4: Batch Export** (Optional, 2 hours)
- Export all analyzed files in project
- Combined CSV with all sweeps
- Summary statistics across entire project
- Progress dialog for batch operations

---

### Future Enhancements

**Integration with ML predictions** (Post v1.0):
- Auto-analyze entire project with ML model
- Show only files with uncertain predictions (confidence < 70%)
- Batch re-train model on all corrected files

**Cloud/network support**:
- Discover files on network drives
- Multi-user project collaboration
- Lock files during analysis to prevent conflicts

**Advanced filtering**:
- Custom filter expressions (e.g., "IF > 2 Hz AND Ti < 0.3s")
- Filter by analysis results (e.g., "files with >10 sighs")
- Regex patterns for filename matching

---

### User Benefits

**Time savings**:
- **No manual browsing**: One-time project setup, auto-discover all files
- **No repetitive loading**: Auto-advance eliminates 10-20 seconds per file
- **Batch operations**: Export all analyses with one click

**Reduced errors**:
- **Complete coverage**: Won't accidentally skip files
- **Consistent workflow**: Same analysis pipeline for all files
- **Progress tracking**: Know exactly which files are done

**Organization**:
- **Logical grouping**: Organize files by experiment type
- **Persistent state**: Resume anytime, remember progress
- **Documentation**: Project config serves as analysis record

**Estimated productivity gain**: **2-3× faster** for batch processing 20+ files

---

## Potential Future Directions (Speculative)

These features are not yet prioritized but represent potential future directions:

- **Code signing** for professional Windows distribution
- **Automated testing framework** with edge case coverage
- **Additional file format support**: WFDB, HDF5, TDT, etc.
- **Real-time data acquisition** capabilities (integrate with DAQ hardware)
- **Advanced statistical analysis modules**: Wavelet coherence, time-frequency analysis
- **Plugin architecture** for custom user algorithms
- **Cloud-based batch processing** and collaboration features
- **Web interface** for remote access and analysis

---

## Feature Prioritization Criteria

When deciding which features to implement next, consider:

1. **Impact**: How many users will benefit?
2. **Effort**: How long will it take to implement?
3. **Risk**: How likely is it to introduce bugs?
4. **Dependencies**: Does it require other features first?
5. **Publication value**: Does it strengthen the JOSS paper?
6. **Community requests**: Are users asking for this?

**High-impact, low-effort features** should be prioritized.

---

## Related Documentation
- **PUBLICATION_ROADMAP.md**: v1.0 publication timeline and ML features
- **PERFORMANCE_OPTIMIZATION_PLAN.md**: Performance improvement priorities
- **MULTI_APP_STRATEGY.md**: Long-term multi-app ecosystem vision
- **RECENT_FEATURES.md**: Recently implemented features
