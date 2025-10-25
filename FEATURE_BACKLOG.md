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

### 🎯 High Priority (Next Up)
- [ ] **NPZ file save/load with full state restoration** (4-6 hours) ← RECOMMENDED NEXT
- [ ] Help button with app usage guide and typical workflow (3-4 hours)
- [ ] Display mode toggles for breathing states - background vs line (4-5 hours)

### 📋 Medium Priority (Scientific Features)
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

### 4. CSV/Text Time-Series Import
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
