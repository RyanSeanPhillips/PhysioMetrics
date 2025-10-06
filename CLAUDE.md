# PlethApp - Breath Analysis Application

## Overview
PlethApp is a PyQt6-based desktop application for advanced respiratory signal analysis. It provides comprehensive tools for breath pattern detection, eupnea/apnea identification, and breathing regularity assessment.

## Project Structure
```
plethapp/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── pleth_app.spec            # PyInstaller configuration
├── build_executable.bat      # Windows build script
├── build_executable.py       # Cross-platform build script
├── version_info.py           # Version metadata generator
├── run_debug.py              # Debug launcher for testing
├── BUILD_INSTRUCTIONS.md     # Detailed build documentation
├── core/                     # Core application modules
│   ├── state.py             # Application state management
│   ├── abf_io.py            # ABF file I/O operations (dispatcher for ABF and SMRX)
│   ├── filters.py           # Signal filtering functions
│   ├── plotting.py          # Matplotlib integration and plot management
│   ├── stim.py              # Stimulus detection algorithms
│   ├── peaks.py             # Peak detection and breath onset/offset analysis
│   ├── metrics.py           # Breathing metrics and pattern analysis
│   ├── navigation.py        # Data navigation utilities
│   ├── editing.py           # Manual peak editing tools
│   ├── export.py            # Data export functionality
│   └── io/                  # File format loaders
│       ├── son64_dll_loader.py   # CED SON64 DLL wrapper (low-level)
│       ├── son64_loader.py       # SMRX loader for PlethApp (high-level)
│       └── s2rx_parser.py        # Spike2 .s2rx XML configuration parser
├── ui/                      # PyQt6 UI definition files
│   └── pleth_app_layout_02.ui # Main application UI layout
├── images/                  # Application icons and assets
├── assets/                  # Additional application assets
└── examples/               # Sample data files
```

## Key Features
- **Advanced Peak Detection**: Sophisticated algorithms for detecting inspiratory peaks, expiratory minima, and breath onsets/offsets
- **Enhanced Expiratory Offset Detection**: Uses both signal zero crossings and derivative analysis with amplitude constraints
- **Eupnea Detection**: Identifies regions of normal, regular breathing patterns
- **Apnea Detection**: Detects breathing gaps longer than configurable thresholds (default: 0.5 seconds)
- **Breathing Regularity Score**: RMSSD-based assessment of breathing pattern variability
- **Real-time Visual Overlays**: Automatic green/red line overlays for eupnea and apnea regions
- **Interactive Data Navigation**: Window-based and sweep-based data exploration
- **Manual Peak Editing**: Add/delete peaks and annotate sighs with keyboard shortcuts (Shift/Ctrl modifiers)
- **Spectral Analysis Window**: Power spectrum, wavelet scalogram, and notch filter configuration
- **Data Export**: CSV export of analyzed breathing metrics
- **Multi-format Support**: ABF and Spike2 SMRX (.smrx) file format support with extensible I/O architecture

## Core Algorithms

### Peak Detection (core/peaks.py)
- **find_peaks()**: Main peak detection using scipy with configurable threshold, prominence, and distance parameters
- **compute_breath_events()**: **ENHANCED** robust breath event detection with multiple fallback strategies:
  - **Robust Onset Detection**: Zero crossing in y signal → dy/dt crossing → fixed fraction fallback → boundary-based fallback
  - **Robust Offset Detection**: Similar multi-level fallback approach ensuring valid breath boundaries
  - **Enhanced Expiratory Detection**: Derivative zero crossing → actual minimum fallback → midpoint estimation
  - **Expiratory Offset Detection**: **ENHANCED** dual-candidate method:
    - Finds both signal zero crossing AND derivative zero crossing with 50% amplitude threshold
    - Selects whichever occurs EARLIER (more physiologically accurate timing)
    - Robust fallbacks for edge cases
  - **Edge Effect Protection**: Special handling for peaks near trace boundaries
  - **Consistent Output**: Always returns arrays of appropriate lengths with bounds checking

### Breathing Pattern Analysis (core/metrics.py + core/robust_metrics.py)
- **detect_eupnic_regions()**: Identifies normal breathing regions using frequency (<5Hz) and duration (≥2s) criteria
- **detect_apneas()**: Detects breathing gaps based on inter-breath intervals
- **compute_regularity_score()**: RMSSD calculation with performance-optimized decimated sampling
- **compute_breath_metrics()**: Comprehensive breath-by-breath analysis including:
  - Breath duration and frequency
  - Peak amplitudes and timing
  - Inter-breath intervals
  - Respiratory variability measures

#### **NEW: Robust Metrics Framework (core/robust_metrics.py)**
Advanced error-resistant metrics calculation system:
- **Graceful Degradation**: Continues processing when individual breath cycles fail
- **Multiple Fallback Strategies**: Each metric has 3-4 fallback methods
- **Bounds Protection**: Comprehensive array bounds checking prevents crashes
- **Misaligned Data Handling**: Works with arrays of different lengths
- **NaN Isolation**: Failed calculations don't cascade to other cycles
- **Consistent Output**: Always produces valid arrays regardless of input quality

**Key Robust Metrics:**
- `robust_compute_if()`: Instantaneous frequency with onset→onset, peak→peak, and estimation fallbacks
- `robust_compute_amp_insp()`: Inspiratory amplitude with baseline detection strategies
- `robust_compute_ti()`: Inspiratory timing with direct, estimated, and default calculations

### Signal Processing (core/filters.py)
- Low-pass and high-pass Butterworth filtering with adjustable filter order
- Mean subtraction with configurable time windows
- Signal inversion capabilities
- Real-time filter parameter adjustment
- **Notch (band-stop) filtering** for removing specific frequency ranges
- **Spectral analysis tools** for identifying noise contamination

### File Format Support (core/io/)

#### Spike2 SMRX Files (SON64 Format)
PlethApp supports reading Spike2 .smrx files using the official CED SON64 library.

**Implementation:**
- **core/io/son64_dll_loader.py**: Low-level ctypes wrapper for `ceds64int.dll`
- **core/io/son64_loader.py**: High-level loader that converts SMRX data to PlethApp format
- **core/io/s2rx_parser.py**: Parser for Spike2 `.s2rx` XML configuration files
- **Dependencies**: Requires CED MATLAB SON library (CEDS64ML) installed at `C:\CEDMATLAB\CEDS64ML\`

**Key Technical Details:**
- Uses `S64ChanDivide` to get actual sample interval in ticks (critical for correct timing)
- Time calculation: `time[i] = (first_tick + i * chan_divide) * time_base`
- Handles multi-segment data (files with gaps) by reading segments and concatenating
- Resamples channels to common time grid using scipy.interpolate.interp1d
- **IMPORTANT**: SMRX files must be closed in Spike2 before opening in PlethApp (CED DLL uses exclusive file locking)

**Supported Features:**
- Waveform channels (ADC and ADCMARK types)
- Multiple sample rates (automatically aligned to lowest rate to avoid upsampling artifacts)
- Full time range with accurate tick-based timing
- Channel metadata (titles, units, scale, offset)
- **Automatic channel visibility filtering**: Reads `.s2rx` configuration files to hide channels marked as hidden in Spike2
  - Looks for `.s2rx` file with same name as `.smrx` file
  - Respects `Vis="0"` attribute in channel settings
  - Defaults to showing all channels if `.s2rx` not found
  - Channels not mentioned in `.s2rx` default to visible

**Example Usage:**
```python
from core.io.son64_loader import load_son64

# Load file (returns PlethApp format)
sr_hz, sweeps_by_channel, channel_names, t = load_son64('file.smrx')
# sweeps_by_channel[channel_name] -> shape (n_samples, 1) - continuous recording
```

**Troubleshooting:**
- Error code -1 from S64Open: File is locked (close in Spike2) or path issue
- Data duplication: Fixed by using S64ChanDivide instead of calculating from sample rate
- Missing channels: Only waveform channels (kind 1 or 7) are loaded

## Development Commands

### Running in Development Mode
```bash
python run_debug.py
```
This launches the application with import checking and debug output.

### Building Executable
```bash
# Method 1: Using batch script (Windows)
build_executable.bat

# Method 2: Using Python script (cross-platform)
python build_executable.py

# Method 3: Manual PyInstaller
python version_info.py
pyinstaller --clean pleth_app.spec
```

### Testing
- Test the source application using `python run_debug.py`
- Test the built executable on a clean machine without Python installed
- Verify all features work: file loading, peak detection, filtering, export

## Lint & Typecheck Commands
Currently no formal linting setup. Consider adding:
```bash
# Future recommendations:
# pip install flake8 mypy
# flake8 core/ main.py
# mypy core/ main.py
```

## Architecture Notes

### State Management
The application uses a centralized state system (`core.state.AppState`) that manages:
- Current data and metadata
- Peak detection results
- Filter parameters
- Navigation state
- User annotations

### UI Architecture
- **Main Window**: `pleth_app_layout_02.ui` - Grid-based layout with left-aligned controls
- **Plotting Integration**: Custom `PlotHost` class managing matplotlib figures within PyQt6
- **Responsive Design**: MainPlot stretches with window, controls remain left-justified
- **Dark Theme**: Custom CSS styling optimized for scientific data visualization

#### **IMPORTANT: Qt Designer Left-Alignment Fix**
Qt Designer tends to remove alignment attributes when saving `.ui` files. To maintain left-justified controls:

1. **Main Grid Layout** (`gridLayout_3`):
   - Row 0: Add `alignment="Qt::AlignLeft|Qt::AlignTop"` to item containing `verticalLayout_8`
   - Row 1: **NO alignment** on MainPlot (must expand to fill space)
   - Row 2: Add `alignment="Qt::AlignLeft|Qt::AlignTop"` to item containing `horizontalLayout_12`

2. **Top Controls Container** (`verticalLayout_8`):
   - Add layout property: `<property name="alignment"><set>Qt::AlignLeft|Qt::AlignTop</set></property>`

3. **Horizontal Layouts Inside verticalLayout_8**:
   - `horizontalLayout_7` (Browse/File Selection)
   - `horizontalLayout_8` (Channel Selection)
   - `horizontalLayout_9` (Filters)
   - `horizontalLayout_10` (Peak Detection)
   - Each needs: `<property name="alignment"><set>Qt::AlignLeft|Qt::AlignVCenter</set></property>`

**Example XML snippet for horizontal layout:**
```xml
<layout class="QHBoxLayout" name="horizontalLayout_7">
 <property name="alignment">
  <set>Qt::AlignLeft|Qt::AlignVCenter</set>
 </property>
 <item>
  <!-- widgets here -->
 </item>
</layout>
```

**Note**: After editing in Qt Designer, these alignment properties may be removed. Re-apply them using the Edit tool or a text editor.

### Performance Optimizations
- **Decimated Sampling**: Metrics calculations use reduced sample rates (~0.1s intervals) for performance
- **Lazy Loading**: Data loaded on-demand to minimize memory usage
- **Efficient Plotting**: Optimized matplotlib backend integration

### Build System
- **PyInstaller Configuration**: Comprehensive `.spec` file with proper dependency handling
- **Directory Distribution**: Fast startup times (~6 seconds) vs single-file distribution
- **Icon Integration**: Custom application icons for Windows executable
- **Dependency Management**: Careful exclusion of conflicting Qt bindings and unused modules

## Deployment Notes
- **Target Platform**: Windows 10/11 (primary), extensible to other platforms
- **Distribution Size**: ~200-400MB due to scientific libraries
- **Startup Time**: ~6 seconds (directory distribution)
- **Runtime Dependencies**: Self-contained executable with no Python installation required

## Robustness Features

### **Peak Detection Robustness**
- **Multi-level Fallbacks**: Each breath event type uses 3-4 detection strategies
- **Edge Effect Handling**: Special processing for peaks near trace boundaries
- **Noisy Signal Tolerance**: Derivative filtering and amplitude constraints reduce false detections
- **Emergency Fallbacks**: Creates reasonable estimates when all primary methods fail

### **Metrics Calculation Robustness**
- **Isolated Failures**: One bad breath cycle doesn't break entire sweep analysis
- **Array Length Mismatches**: Handles cases where onsets, offsets, and peaks arrays have different lengths
- **Bounds Checking**: All array accesses are validated to prevent index errors
- **Graceful Degradation**: Returns partial results when possible rather than complete failure

### **Usage: Enabling Robust Mode**
To enable the enhanced robust metrics (optional):

1. **Uncomment the integration code** in `core/metrics.py` (lines 966-971):
   ```python
   try:
       from core.robust_metrics import enhance_metrics_with_robust_fallbacks
       METRICS = enhance_metrics_with_robust_fallbacks(METRICS)
       print("Enhanced metrics with robust fallbacks enabled.")
   except ImportError:
       print("Robust metrics module not available. Using standard metrics.")
   ```

2. **Restart the application** - the robust metrics will be used automatically for all calculations

## Recent Feature Additions

### Spectral Analysis Window (2025-10-02)
A comprehensive spectral analysis tool for identifying and filtering oscillatory noise contamination:

**Features:**
- **Power Spectrum (Welch method)**: High-resolution frequency analysis (0-30 Hz range optimized for breathing)
  - Separate spectra for full trace (blue) and during-stimulation periods (orange)
  - nperseg=32768, 90% overlap for maximum resolution
- **Wavelet Scalogram**: Time-frequency analysis using complex Morlet wavelets
  - Frequency range: 0.5-30 Hz
  - Time normalized to stimulation onset (t=0)
  - Percentile-based color scaling (95th) to handle transient sniffing bouts
  - Stim on/offset markers (lime green dashed lines)
- **Notch Filter Controls**: Interactive band-stop filter configuration
  - Specify lower and upper frequency bounds
  - 4th-order Butterworth band-stop filter
  - Applied to main signal when dialog is closed
- **Sweep Navigation**: Step through sweeps within the spectral analysis view
- **Aligned Panels**: GridSpec layout ensures power spectrum and wavelet plots have matching edges

**Implementation Details:**
- Located in `main.py` as `SpectralAnalysisDialog` class (lines ~1970-2330)
- Button added to filter controls: `SpectralAnalysisButton`
- Notch filter integrated into signal processing pipeline (`_current_trace()`, `_apply_notch_filter()`)
- Filter parameters included in cache key (`_proc_key()`) to trigger recomputation

**To Remove This Feature:**
1. Delete `SpectralAnalysisDialog` class from `main.py`
2. Remove `SpectralAnalysisButton` from `ui/pleth_app_layout_02.ui` (horizontalLayout_9)
3. Remove notch filter code from `_current_trace()` (lines ~621-623)
4. Remove `_apply_notch_filter()` method (lines ~1164-1193)
5. Remove notch filter from `_proc_key()` cache key (lines ~369-370)
6. Remove notch filter instance variables from `__init__()` (lines ~63-65)
7. Remove button connection: `self.SpectralAnalysisButton.clicked.connect(...)` (line ~116)
8. Remove `on_spectral_analysis_clicked()` method (lines ~1816-1863)

### Adjustable Filter Order (2025-10-02)
Added UI control for Butterworth filter order to enable stronger frequency attenuation:

**Features:**
- **Filter Order Spinbox**: Range 2-10, default 4
  - Located in filter controls (horizontalLayout_9)
  - Higher order = steeper roll-off at cutoff frequency
  - More aggressive elimination of frequencies beyond cutoff
- **Cache Integration**: Filter order included in processing cache key
- **Real-time Updates**: Changes trigger immediate signal reprocessing

**Implementation Details:**
- UI widget: `FilterOrderSpin` (QSpinBox) in `ui/pleth_app_layout_02.ui` (lines ~834-853)
- Label: `FilterOrderLabel` (lines ~813-831)
- Connected to `update_and_redraw()` via `valueChanged` signal (line ~107)
- Stored in `self.filter_order` instance variable (line ~68)
- Passed to `filters.apply_all_1d()` as `order` parameter (line ~618)
- Included in `_proc_key()` for cache invalidation (line ~368)

**To Remove This Feature:**
1. Delete `FilterOrderSpin` and `FilterOrderLabel` from `ui/pleth_app_layout_02.ui` (lines ~812-854)
2. Remove `self.filter_order` instance variable from `__init__()` (line ~68)
3. Remove spinbox connection from `__init__()` (line ~107)
4. Remove filter order update in `update_and_redraw()` (line ~544)
5. Remove `order=self.filter_order` from `_current_trace()` call to `apply_all_1d()` (line ~618)
6. Remove filter order from `_proc_key()` cache key (line ~368)

### Manual Peak Editing Enhancements (2025-10-02)
Improved peak editing workflow with keyboard modifiers and precision controls:

**Features:**
- **Keyboard Shortcuts**:
  - Shift key: Toggle to Delete Peak mode from Add Peak mode (and vice versa)
  - Ctrl key: Switch to Add Sigh mode from any mode
  - Allows quick mode switching without button clicks
- **Precise Peak Deletion**: Only deletes the single closest peak within ±80ms window
  - Previous behavior: deleted all peaks in window
  - New behavior: finds closest peak using `np.argmin(distances)` and deletes only that one

**Implementation Details:**
- Mode switching in `_on_plot_click_add_peak()` and `_on_plot_click_delete_peak()`
- Uses `QApplication.keyboardModifiers()` to detect Shift/Ctrl
- `_force_mode` parameter prevents infinite recursion
- Button labels updated to show shortcuts (e.g., "Add Peak (Shift: Delete)")

**To Revert to Previous Behavior:**
1. Remove modifier key detection from `_on_plot_click_add_peak()` and `_on_plot_click_delete_peak()`
2. Change delete logic back to: `pks_new = pks[(pks < i_lo) | (pks > i_hi)]` (deletes all in window)

## Development Roadmap

### High Priority (Next Implementation Phase)

#### 1. High-Resolution Splash Screen
- **Description**: Replace current splash screen with higher resolution image for better visual presentation
- **Files to modify**: `run_debug.py`, image assets
- **Effort**: 30 minutes

#### 2. NPZ File Save/Load with Full State Restoration
- **Description**: Save all analysis data (traces, peaks, metrics, annotations) to NPZ file for later review
- **Features**:
  - Save: raw data, processed traces, detected peaks/events, calculated metrics, manual edits, filter settings
  - Load: Restore complete analysis state - user can review, verify, or modify previous work
  - Auto-populate filename when re-saving (update existing NPZ file)
  - Versioned format with backward compatibility checking
- **Use case**: Quality control, collaborative review, incremental analysis sessions
- **File structure**:
  ```python
  npz_data = {
      'raw_sweeps': dict,          # Original channel data
      'sr_hz': float,
      'channel_names': list,
      'peaks_by_sweep': dict,      # All detected features
      'breath_metrics': dict,      # Computed metrics
      'manual_edits': dict,        # User annotations
      'filter_params': dict,       # Processing settings
      'version': str               # Format version for compatibility
  }
  ```
- **Files to modify**: `core/export.py` (add save_npz/load_npz), `main.py` (file dialog, state restoration)
- **Effort**: 5-6 hours

#### 3. Multi-File ABF Concatenation
- **Description**: Load multiple ABF files as concatenated sweeps
- **Features**:
  - Multi-select in file dialog → files treated as sequential sweeps
  - Validation checks:
    - Same number of channels across all files
    - Same channel names
    - Same sample rate
    - Same date (extracted from filename format: `YYYYMMDD####.abf`)
  - Warning dialog if checks fail with option to proceed anyway
  - Display concatenated file info in status bar: "File 1 of 3: 2024010_0001.abf (Sweeps 0-9)"
- **Files to modify**: `main.py` (file dialog, validation), `core/abf_io.py` (concatenation logic)
- **Effort**: 4-5 hours

#### 4. CSV/Text Time-Series Import
- **Description**: Load arbitrary time-series data from CSV/text files
- **Features**:
  - File preview dialog showing first 20 rows
  - Column selection UI: user picks time column + data columns
  - Auto-detect headers, delimiter (comma/tab/space), decimal separator
  - Sample rate detection: auto-calculate from time column or user-specified
  - Map columns to "sweeps" (each selected column becomes a sweep)
- **UI Components**:
  - `CSVImportDialog` with table preview and column selectors
  - Checkbox: "First row is header"
  - Spinbox: "Sample rate (Hz)" with auto-detect option
- **Files to create**: `core/io/csv_loader.py`, `CSVImportDialog` in `main.py`
- **Effort**: 6-7 hours

#### 5. ✅ Spike2 .smrx File Support (COMPLETED)
- **Status**: ✅ **IMPLEMENTED** (2025-10-05)
- **Implementation**: Direct CED DLL wrapper using official CEDS64ML library
- **Features**:
  - Read .smrx (SON64) format using ctypes wrapper for `ceds64int.dll`
  - Multi-segment waveform reading with gap handling
  - Accurate tick-based timing using `S64ChanDivide`
  - Multi-rate channel support with automatic alignment
  - Full channel metadata extraction
- **Dependencies**: Requires CED MATLAB SON library installed at `C:\CEDMATLAB\CEDS64ML\`
- **Files created**: `core/io/son64_dll_loader.py`, `core/io/son64_loader.py`
- **Files modified**: `core/abf_io.py` (dispatcher), `main.py` (file dialog)
- **Note**: Files must be closed in Spike2 before opening in PlethApp (exclusive file lock)
- **See**: "File Format Support" section above for detailed documentation

#### 6. Move Point Editing Mode
- **Description**: Add button/mode to manually drag and reposition detected peaks (inspiratory, expiratory, onsets, offsets)
- **Implementation**: New editing mode with click-and-drag functionality
- **Use case**: Fine-tune automated detection results
- **Files to modify**: `main.py` (editing modes), `core/editing.py`
- **Effort**: 3-4 hours

#### 7. Enhanced Eupnea Threshold Controls
- **Description**: Convert "Eupnea Thresh (Hz)" label to clickable button/link
- **Features**:
  - Opens dialog with all eupnea detection parameters (frequency threshold, duration threshold, regularity criteria)
  - Manual mode: User can click to highlight/annotate eupnic regions
  - Visual feedback with region overlays
- **Files to modify**: `main.py`, possibly new `EupneaControlDialog` class
- **Effort**: 4-5 hours

#### 8. Enhanced Outlier Threshold Controls
- **Description**: Convert "Outlier Thresh (SD)" label to clickable button/link
- **Features**:
  - Opens dialog to select which metrics to use for outlier detection (Ti, frequency, amplitude, etc.)
  - Multi-metric selection with individual SD thresholds
  - Preview of flagged breaths before applying
- **Files to modify**: `main.py`, `core/metrics.py`
- **Effort**: 4-5 hours

#### 9. Statistical Significance in Consolidated Data
- **Description**: Add statistical testing to identify when stim response differs significantly from baseline
- **Features**:
  - Three new columns in consolidated CSV output:
    - `cohens_d`: Effect size at each timepoint (mean - baseline_mean) / baseline_sd
    - `p_value`: Uncorrected paired t-test p-value (timepoint sweeps vs baseline sweeps)
    - `sig_corrected`: Boolean flag after Bonferroni correction (p < 0.05/n_timepoints)
  - Visual enhancements in consolidated plot:
    - Shaded gray background for significant regions
    - Asterisks for significance levels: `*` p<0.05, `**` p<0.01, `***` p<0.001
    - Horizontal dashed lines at Cohen's d = ±0.5 (medium effect size)
  - User-configurable options:
    - Baseline window (default: -2 to 0 sec pre-stim)
    - Significance threshold (default: 0.05)
    - Correction method: Bonferroni (conservative) or None
- **Alternative methods** (future consideration):
  - Cluster-based permutation testing for sustained effects
  - Confidence interval non-overlap flagging
- **Files to modify**: `main.py` (consolidation dialog, plotting), `core/metrics.py` (statistical helpers)
- **Dependencies**: scipy.stats (already included)
- **Effort**: 4-5 hours

### Medium Priority

#### 10. Sniffing Bout Detection and Annotation
- **Description**: Automated and manual detection of high-frequency sniffing bouts
- **Features**:
  - Algorithmic detection based on rapid, shallow breathing patterns
  - Manual annotation mode: click-and-drag to mark sniffing regions
  - Visual indicators (color-coded overlays similar to eupnea/apnea)
- **Files to modify**: `core/metrics.py`, `main.py`
- **Effort**: 5-6 hours

#### 11. Expiratory Onset Detection
- **Description**: Add separate expiratory onset point (distinct from inspiratory offset)
- **Rationale**: Rare cases have gap between inspiratory offset and expiratory onset
- **Implementation**: Extend `compute_breath_events()` in `core/peaks.py`
- **UI changes**: Add expiratory onset markers to plots
- **Files to modify**: `core/peaks.py`, `core/metrics.py`, `main.py` (plotting)
- **Effort**: 3-4 hours

#### 12. Dark Mode for Main Plot
- **Description**: Toggle dark theme for matplotlib plot area (background, grid, text colors)
- **Implementation**: Add checkbox/button to switch between light and dark plot themes
- **Files to modify**: `main.py` (plotting section), possibly `core/plotting.py`
- **Effort**: 2-3 hours

### Long-Term / Major Features

#### 13. Universal Data Loader Framework (Cross-App Infrastructure)
- **Description**: Create modular, reusable file loading system for all neuroscience apps
- **Motivation**: Unified interface for PlethApp, photometry analysis, Spike2 viewer, Neuropixels pipeline
- **Architecture**:
  ```
  breathtools_io/  (or neuro_io/)
  ├── __init__.py
  ├── base.py              # Abstract DataLoader protocol
  ├── loaders/
  │   ├── abf.py          # Axon Binary Format (pyabf)
  │   ├── smrx.py         # Spike2 (neo/sonpy)
  │   ├── csv.py          # Generic CSV/text
  │   ├── spikeglx.py     # Neuropixels (spikeglx-python)
  │   ├── tdt.py          # Tucker-Davis systems
  │   └── photometry.py   # Multi-file photometry workflows
  ├── registry.py          # Auto-detect file format
  └── utils.py            # Concatenation, validation, alignment
  ```
- **Unified Interface**:
  ```python
  from breathtools_io import load_data

  # Auto-detect format or specify explicitly
  data = load_data("file.abf")  # Returns standardized dict
  data = load_data(["f1.csv", "f2.csv"], format="photometry")

  # All loaders return:
  {
      'signals': dict[str, np.ndarray],  # channel -> (samples, trials)
      'sample_rate': float,
      'time': np.ndarray,
      'metadata': dict,
      'source_files': list[Path]
  }
  ```
- **Multi-File Strategies**:
  - `merge_strategy="concatenate"`: Sequential sweeps (PlethApp ABF workflow)
  - `merge_strategy="align_timestamps"`: Sync multi-channel photometry
  - `merge_strategy="interleave"`: Alternating channels
  - Validation: same date, same channels, same sample rate
- **Benefits**:
  - Write file loader once, use in all apps
  - Easy to add new formats (just add new loader class)
  - Consistent error handling and validation
  - Facilitates data sharing between pipelines
- **Distribution**: Standalone pip package or shared module across projects
- **Files to create**: New repository `breathtools_io/` or submodule in existing project
- **Effort**: 12-15 hours (saves 3-5 hours per future app)

#### 14. ML-Ready Data Export
- **Description**: Export format optimized for machine learning training and analysis
- **Features**:
  - Structured output with features + labels (breath events, eupnea regions, sighs, sniffs)
  - Support for multiple formats: CSV (pandas-compatible), HDF5 (large datasets), JSON (metadata)
  - Include raw signal segments, computed features, and manual annotations
  - Batch export across multiple files/sweeps
  - Standardized schema for reproducible ML pipelines
- **Output structure**:
  - `breath_features.csv`: Per-breath metrics (Ti, Te, amplitude, frequency, etc.)
  - `annotations.csv`: User labels (eupnea, sniff bouts, quality flags, manual edits)
  - `raw_segments.h5`: Time-aligned signal windows around each breath
  - `metadata.json`: Experimental parameters, detection settings, file provenance
- **Files to modify**: `core/export.py`, add ML export dialog to `main.py`
- **Effort**: 4-5 hours

#### 15. Machine Learning Integration
- **Description**: Train models on exported labeled data to improve automated detection
- **Prerequisites**: ML-Ready Data Export (item #14)
- **Implementation Strategy**: Phased approach with increasing automation

  **Phase 1: ML-Assisted Flagging (Non-invasive)**
  - ML runs after traditional peak detection
  - Flags potentially problematic breaths for user review
  - Visual markers on plot: orange (sniff), red (artifact), yellow (uncertain)
  - Click flag to accept/reject/manually edit
  - User always has final control

  **Phase 2: ML-Enhanced Refinement (Optional)**
  - Checkbox: "Use ML to refine peak positions"
  - Shows both original (faded) and ML-adjusted (highlighted) detections
  - Only high-confidence adjustments (>80%) suggested
  - User reviews all changes before accepting

  **Phase 3: Active Learning Loop (Continuous Improvement)**
  - Manual corrections automatically added to training dataset
  - "Export Training Data" includes all user edits
  - Model improves over time, adapts to specific recording conditions
  - Model versioning and performance tracking

- **UI Components**:
  - ML Control Panel with flagging options and confidence threshold slider
  - Model selector dropdown (switch between model versions)
  - Review Flags dialog showing all flagged breaths with confidence scores
  - Export Training Data button for contributing to model improvement

- **Features**:
  - Train classifiers for: breath quality, eupnea regions, sniff detection, sigh detection, artifact removal
  - Confidence scores and uncertainty visualization
  - Fallback to traditional detection for low-confidence predictions
  - Multiple detection modes: Traditional / ML-Assisted / Hybrid (user selectable)

- **Technologies**: scikit-learn (initial), TensorFlow/PyTorch (optional for advanced models)
- **Files to create**: `core/ml_models.py`, `core/ml_training.py`, `MLControlDialog` in `main.py`
- **Effort**:
  - Phase 1 (Flagging): 8-10 hours
  - Phase 2 (Refinement): 6-8 hours
  - Phase 3 (Active Learning): 4-6 hours
  - Total: 18-24 hours

#### 16. Core Modularization (Breathtools Package)
- **Description**: Refactor core analysis functions into standalone, reusable library
- **Features**:
  - Pip-installable package independent of GUI
  - Generic data loaders (CSV, HDF5, not just ABF)
  - Integration with fiber photometry and electrophysiology pipelines
  - Standalone examples and API documentation
- **Implementation**:
  - Create `core/__init__.py` with public API
  - Extract pure data models from `AppState`
  - Add `pyproject.toml` for pip installation
  - Create example scripts (`examples/analyze_breath_simple.py`)
- **Files to modify**: Entire `core/` restructure
- **Effort**: 8-10 hours

#### 17. PyPI Publication
- **Description**: Publish app to Python Package Index for public `pip install`
- **Prerequisites**: Choose professional package name, add licensing
- **Steps**:
  - Check name availability on PyPI
  - Create proper package structure (`pyproject.toml`, entry points)
  - Test with `pip install -e .`
  - Build and upload to PyPI with `twine`
- **Alternative**: Install from GitHub (`pip install git+https://...`)
- **Effort**: 4-6 hours (first-time setup)

### Potential Future Directions (Speculative)
- Code signing for professional Windows distribution
- Automated testing framework with edge case coverage
- Additional file format support (EDF, WFDB beyond ABF)
- Real-time data acquisition capabilities
- Advanced statistical analysis modules (wavelet coherence, time-frequency)
- Plugin architecture for custom user algorithms
- Cloud-based batch processing and collaboration features

## Contributing
When modifying the codebase:
1. Test changes using `run_debug.py` first
2. Follow existing code conventions (see main.py imports and core module structure)
3. Update this CLAUDE.md file for significant architectural changes
4. Rebuild executable and test on clean machine before distribution
5. Update version info in `version_info.py` for releases

## Contact & Support
For technical issues or enhancement requests, refer to the project documentation or create an issue in the repository.