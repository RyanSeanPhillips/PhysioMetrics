# PlethApp - Breath Analysis Application

## Overview
PlethApp is a PyQt6-based desktop application for advanced respiratory signal analysis. It provides comprehensive tools for breath pattern detection, eupnea/apnea identification, and breathing regularity assessment.

**Current Status**: Active development toward v1.0 publication (JOSS - Journal of Open Source Software)
- **Version**: Pre-release (v0.9.x)
- **Publication Goal**: Methods paper with ML breath classification (see PUBLICATION_ROADMAP.md)
- **License**: MIT (planned) for maximum adoption and citation
- **Long-term Vision**: Foundation for multi-app neurophysiology ecosystem (see MULTI_APP_STRATEGY.md)

---

## Quick Start

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

See **BUILD_INSTRUCTIONS.md** for detailed build documentation.

---

## Project Structure
```
plethapp/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── pleth_app.spec                  # PyInstaller configuration
├── build_executable.bat            # Windows build script
├── build_executable.py             # Cross-platform build script
├── version_info.py                 # Version metadata generator
├── run_debug.py                    # Debug launcher for testing
│
├── 📚 Documentation/
│   ├── CLAUDE.md                   # This file - Project overview
│   ├── ALGORITHMS.md               # Core algorithms and signal processing
│   ├── FILE_FORMATS.md             # Supported file formats (ABF, SMRX, EDF)
│   ├── ARCHITECTURE.md             # Software architecture and design patterns
│   ├── FEATURE_BACKLOG.md          # Planned features and roadmap
│   ├── RECENT_FEATURES.md          # Recently implemented features
│   ├── BUILD_INSTRUCTIONS.md       # Detailed build process
│   ├── PROFILING_GUIDE.md          # Performance profiling with line_profiler
│   ├── SESSION_SUMMARY.md          # Current development session notes
│   ├── PUBLICATION_ROADMAP.md      # v1.0 publication plan (12-week timeline)
│   ├── PERFORMANCE_OPTIMIZATION_PLAN.md  # Performance improvement strategy
│   └── MULTI_APP_STRATEGY.md       # Long-term multi-app ecosystem vision
│
├── core/                           # Core application modules (90% modular)
│   ├── state.py                   # Application state management
│   ├── abf_io.py                  # File I/O dispatcher (ABF, SMRX, EDF)
│   ├── filters.py                 # Signal filtering functions
│   ├── plotting.py                # Matplotlib integration
│   ├── navigation.py              # Data navigation utilities
│   ├── stim.py                    # Stimulus detection algorithms
│   ├── peaks.py                   # Peak detection and breath events
│   ├── metrics.py                 # Breathing metrics and pattern analysis
│   ├── robust_metrics.py          # Enhanced error-resistant metrics
│   ├── editing.py                 # Manual peak editing tools
│   ├── export.py                  # Data export functionality
│   └── io/                        # File format loaders
│       ├── son64_dll_loader.py   # CED SON64 DLL wrapper (low-level)
│       ├── son64_loader.py       # SMRX loader (high-level)
│       ├── s2rx_parser.py        # Spike2 .s2rx XML configuration parser
│       └── edf_loader.py         # EDF/EDF+ loader (pyedflib)
│
├── dialogs/                        # PyQt6 dialog windows
│   ├── gmm_clustering_dialog.py   # GMM eupnea/sniffing classification
│   └── event_detection_dialog.py  # Peak detection configuration
│
├── editing/                        # Editing mode implementations
│   └── editing_modes.py           # Manual peak editing (add/delete/move)
│
├── export/                        # Export functionality
│   └── export_manager.py          # Data export and summary generation
│
├── ui/                            # PyQt6 UI definition files
│   └── pleth_app_layout_02.ui    # Main application UI layout
│
├── images/                        # Application icons and assets
├── assets/                        # Additional application assets
└── examples/                      # Sample data files
```

---

## Key Features

### Data Analysis
- **Advanced Peak Detection**: Multi-level fallback algorithms for robust breath detection
- **Breath Event Detection**: Onsets, offsets, inspiratory peaks, expiratory minima
- **Eupnea Detection**: Identifies regions of normal, regular breathing patterns
- **Apnea Detection**: Detects breathing gaps longer than configurable thresholds
- **Breathing Regularity Score**: RMSSD-based assessment of breathing pattern variability
- **GMM Clustering**: Automatic eupnea/sniffing classification using Gaussian Mixture Models

### Signal Processing
- **Butterworth Filtering**: Adjustable high-pass, low-pass, and filter order (2-10)
- **Notch Filtering**: Band-stop filter for removing specific frequency ranges
- **Spectral Analysis**: Power spectrum and wavelet scalogram visualization
- **Mean Subtraction**: Baseline drift removal

### User Interface
- **Real-time Visual Overlays**: Automatic green/red line overlays for eupnea and apnea regions
- **Interactive Data Navigation**: Window-based and sweep-based data exploration
- **Manual Peak Editing**: Add/delete peaks with keyboard shortcuts (Shift/Ctrl modifiers)
- **Dark Theme**: Custom CSS styling optimized for scientific data visualization
- **Status Bar with History**: Timing messages and message history dropdown

### Data Import/Export
- **Multi-format Support**: ABF, Spike2 SMRX (.smrx), and EDF (.edf) file formats
- **CSV Export**: Analyzed breathing metrics and consolidated time-course data
- **Spectral Analysis Export**: Power spectra and wavelet data

**See ALGORITHMS.md for algorithmic details.**
**See FILE_FORMATS.md for file format documentation.**
**See RECENT_FEATURES.md for recently added features.**

---

## Documentation Map

### Core Documentation
- **CLAUDE.md** (this file): Quick start and project overview
- **ALGORITHMS.md**: Core algorithms, signal processing, and robustness features
- **FILE_FORMATS.md**: Supported file formats (ABF, SMRX, EDF) with troubleshooting
- **ARCHITECTURE.md**: Software architecture, UI design, performance optimizations
- **BUILD_INSTRUCTIONS.md**: Detailed PyInstaller build process and distribution
- **PROFILING_GUIDE.md**: Performance profiling with line_profiler

### Feature Documentation
- **RECENT_FEATURES.md**: Recently implemented features with removal instructions
- **FEATURE_BACKLOG.md**: Planned features and development roadmap
- **SESSION_SUMMARY.md**: Current development session notes and recent changes

### Strategic Planning
- **PUBLICATION_ROADMAP.md**: v1.0 publication timeline (12-week plan to JOSS submission)
  - ML breath classifier implementation
  - Headless API extraction
  - Usage tracking and telemetry

- **PERFORMANCE_OPTIMIZATION_PLAN.md**: Performance improvement strategy
  - 3 implementation phases (Quick wins → Vectorization → Advanced)
  - Expected 2-5× speedup for exports
  - Eliminate editing lag with large files

- **MULTI_APP_STRATEGY.md**: Long-term multi-app ecosystem vision
  - Extract reusable packages (neurodata-io, neuroviz, neurocore)
  - Enable rapid development of photometry and Neuropixels apps
  - 60-70% code reuse across applications

### Decision Tree: Which Document to Read?

| Question | Read |
|----------|------|
| How do I build the application? | **BUILD_INSTRUCTIONS.md** |
| How does peak detection work? | **ALGORITHMS.md** |
| How do I add support for a new file format? | **FILE_FORMATS.md** |
| What's the UI architecture? | **ARCHITECTURE.md** |
| **What should I work on next? (Todo list)** | **FEATURE_BACKLOG.md** ← Checklist at top! |
| What features are planned? | **FEATURE_BACKLOG.md** |
| What was recently added? | **RECENT_FEATURES.md** |
| When will v1.0 be released? | **PUBLICATION_ROADMAP.md** |
| Why is the app slow? | **PERFORMANCE_OPTIMIZATION_PLAN.md** |
| How do I profile performance? | **PROFILING_GUIDE.md** |
| How can I reuse PlethApp code for photometry? | **MULTI_APP_STRATEGY.md** |
| What was worked on recently? | **SESSION_SUMMARY.md** |

---

## Development Status & Timeline

### Current Priorities (2025 Q1-Q2)

**Phase 1: Performance Optimization** (see PERFORMANCE_OPTIMIZATION_PLAN.md)
- Cache eupnea masks (4× speedup for exports)
- Optional auto-GMM checkbox (eliminate editing lag)
- Vectorize export loops (3-5× speedup)

**Phase 2: ML Implementation** (see PUBLICATION_ROADMAP.md)
- Random Forest + XGBoost breath classifier (40% faster than GMM)
- Active learning integration
- ML-ready data export format

**Phase 3: v1.0 Publication** (JOSS submission, ~12 weeks)
- Complete documentation and examples
- Zenodo DOI and citation metadata
- Anonymous opt-in telemetry for usage statistics

### Version History

**Current Version**: v0.9.x (pre-release, active development)

### Planned Milestones
- **v0.9.5** (2025 Q1): Performance optimizations complete
- **v1.0.0** (2025 Q2): First public release with ML breath classifier (JOSS submission)
- **v1.1.0** (2025 Q3): Usage statistics and active learning integration
- **v2.0.0** (2025 Q4+): Multi-app infrastructure extraction

### Recent Major Changes
- ✅ Multi-file ABF concatenation (2025-10)
- ✅ Spike2 .smrx file support via CED SON64 library (2025-10)
- ✅ EDF/EDF+ file support (2025-10)
- ✅ Enhanced modular architecture with separate managers (2025-10)
- ✅ GMM clustering dialog improvements (2025-10)
- ✅ Status bar enhancements with timing and message history (2025-10)
- ✅ Performance optimization (lightweight GMM refresh) (2025-10)

See **SESSION_SUMMARY.md** for detailed recent development notes.

---

## Testing

### Manual Testing Checklist
- Test the source application using `python run_debug.py`
- Test the built executable on a clean machine without Python installed
- Verify all features work:
  - File loading (ABF, SMRX, EDF)
  - Peak detection and breath feature detection
  - GMM clustering
  - Manual peak editing
  - Filtering and spectral analysis
  - Data export (CSV, summary)

### Automated Testing
Currently no formal testing framework. Consider adding:
```bash
# Future recommendations:
# pip install pytest pytest-qt
# pytest tests/
```

---

## Contributing

When modifying the codebase:

1. **Test changes** using `run_debug.py` first
2. **Follow existing code conventions** (see main.py imports and core module structure)
3. **Update documentation** for significant changes:
   - **ALGORITHMS.md** for algorithm changes
   - **FILE_FORMATS.md** for new file format support
   - **ARCHITECTURE.md** for architectural changes
   - **RECENT_FEATURES.md** for new features
   - **SESSION_SUMMARY.md** for implementation notes
4. **Rebuild executable** and test on clean machine before distribution
5. **Update version info** in `version_info.py` for releases

---

## Architecture Philosophy

PlethApp is designed with **modularity by default**:
- Core modules (`core/`) are domain-agnostic signal processing tools
- File loaders (`core/io/`) use standardized output format
- UI components don't depend on breath-specific logic
- State management (`core/state.py`) separates data from processing

**This 90% modular architecture makes PlethApp ready for**:
1. Integration of ML classifiers (no major refactoring needed)
2. Extraction into standalone packages (already organized by function)
3. Reuse in photometry/ephys applications (generic interfaces)

**See ARCHITECTURE.md for detailed architectural documentation.**
**See MULTI_APP_STRATEGY.md for long-term multi-app ecosystem vision.**

---

## Contact & Support

For technical issues or enhancement requests, refer to the project documentation or create an issue in the repository.

---

## Additional Resources

### External Dependencies Documentation
- **PyQt6**: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- **matplotlib**: https://matplotlib.org/stable/contents.html
- **scipy**: https://docs.scipy.org/doc/scipy/
- **numpy**: https://numpy.org/doc/stable/
- **pyabf**: https://pyabf.readthedocs.io/
- **pyedflib**: https://pyedflib.readthedocs.io/

### Related Projects
- **CED Spike2**: https://ced.co.uk/products/spike2
- **Axon pCLAMP**: https://www.moleculardevices.com/products/axon-patch-clamp-system/acquisition-and-analysis-software/pclamp-software-suite

---

**For detailed information on any topic, consult the appropriate documentation file listed in the Documentation Map above.**
