# Changelog

All notable changes to PhysioMetrics will be documented in this file.

**Developer:** Ryan Sean Phillips
**Institution:** Seattle Children's Research Institute
**Funding:** NIDA K01DA058543
**Repository:** https://github.com/RyanSeanPhillips/PhysioMetrics

---

## [1.0.13] - 2024-11-XX

### Added
- Signal quality assessment in Auto-Detect dialog
  - Exponential + Gaussian mixture model fitting to peak height histogram
  - Automatic valley detection for optimal threshold placement
  - Color-coded signal quality metric (Excellent → Very Poor)
  - Visual curve fitting display with individual components
- GitHub release update checker
  - Automatic check for new versions on GitHub
  - Non-intrusive notification with download link
  - Background thread with graceful error handling

### Fixed
- Histogram display now uses correct parameters (99th percentile, 200 bins) on first load
- Threshold line can now be dragged even when Y2 metric is displayed
- Y-axis autoscaling now works consistently using percentile-based scaling (1st-99th + 25% padding)
- Resolved Y2 axis blocking threshold line dragging by adjusting z-order

### Changed
- Default threshold changed from Otsu's method to valley threshold (local minimum)
- Improved histogram parameter calculation during peak detection

### Known Issues
- Zero crossing markers may appear slightly offset on recordings with large DC offset removed by high-pass filtering
  - Issue is cosmetic and does not affect breath detection accuracy
  - Will be addressed in future ML refactor

---

## [1.0.10] - 2024-XX-XX

### Added
- Multi-file ABF concatenation support
- Spike2 .smrx file format support via CED SON64 library
- EDF/EDF+ file format support
- GMM clustering for eupnea/sniffing classification
- Status bar with timing and message history

### Changed
- Enhanced modular architecture with separate managers
- Performance optimization for GMM refresh

---

## Release Notes Format

Each release includes:
- **Version number**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Date**: Release date
- **Changes**: Organized by category (Added, Fixed, Changed, Removed, Known Issues)
- **Author**: Ryan Sean Phillips
- **Funding**: NIDA K01DA058543

---

## How to Report Issues

Found a bug or have a feature request? Please open an issue on GitHub:
https://github.com/RyanSeanPhillips/PhysioMetrics/issues

---

**PhysioMetrics** is developed by Ryan Sean Phillips at Seattle Children's Research Institute and funded by NIDA K01DA058543.
