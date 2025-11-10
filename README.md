# PhysioMetrics - Respiratory Signal Analysis

**PhysioMetrics** is a desktop application for advanced respiratory signal analysis, providing comprehensive tools for breath pattern detection, eupnea/apnea identification, and breathing regularity assessment.

![Version](https://img.shields.io/badge/version-1.0.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Author & Funding

**Developed by:** Ryan Sean Phillips
**Institution:** Seattle Children's Research Institute, Norcliffe Foundation Center for Integrative Brain Research
**Contact:** ryan.phillips@seattlechildrens.org
**ORCID:** [0000-0002-8570-2348](https://orcid.org/0000-0002-8570-2348)

**Funding:** This work was supported by the National Institute on Drug Abuse (NIDA) K01 Award K01DA058543.

PhysioMetrics was developed as part of independent research funded by an NIH K01 Career Development Award to support respiratory signal analysis and breathing pattern characterization.

## Features

- **Advanced Peak Detection**: Multi-level fallback algorithms for robust breath detection
- **Breath Event Analysis**: Automatic detection of onsets, offsets, inspiratory peaks, and expiratory minima
- **Eupnea/Apnea Detection**: Identifies regions of normal breathing and breathing gaps
- **GMM Clustering**: Automatic eupnea/sniffing classification using Gaussian Mixture Models
- **Signal Processing**: Butterworth filtering, notch filters, and spectral analysis
- **Multi-Format Support**: Load ABF (Axon) and EDF files
- **Interactive Editing**: Manual peak editing with keyboard shortcuts
- **Data Export**: Export analyzed data to CSV with comprehensive summary reports

## Download

**[Download PhysioMetrics v1.0.11 for Windows](https://github.com/RyanSeanPhillips/PhysioMetrics/releases/latest)**

Download the ZIP file, extract it, and run `PhysioMetrics_v1.0.11.exe` - no installation required!

## Requirements

### For Running the Executable
- Windows 10 or later
- No Python installation required

### For Running from Source
- Python 3.11 or later
- See `requirements.txt` for dependencies

## Building from Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/RyanSeanPhillips/PhysioMetrics.git
   cd PhysioMetrics
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Build executable** (optional)
   ```bash
   python build_executable.py
   ```

   See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed build documentation.

## Quick Start

1. Launch PhysioMetrics
2. Load a data file (ABF, SMRX, or EDF format)
3. Adjust filter settings if needed
4. Click "Auto-Detect" to identify breath peaks
5. Use manual editing tools to refine peak detection
6. Export analyzed data to CSV

## File Format Support

- **ABF (Axon Binary Format)**: Axon pCLAMP files (.abf)
- **EDF/EDF+**: European Data Format files (.edf)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use PhysioMetrics in your research, please cite:

```
Phillips, R.S. (2025). PhysioMetrics: Advanced Respiratory Signal Analysis Software (Version 1.0.11) [Software].
GitHub. https://github.com/RyanSeanPhillips/PhysioMetrics
DOI: [Zenodo DOI will be added upon first major release]
```

**BibTeX:**
```bibtex
@software{phillips2025physiometrics,
  author = {Phillips, Ryan Sean},
  title = {PhysioMetrics: Advanced Respiratory Signal Analysis Software},
  year = {2025},
  version = {1.0.11},
  url = {https://github.com/RyanSeanPhillips/PhysioMetrics},
  note = {Funded by NIDA K01 Award K01DA058543}
}
```

## Support

For issues, questions, or feature requests, please open an issue on GitHub:
https://github.com/RyanSeanPhillips/PhysioMetrics/issues

## Acknowledgments

This software was developed by Ryan Sean Phillips with support from the National Institute on Drug Abuse (NIDA) K01 Award K01DA058543.

PhysioMetrics uses the following open-source libraries:
- PyQt6 for the user interface
- NumPy and SciPy for signal processing
- Matplotlib for data visualization
- pyABF for ABF file support
- pyEDFlib for EDF file support

---

**Version**: 1.0.11
**Developer**: Ryan Sean Phillips
**Institution**: Seattle Children's Research Institute
**License**: MIT
**Funding**: NIDA K01DA058543
**Repository**: https://github.com/RyanSeanPhillips/PhysioMetrics
