# PlethApp - Breath Analysis Tool

**PlethApp** is a desktop application for advanced respiratory signal analysis, providing comprehensive tools for breath pattern detection, eupnea/apnea identification, and breathing regularity assessment.

![Version](https://img.shields.io/badge/version-1.0.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

## Features

- **Advanced Peak Detection**: Multi-level fallback algorithms for robust breath detection
- **Breath Event Analysis**: Automatic detection of onsets, offsets, inspiratory peaks, and expiratory minima
- **Eupnea/Apnea Detection**: Identifies regions of normal breathing and breathing gaps
- **GMM Clustering**: Automatic eupnea/sniffing classification using Gaussian Mixture Models
- **Signal Processing**: Butterworth filtering, notch filters, and spectral analysis
- **Multi-Format Support**: Load ABF (Axon), SMRX (Spike2), and EDF files
- **Interactive Editing**: Manual peak editing with keyboard shortcuts
- **Data Export**: Export analyzed data to CSV with comprehensive summary reports

## Download

**[Download PlethApp v1.0.11 for Windows](https://github.com/RyanSeanPhillips/plethapp_GUI/releases/latest)**

Download the ZIP file, extract it, and run `PlethApp_v1.0.11.exe` - no installation required!

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
   git clone https://github.com/RyanSeanPhillips/plethapp_GUI.git
   cd plethapp_GUI
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

1. Launch PlethApp
2. Load a data file (ABF, SMRX, or EDF format)
3. Adjust filter settings if needed
4. Click "Auto-Detect" to identify breath peaks
5. Use manual editing tools to refine peak detection
6. Export analyzed data to CSV

## File Format Support

- **ABF (Axon Binary Format)**: Axon pCLAMP files
- **SMRX (Spike2)**: CED Spike2 files
- **EDF/EDF+**: European Data Format files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use PlethApp in your research, please cite:

```
PlethApp: Advanced Breath Analysis Tool
Ryan Phillips (2024)
https://github.com/RyanSeanPhillips/plethapp_GUI
```

## Support

For issues, questions, or feature requests, please open an issue on GitHub:
https://github.com/RyanSeanPhillips/plethapp_GUI/issues

## Acknowledgments

PlethApp uses the following open-source libraries:
- PyQt6 for the user interface
- NumPy and SciPy for signal processing
- Matplotlib for data visualization
- pyABF for ABF file support
- pyEDFlib for EDF file support

---

**Version**: 1.0.11
**Author**: Ryan Phillips
**Repository**: https://github.com/RyanSeanPhillips/plethapp_GUI
