---
title: 'PhysioMetrics: Advanced Respiratory Signal Analysis Software'
tags:
  - Python
  - neuroscience
  - respiratory analysis
  - breathing patterns
  - signal processing
  - machine learning
  - plethysmography
authors:
  - name: Ryan Sean Phillips
    orcid: 0000-0002-8570-2348
    affiliation: 1
affiliations:
  - name: Seattle Children's Research Institute, Norcliffe Foundation Center for Integrative Brain Research, Seattle, WA, USA
    index: 1
date: 10 November 2024
bibliography: paper.bib
---

# Summary

PhysioMetrics is a desktop application for respiratory signal analysis in neuroscience research. It provides automated breath detection using adaptive thresholding algorithms, machine learning classification of breathing patterns, and interactive tools for manual curation. The software addresses the need for robust, user-friendly tools that can handle diverse recording conditions while enabling reproducible analysis of breathing patterns in rodent models. PhysioMetrics supports common neurophysiology file formats (Axon ABF, Spike2 SMRX, EDF) and exports comprehensive datasets suitable for statistical analysis and visualization.

# Statement of Need

Respiratory analysis is critical for understanding brain-body interactions, stress responses, autonomic dysfunction, and drug effects in neuroscience research [@Ramirez2013; @Feldman2016]. Existing approaches present significant limitations: commercial solutions (Spike2, pCLAMP) require manual threshold adjustment for each recording and lack modern machine learning capabilities, while open-source alternatives focus primarily on human sleep studies rather than rodent plethysmography with its unique signal characteristics (high frequency variability, frequent sniffing bouts, variable signal quality).

PhysioMetrics fills this gap by combining: (1) adaptive algorithms that automatically adjust to variable signal quality without manual intervention, (2) machine learning-based classification of eupnea, sniffing, and sighs, (3) interactive editing tools for expert curation and quality control, and (4) native support for common neurophysiology file formats. The software enables researchers to process large datasets reproducibly while maintaining the ability to manually curate results when needed.

# Key Features

## Automated Breath Detection

PhysioMetrics implements a multi-level fallback algorithm for robust peak detection across diverse signal qualities. The pipeline includes:

- **Adaptive Thresholding**: Otsu's method automatically determines optimal height thresholds from peak amplitude distributions, with local minimum refinement to separate noise from signal peaks [@Otsu1979]
- **Multi-level Fallback**: If primary detection fails, the system automatically attempts alternative methods with relaxed parameters
- **Breath Event Detection**: Automatic identification of inspiratory peaks, expiratory minima, breath onsets, and offsets using derivative-based methods

## Machine Learning Classification

The software includes ensemble machine learning models for automated breath pattern classification:

- **Random Forest and XGBoost Classifiers**: Train models on user-annotated breaths to distinguish eupnea (normal breathing), sniffing, and sighs based on 45 extracted features
- **GMM Clustering**: Gaussian Mixture Models provide unsupervised classification when labeled training data is unavailable [@Reynolds2009]
- **Active Learning Support**: Interactive labeling tools enable iterative model refinement with minimal manual annotation

Preliminary validation on rodent plethysmography data (n=500 breaths from 10 animals) demonstrates >90% classification accuracy for distinguishing eupnea from sniffing patterns.

## Interactive Editing and Visualization

PhysioMetrics provides a comprehensive graphical interface for data exploration and manual curation:

- **Manual Peak Editing**: Add, delete, or move detected peaks with keyboard shortcuts and click-based interaction
- **Region Marking**: Mark sniffing bouts, artifact regions, or data segments for exclusion from analysis
- **Real-time Visual Feedback**: Automatic overlays show eupnea (green), apnea (red), and sniffing (purple) regions
- **Spectral Analysis**: Power spectrum and wavelet scalogram visualization for frequency-domain analysis

## Signal Processing

- **Butterworth Filtering**: Configurable high-pass (baseline drift removal), low-pass (noise reduction), and notch filters (line noise removal)
- **Multi-Format Support**: Native loaders for Axon ABF (pCLAMP), Spike2 SMRX (64-bit), and EDF/EDF+ files
- **Multi-File Concatenation**: Combine multiple recordings from repeated experiments into single analysis sessions

## Comprehensive Metrics and Export

PhysioMetrics computes 35+ breathing metrics per breath, including:

- **Timing Metrics**: Instantaneous frequency, inspiratory time (Ti), expiratory time (Te), interbreath interval
- **Amplitude Metrics**: Inspiratory amplitude, expiratory amplitude, peak-to-trough distance
- **Pattern Metrics**: Breathing regularity score (RMSSD-based), ventilation proxy (frequency × amplitude)
- **Waveform Metrics**: Maximum inspiratory/expiratory derivatives, area under curve, breath shape metrics

Export options include:
- **CSV Files**: Per-breath metrics, time-aligned metric traces, event intervals
- **NPZ Bundles**: Compressed binary data for Python-based downstream analysis
- **PDF Summaries**: Multi-page visualization reports with statistical summaries
- **Session Files**: Complete analysis state for reproducible workflows

# Implementation

PhysioMetrics is implemented in Python 3.11 with a PyQt6 graphical user interface. Core signal processing leverages NumPy [@Harris2020] and SciPy [@Virtanen2020] for filtering and peak detection. Machine learning classification uses scikit-learn [@Pedregosa2011] for Random Forest models and XGBoost [@Chen2016] for gradient boosting. Data visualization is provided by matplotlib [@Hunter2007] with interactive navigation and real-time plot updates.

The software follows a modular architecture with clear separation between signal processing (`core/`), machine learning (`core/ml/`), file I/O (`core/io/`), and user interface (`dialogs/`, `editing/`, `ui/`). This design enables extension to other physiological signals beyond respiratory analysis. A headless Python API (in development) will enable scripted batch processing without the graphical interface.

Example usage (GUI):
```python
# Load recording
File → Open → Select .abf, .smrx, or .edf file

# Configure detection
Select respiratory channel → Adjust filters if needed → Click "Apply"

# Review and edit
Use manual editing modes to refine detection → Label breaths if training ML

# Export results
File → Save Data → Select output formats
```

Planned headless API:
```python
from physiometrics import BreathAnalyzer

# Load and process
analyzer = BreathAnalyzer('recording.abf', channel='Resp')
analyzer.apply_filters(highpass=0.5, lowpass=50)
analyzer.detect_peaks(auto_threshold=True)

# Classify patterns
analyzer.classify_breaths(method='random_forest')

# Export
analyzer.export_csv('results.csv')
```

# Validation and Use Cases

PhysioMetrics is currently in use for:
- Optogenetic stimulation studies (identifying breathing responses to neural circuit manipulation)
- Pharmacological experiments (assessing drug effects on breathing patterns)
- Developmental studies (characterizing breathing maturation in neonatal rodents)
- Stress response experiments (analyzing breathing regularity changes)

The software handles recordings ranging from 30 seconds to several hours in duration and supports multi-sweep experiments with 10+ repeated trials.

# Acknowledgments

This work was supported by the National Institute on Drug Abuse (NIDA) K01 Award K01DA058543. The author thanks the neuroscience community for feedback during development and the open-source scientific Python ecosystem that made this work possible.

# References
