"""
Training Results Dialog

Shows ML model training results including:
- Accuracy metrics
- Feature importance plot
- Confusion matrix
- Per-class metrics
"""

import sys

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QWidget, QTextEdit, QPushButton,
    QScrollArea, QFrame, QGroupBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from core.ml_training import TrainingResult


class TrainingResultsDialog(QDialog):
    """Dialog to display training results with plots and metrics."""

    def __init__(self, result: TrainingResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle(f"Training Results - {result.model_name}")
        self.resize(1000, 700)

        self._init_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value)
                )
            except Exception:
                pass

    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # Header with key metrics
        header = self._create_header()
        layout.addWidget(header)

        # Tabbed interface
        tabs = QTabWidget()

        # Tab 1: Overview
        overview_tab = self._create_overview_tab()
        tabs.addTab(overview_tab, "Overview")

        # Tab 2: Feature Importance
        if self.result.feature_importance_plot:
            feature_tab = self._create_plot_tab(
                self.result.feature_importance_plot,
                "Top features contributing to model predictions"
            )
            tabs.addTab(feature_tab, "Feature Importance")

        # Tab 3: Confusion Matrix
        if self.result.confusion_matrix_plot:
            confusion_tab = self._create_plot_tab(
                self.result.confusion_matrix_plot,
                "Model prediction accuracy by class"
            )
            tabs.addTab(confusion_tab, "Confusion Matrix")

        # Tab 4: Detailed Metrics
        metrics_tab = self._create_metrics_tab()
        tabs.addTab(metrics_tab, "Detailed Metrics")

        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_model_btn = QPushButton("Save Model...")
        self.save_model_btn.setMinimumWidth(120)
        button_layout.addWidget(self.save_model_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.setMinimumWidth(120)
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def _create_header(self) -> QWidget:
        """Create header with key metrics."""
        header = QFrame()
        header.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        layout = QHBoxLayout(header)

        # Model name and type
        title_label = QLabel(f"<b>{self.result.model_name}</b>")
        title_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(title_label)

        layout.addStretch()

        # Key metrics
        metrics_text = f"""
        <table>
        <tr><td><b>Test Accuracy:</b></td><td style='color: #4ec9b0;'>{self.result.test_accuracy:.1%}</td></tr>
        <tr><td><b>CV Score:</b></td><td>{self.result.cv_mean:.1%} ± {self.result.cv_std:.1%}</td></tr>
        """

        if self.result.baseline_accuracy is not None:
            improvement_color = "#4ec9b0" if self.result.accuracy_improvement > 0 else "#ce9178"
            metrics_text += f"""
            <tr><td><b>Improvement:</b></td><td style='color: {improvement_color};'>+{self.result.accuracy_improvement:.1%}</td></tr>
            <tr><td><b>Error Reduction:</b></td><td style='color: {improvement_color};'>{self.result.error_reduction_pct:.1f}%</td></tr>
            """

        metrics_text += "</table>"

        metrics_label = QLabel(metrics_text)
        layout.addWidget(metrics_label)

        return header

    def _create_overview_tab(self) -> QWidget:
        """Create overview tab with summary metrics."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # Dataset info
        dataset_group = QGroupBox("Dataset Information")
        dataset_layout = QVBoxLayout()

        dataset_info = f"""
        <b>Training samples:</b> {self.result.n_train}<br>
        <b>Test samples:</b> {self.result.n_test}<br>
        <b>Number of features:</b> {self.result.n_features}<br>
        <b>Classes:</b> {', '.join(self.result.class_labels)}
        """

        dataset_label = QLabel(dataset_info)
        dataset_layout.addWidget(dataset_label)
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # Performance metrics
        performance_group = QGroupBox("Performance Metrics")
        performance_layout = QVBoxLayout()

        performance_text = f"""
        <table cellspacing='10'>
        <tr><th align='left'>Metric</th><th align='left'>Value</th></tr>
        <tr><td>Training Accuracy</td><td style='color: #4ec9b0;'><b>{self.result.train_accuracy:.1%}</b></td></tr>
        <tr><td>Test Accuracy</td><td style='color: #4ec9b0;'><b>{self.result.test_accuracy:.1%}</b></td></tr>
        <tr><td>Cross-Validation Mean</td><td>{self.result.cv_mean:.1%}</td></tr>
        <tr><td>Cross-Validation Std Dev</td><td>± {self.result.cv_std:.1%}</td></tr>
        """

        if self.result.baseline_accuracy is not None:
            performance_text += f"""
            <tr><td colspan='2'><hr></td></tr>
            <tr><td>Baseline Accuracy</td><td>{self.result.baseline_accuracy:.1%}</td></tr>
            <tr><td>Absolute Improvement</td><td style='color: #4ec9b0;'><b>+{self.result.accuracy_improvement:.1%}</b></td></tr>
            <tr><td>Error Reduction</td><td style='color: #4ec9b0;'><b>{self.result.error_reduction_pct:.1f}%</b></td></tr>
            """

        performance_text += "</table>"

        performance_label = QLabel(performance_text)
        performance_layout.addWidget(performance_label)
        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)

        # Top 10 features
        features_group = QGroupBox("Top 10 Most Important Features")
        features_layout = QVBoxLayout()

        top_features = self.result.feature_importance.head(10)
        features_text = "<table cellspacing='5'><tr><th align='left'>Rank</th><th align='left'>Feature</th><th align='left'>Importance</th></tr>"

        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            features_text += f"<tr><td>{idx}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"

        features_text += "</table>"

        features_label = QLabel(features_text)
        features_label.setWordWrap(True)
        features_layout.addWidget(features_label)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        layout.addStretch()

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)

        return wrapper

    def _create_plot_tab(self, plot_bytes: bytes, description: str) -> QWidget:
        """Create a tab displaying a plot."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-style: italic; padding: 10px;")
        layout.addWidget(desc_label)

        # Plot
        pixmap = QPixmap()
        pixmap.loadFromData(plot_bytes)

        plot_label = QLabel()
        plot_label.setPixmap(pixmap)
        plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidget(plot_label)
        scroll.setWidgetResizable(False)  # Keep original size
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(scroll)

        return widget

    def _create_metrics_tab(self) -> QWidget:
        """Create detailed metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # Per-class metrics
        metrics_group = QGroupBox("Per-Class Metrics")
        metrics_layout = QVBoxLayout()

        metrics_text = "<table cellspacing='10' cellpadding='5'>"
        metrics_text += "<tr style='background-color: #2d2d30;'><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>"

        for label in self.result.class_labels:
            precision = self.result.precision.get(label, 0)
            recall = self.result.recall.get(label, 0)
            f1 = self.result.f1_score.get(label, 0)

            metrics_text += f"""
            <tr>
                <td><b>{label}</b></td>
                <td>{precision:.3f}</td>
                <td>{recall:.3f}</td>
                <td>{f1:.3f}</td>
            </tr>
            """

        metrics_text += "</table>"

        metrics_label = QLabel(metrics_text)
        metrics_layout.addWidget(metrics_label)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Confusion Matrix (numeric)
        cm_group = QGroupBox("Confusion Matrix (Numeric)")
        cm_layout = QVBoxLayout()

        cm_text = "<table cellspacing='5' cellpadding='8' border='1' style='border-collapse: collapse;'>"
        cm_text += "<tr style='background-color: #2d2d30;'><th></th>"

        # Header row
        for label in self.result.class_labels:
            cm_text += f"<th>Pred: {label}</th>"
        cm_text += "</tr>"

        # Data rows
        for i, true_label in enumerate(self.result.class_labels):
            cm_text += f"<tr><td style='background-color: #2d2d30;'><b>True: {true_label}</b></td>"
            for j, pred_label in enumerate(self.result.class_labels):
                count = self.result.confusion_matrix[i, j]
                # Highlight diagonal (correct predictions)
                bg_color = "#1e4620" if i == j else ""
                cm_text += f"<td style='text-align: center; background-color: {bg_color};'>{count}</td>"
            cm_text += "</tr>"

        cm_text += "</table>"

        cm_label = QLabel(cm_text)
        cm_layout.addWidget(cm_label)
        cm_group.setLayout(cm_layout)
        layout.addWidget(cm_group)

        layout.addStretch()

        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)

        return wrapper

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }

            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background-color: #1e1e1e;
            }

            QTabBar::tab {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 8px 16px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background-color: #094771;
                color: #ffffff;
            }

            QTabBar::tab:hover {
                background-color: #3e3e42;
            }

            QPushButton {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #3e3e42;
            }

            QPushButton:pressed {
                background-color: #094771;
            }

            QLabel {
                color: #d4d4d4;
            }

            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #d4d4d4;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }

            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }

            QFrame {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 5px;
                padding: 10px;
            }
        """)
