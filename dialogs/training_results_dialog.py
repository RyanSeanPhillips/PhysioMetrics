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
    QScrollArea, QFrame, QGroupBox, QFileDialog, QMessageBox, QApplication,
    QMenu
)
from PyQt6.QtGui import QPixmap, QFont, QPainter, QAction
from PyQt6.QtCore import Qt
from PyQt6.QtPrintSupport import QPrinter
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
        self._setup_context_menu()

    def _setup_context_menu(self):
        """Set up right-click context menu for export options."""
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        """Show context menu with export options."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d30;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QMenu::separator {
                height: 1px;
                background-color: #3e3e42;
                margin: 4px 10px;
            }
        """)

        # Export submenu
        export_menu = menu.addMenu("Export...")

        png_action = QAction("Export as PNG (High Resolution)", self)
        png_action.triggered.connect(self._export_screenshot)
        export_menu.addAction(png_action)

        pdf_action = QAction("Export as PDF (Vector)", self)
        pdf_action.triggered.connect(self._export_pdf)
        export_menu.addAction(pdf_action)

        jpg_action = QAction("Export as JPEG", self)
        jpg_action.triggered.connect(lambda: self._export_image('jpg'))
        export_menu.addAction(jpg_action)

        export_menu.addSeparator()

        clipboard_action = QAction("Copy to Clipboard", self)
        clipboard_action.triggered.connect(self._copy_to_clipboard)
        export_menu.addAction(clipboard_action)

        menu.exec(self.mapToGlobal(pos))

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

        # Store tabs reference for screenshot
        self.tabs = tabs

        # Buttons
        button_layout = QHBoxLayout()

        # Export buttons on the left
        export_label = QLabel("Export:")
        export_label.setStyleSheet("color: #888;")
        button_layout.addWidget(export_label)

        # PNG export button
        self.export_png_btn = QPushButton("PNG")
        self.export_png_btn.setToolTip("Export as high-resolution PNG image")
        self.export_png_btn.setFixedWidth(60)
        self.export_png_btn.clicked.connect(self._export_screenshot)
        button_layout.addWidget(self.export_png_btn)

        # PDF export button
        self.export_pdf_btn = QPushButton("PDF")
        self.export_pdf_btn.setToolTip("Export as PDF document (vector quality)")
        self.export_pdf_btn.setFixedWidth(60)
        self.export_pdf_btn.clicked.connect(self._export_pdf)
        button_layout.addWidget(self.export_pdf_btn)

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

    def _export_screenshot(self):
        """Export the dialog as a high-resolution screenshot."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
        from PyQt6.QtCore import QTimer

        # Default filename based on model name
        safe_model_name = self.result.model_name.replace(" ", "_").replace("/", "-")
        default_name = f"ML_Training_{safe_model_name}.png"

        # Show save dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Screenshot",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )

        if not filepath:
            return  # User cancelled

        # Ensure correct extension
        if not (filepath.lower().endswith('.png') or filepath.lower().endswith('.jpg')):
            filepath += '.png'

        try:
            # Get device pixel ratio for high-DPI displays
            screen = QApplication.primaryScreen()
            dpr = screen.devicePixelRatio() if screen else 1.0

            # Capture at higher resolution (2x for publication quality)
            scale_factor = max(2.0, dpr)

            # Grab the dialog content
            pixmap = self.grab()

            # Scale up for higher resolution if needed
            if scale_factor > 1.0:
                from PyQt6.QtCore import Qt as QtCore
                scaled_size = pixmap.size() * scale_factor
                pixmap = pixmap.scaled(
                    scaled_size,
                    QtCore.AspectRatioMode.KeepAspectRatio,
                    QtCore.TransformationMode.SmoothTransformation
                )

            # Save
            quality = 95 if filepath.lower().endswith('.jpg') else -1
            success = pixmap.save(filepath, quality=quality)

            if success:
                QMessageBox.information(
                    self,
                    "Screenshot Saved",
                    f"Screenshot exported to:\n{filepath}\n\n"
                    f"Resolution: {pixmap.width()} x {pixmap.height()} pixels"
                )
                print(f"[ml-training] Screenshot saved: {filepath} ({pixmap.width()}x{pixmap.height()})")
            else:
                raise Exception("Failed to save image")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export screenshot:\n{str(e)}"
            )
            print(f"[ml-training] Screenshot export failed: {e}")

    def _export_pdf(self):
        """Export the dialog as a PDF document with vector quality."""
        # Default filename based on model name
        safe_model_name = self.result.model_name.replace(" ", "_").replace("/", "-")
        default_name = f"ML_Training_{safe_model_name}.pdf"

        # Show save dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export as PDF",
            default_name,
            "PDF Document (*.pdf);;All Files (*)"
        )

        if not filepath:
            return  # User cancelled

        # Ensure correct extension
        if not filepath.lower().endswith('.pdf'):
            filepath += '.pdf'

        try:
            # Set up printer for PDF output
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(filepath)

            # Set page size to match dialog aspect ratio
            from PyQt6.QtCore import QMarginsF
            from PyQt6.QtGui import QPageSize, QPageLayout

            # Use A4 landscape for better fit
            printer.setPageOrientation(QPageLayout.Orientation.Landscape)

            # Create painter and render
            painter = QPainter()
            if not painter.begin(printer):
                raise Exception("Failed to initialize PDF painter")

            # Calculate scaling to fit dialog on page
            page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
            dialog_size = self.size()

            scale_x = page_rect.width() / dialog_size.width()
            scale_y = page_rect.height() / dialog_size.height()
            scale = min(scale_x, scale_y) * 0.95  # 95% to add margins

            # Center on page
            offset_x = (page_rect.width() - dialog_size.width() * scale) / 2
            offset_y = (page_rect.height() - dialog_size.height() * scale) / 2

            painter.translate(offset_x, offset_y)
            painter.scale(scale, scale)

            # Render the dialog
            self.render(painter)

            painter.end()

            QMessageBox.information(
                self,
                "PDF Saved",
                f"PDF exported to:\n{filepath}"
            )
            print(f"[ml-training] PDF saved: {filepath}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export PDF:\n{str(e)}"
            )
            print(f"[ml-training] PDF export failed: {e}")
            import traceback
            traceback.print_exc()

    def _export_image(self, format: str):
        """Export dialog as image in specified format."""
        safe_model_name = self.result.model_name.replace(" ", "_").replace("/", "-")
        default_name = f"ML_Training_{safe_model_name}.{format}"

        format_filters = {
            'png': "PNG Image (*.png)",
            'jpg': "JPEG Image (*.jpg)",
            'bmp': "Bitmap Image (*.bmp)"
        }

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Export as {format.upper()}",
            default_name,
            format_filters.get(format, "All Files (*)")
        )

        if not filepath:
            return

        if not filepath.lower().endswith(f'.{format}'):
            filepath += f'.{format}'

        try:
            # Capture at 2x resolution
            pixmap = self.grab()
            scaled_size = pixmap.size() * 2
            pixmap = pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            quality = 95 if format == 'jpg' else -1
            if pixmap.save(filepath, quality=quality):
                QMessageBox.information(self, "Export Successful",
                    f"Image exported to:\n{filepath}\n\nResolution: {pixmap.width()} x {pixmap.height()}")
            else:
                raise Exception("Failed to save image")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export image:\n{str(e)}")

    def _copy_to_clipboard(self):
        """Copy dialog screenshot to clipboard."""
        try:
            # Capture at 2x resolution for better quality
            pixmap = self.grab()
            scaled_size = pixmap.size() * 2
            pixmap = pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            clipboard = QApplication.clipboard()
            clipboard.setPixmap(pixmap)

            QMessageBox.information(self, "Copied",
                f"Screenshot copied to clipboard!\n\nResolution: {pixmap.width()} x {pixmap.height()} pixels\n\n"
                "You can now paste (Ctrl+V) into any application.")
            print(f"[ml-training] Copied to clipboard ({pixmap.width()}x{pixmap.height()})")

        except Exception as e:
            QMessageBox.critical(self, "Copy Failed", f"Failed to copy to clipboard:\n{str(e)}")
