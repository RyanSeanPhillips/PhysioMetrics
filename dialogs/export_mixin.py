"""
Export Mixin for Dialogs

Provides right-click context menu with export options (PNG, PDF, clipboard)
for any QDialog that inherits from this mixin.

Usage:
    from dialogs.export_mixin import ExportMixin

    class MyDialog(ExportMixin, QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setup_export_menu()  # Call after UI setup
"""

from PyQt6.QtWidgets import QMenu, QFileDialog, QMessageBox, QApplication
from PyQt6.QtGui import QAction, QPainter
from PyQt6.QtCore import Qt
from PyQt6.QtPrintSupport import QPrinter


class ExportMixin:
    """Mixin class that adds right-click export functionality to dialogs."""

    def setup_export_menu(self, default_filename: str = None):
        """
        Set up right-click context menu for export options.
        Call this at the end of your dialog's __init__ method.

        Args:
            default_filename: Base filename for exports (without extension).
                            If None, uses the dialog's window title.
        """
        self._export_default_filename = default_filename
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_export_context_menu)

    def _get_export_filename(self):
        """Get the default filename for exports."""
        if hasattr(self, '_export_default_filename') and self._export_default_filename:
            return self._export_default_filename
        # Fall back to window title, sanitized for filenames
        title = self.windowTitle() or "Export"
        return title.replace(" ", "_").replace("/", "-").replace(":", "-").replace("*", "")

    def _show_export_context_menu(self, pos):
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
        png_action.triggered.connect(lambda: self._export_dialog_as_image('png'))
        export_menu.addAction(png_action)

        pdf_action = QAction("Export as PDF (Vector)", self)
        pdf_action.triggered.connect(self._export_dialog_as_pdf)
        export_menu.addAction(pdf_action)

        jpg_action = QAction("Export as JPEG", self)
        jpg_action.triggered.connect(lambda: self._export_dialog_as_image('jpg'))
        export_menu.addAction(jpg_action)

        export_menu.addSeparator()

        clipboard_action = QAction("Copy to Clipboard", self)
        clipboard_action.triggered.connect(self._copy_dialog_to_clipboard)
        export_menu.addAction(clipboard_action)

        # Check if dialog has a matplotlib figure for plot-only export
        if hasattr(self, 'figure') and hasattr(self.figure, 'axes') and self.figure.axes:
            export_menu.addSeparator()
            plot_menu = menu.addMenu("Export Plot Only...")

            plot_png = QAction("Plot as PNG", self)
            plot_png.triggered.connect(lambda: self._export_matplotlib_figure('png'))
            plot_menu.addAction(plot_png)

            plot_svg = QAction("Plot as SVG (Vector)", self)
            plot_svg.triggered.connect(lambda: self._export_matplotlib_figure('svg'))
            plot_menu.addAction(plot_svg)

            plot_pdf = QAction("Plot as PDF (Vector)", self)
            plot_pdf.triggered.connect(lambda: self._export_matplotlib_figure('pdf'))
            plot_menu.addAction(plot_pdf)

        menu.exec(self.mapToGlobal(pos))

    def _export_dialog_as_image(self, format: str):
        """Export entire dialog as image."""
        default_name = f"{self._get_export_filename()}.{format}"

        format_filters = {
            'png': "PNG Image (*.png)",
            'jpg': "JPEG Image (*.jpg)",
            'bmp': "Bitmap Image (*.bmp)"
        }

        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export as {format.upper()}", default_name,
            format_filters.get(format, "All Files (*)")
        )

        if not filepath:
            return

        if not filepath.lower().endswith(f'.{format}'):
            filepath += f'.{format}'

        try:
            # Capture at 2x resolution for publication quality
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
                    f"Image exported to:\n{filepath}\n\n"
                    f"Resolution: {pixmap.width()} x {pixmap.height()} pixels")
                print(f"[export] Saved {format.upper()}: {filepath}")
            else:
                raise Exception("Failed to save image")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export:\n{str(e)}")
            print(f"[export] Failed: {e}")

    def _export_dialog_as_pdf(self):
        """Export entire dialog as PDF with vector quality."""
        default_name = f"{self._get_export_filename()}.pdf"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export as PDF", default_name,
            "PDF Document (*.pdf)"
        )

        if not filepath:
            return

        if not filepath.lower().endswith('.pdf'):
            filepath += '.pdf'

        try:
            from PyQt6.QtGui import QPageLayout

            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(filepath)
            printer.setPageOrientation(QPageLayout.Orientation.Landscape)

            painter = QPainter()
            if not painter.begin(printer):
                raise Exception("Failed to initialize PDF painter")

            # Scale to fit page with margins
            page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
            dialog_size = self.size()

            scale_x = page_rect.width() / dialog_size.width()
            scale_y = page_rect.height() / dialog_size.height()
            scale = min(scale_x, scale_y) * 0.95  # 5% margin

            # Center on page
            offset_x = (page_rect.width() - dialog_size.width() * scale) / 2
            offset_y = (page_rect.height() - dialog_size.height() * scale) / 2

            painter.translate(offset_x, offset_y)
            painter.scale(scale, scale)
            self.render(painter)
            painter.end()

            QMessageBox.information(self, "PDF Saved",
                f"PDF exported to:\n{filepath}")
            print(f"[export] Saved PDF: {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export PDF:\n{str(e)}")
            print(f"[export] PDF failed: {e}")
            import traceback
            traceback.print_exc()

    def _copy_dialog_to_clipboard(self):
        """Copy dialog screenshot to clipboard at high resolution."""
        try:
            # Capture at 2x resolution
            pixmap = self.grab()
            scaled_size = pixmap.size() * 2
            pixmap = pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            clipboard = QApplication.clipboard()
            clipboard.setPixmap(pixmap)

            QMessageBox.information(self, "Copied to Clipboard",
                f"Screenshot copied!\n\n"
                f"Resolution: {pixmap.width()} x {pixmap.height()} pixels\n\n"
                "Paste with Ctrl+V into any application.")
            print(f"[export] Copied to clipboard ({pixmap.width()}x{pixmap.height()})")

        except Exception as e:
            QMessageBox.critical(self, "Copy Failed", f"Failed to copy:\n{str(e)}")

    def _export_matplotlib_figure(self, format: str):
        """Export matplotlib figure (if present) to file."""
        if not hasattr(self, 'figure'):
            QMessageBox.warning(self, "No Plot", "No matplotlib figure to export.")
            return

        default_name = f"{self._get_export_filename()}_plot.{format}"

        format_filters = {
            'png': "PNG Image (*.png)",
            'svg': "SVG Vector (*.svg)",
            'pdf': "PDF Document (*.pdf)"
        }

        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export Plot as {format.upper()}", default_name,
            format_filters.get(format, "All Files (*)")
        )

        if not filepath:
            return

        if not filepath.lower().endswith(f'.{format}'):
            filepath += f'.{format}'

        try:
            dpi = 300 if format == 'png' else 150
            self.figure.savefig(
                filepath,
                format=format,
                dpi=dpi,
                bbox_inches='tight',
                facecolor=self.figure.get_facecolor(),
                edgecolor='none'
            )

            QMessageBox.information(self, "Plot Exported",
                f"Plot exported to:\n{filepath}")
            print(f"[export] Saved plot: {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export plot:\n{str(e)}")
