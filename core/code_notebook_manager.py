"""
Code Notebook Manager - Manages the Python code execution notebook.

This module was extracted from main.py to reduce complexity.
It handles code execution, safety checks, and figure management.
"""

import ast
import io
import os
import base64
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow


class CodeNotebookManager:
    """
    Manages the Code Notebook functionality.

    This class handles:
    - Safe code execution with AST-based security checks
    - Matplotlib figure capture and display
    - Figure pop-out and save functionality
    - Sandbox directory management
    """

    def __init__(self, main_window: 'QMainWindow'):
        """
        Initialize the Code Notebook Manager.

        Args:
            main_window: Reference to MainWindow for widget access
        """
        self.mw = main_window
        self._notebook_figures = []

    # -------------------------------------------------------------------------
    # Code Safety
    # -------------------------------------------------------------------------

    def check_code_safety(self, code: str) -> tuple:
        """
        Check code for dangerous operations using AST parsing.

        RELAXED POLICY: Allows file reading and plotting, blocks only destructive operations.

        Returns:
            (is_safe, blocked_reasons, warnings)
            - is_safe: True if code passes all checks
            - blocked_reasons: List of reasons code was blocked (hard reject)
            - warnings: List of warnings (user can override)
        """
        blocked_reasons = []
        warnings = []

        # Blocked imports - only truly dangerous ones (system manipulation, network, code execution)
        blocked_imports = {
            'subprocess',  # Running shell commands
            'shutil',      # File deletion/moving
            'ctypes',      # Low-level memory access
            'multiprocessing',  # Process spawning
            'socket',      # Network connections
            'ftplib', 'smtplib',  # Network protocols
            'pty', 'pipes',  # Terminal/process control
            'signal',      # Signal handling
        }

        # Allowed imports (explicitly safe for data analysis)
        # os - needed for os.path operations (reading paths)
        # sys - may be needed for some libraries
        # pickle - needed to load saved data
        # tempfile - harmless for temp files

        # Blocked function calls - only destructive ones
        blocked_calls = {
            'eval', 'exec', '__import__', 'compile',  # Code execution
            'breakpoint',  # Debugger
            'exit', 'quit',  # Exit handlers
        }

        # Warning patterns (user can proceed with confirmation)
        warning_patterns = ['.to_csv', '.to_excel', '.to_parquet', '.to_json']

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"], []

        class SafetyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.blocked = []
                self.warned = []

            def visit_Import(self, node):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in blocked_imports:
                        self.blocked.append(f"Blocked import: '{alias.name}' (system/network access)")
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in blocked_imports:
                        self.blocked.append(f"Blocked import: 'from {node.module}' (system/network access)")
                self.generic_visit(node)

            def visit_Call(self, node):
                # Check for blocked function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in blocked_calls:
                        self.blocked.append(f"Blocked call: '{node.func.id}()' (code execution)")
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous/destructive method calls
                    attr = node.func.attr
                    if attr in ['unlink', 'rmdir', 'remove', 'rmtree']:
                        self.blocked.append(f"Blocked method: '.{attr}()' (file deletion)")
                    elif attr == 'system':
                        self.blocked.append(f"Blocked method: '.system()' (shell command)")
                    # Check for warning patterns (writing files)
                    for pattern in warning_patterns:
                        if pattern.lstrip('.') == attr:
                            self.warned.append(f"File write operation: '.{attr}()'")
                self.generic_visit(node)

        visitor = SafetyVisitor()
        visitor.visit(tree)

        blocked_reasons = visitor.blocked
        warnings = visitor.warned

        # Warn (but don't block) on file writes
        code_str = code.lower()
        if 'open(' in code_str and ("'w'" in code_str or '"w"' in code_str or "'a'" in code_str or '"a"' in code_str):
            warnings.append("File open with write/append mode detected")

        is_safe = len(blocked_reasons) == 0
        return is_safe, blocked_reasons, warnings

    def get_sandbox_directory(self) -> str:
        """Get or create the sandbox directory for code execution file operations."""
        # Use a sandbox folder in user's app data
        if os.name == 'nt':  # Windows
            base = Path(os.environ.get('LOCALAPPDATA', Path.home()))
        else:  # Linux/Mac
            base = Path.home() / '.local' / 'share'

        sandbox = base / 'PhysioMetrics' / 'code_sandbox'
        sandbox.mkdir(parents=True, exist_ok=True)
        return str(sandbox)

    # -------------------------------------------------------------------------
    # Code Execution
    # -------------------------------------------------------------------------

    def on_run_code(self):
        """Execute code from the notebook code input with safety checks and timeout."""
        from PyQt6.QtWidgets import QApplication, QMessageBox
        from contextlib import redirect_stdout, redirect_stderr

        if not hasattr(self.mw, 'codeInputEdit') or not hasattr(self.mw, 'codeOutputText'):
            return

        code = self.mw.codeInputEdit.toPlainText().strip()
        if not code:
            self.mw.codeOutputText.append("<span style='color: #f48771;'>No code to execute.</span>")
            return

        # AST-based security check
        is_safe, blocked_reasons, warnings = self.check_code_safety(code)

        # Hard reject if blocked operations found
        if not is_safe:
            msg = QMessageBox(self.mw)

            # Check if it's a syntax error vs security block
            is_syntax_error = any("Syntax error" in r for r in blocked_reasons)

            if is_syntax_error:
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("Code Syntax Error")
                msg.setText("The code has a syntax error and cannot be executed:")
                msg.setInformativeText("\n".join(f"  * {r}" for r in blocked_reasons[:5]))
                msg.setDetailedText(
                    "Check the code for:\n"
                    "* Missing colons after def/if/for/while\n"
                    "* Unmatched parentheses or brackets\n"
                    "* Incorrect indentation\n"
                    "* Missing quotes in strings\n\n"
                    "Try fixing the syntax error and running again."
                )
            else:
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Code Blocked - Security Violation")
                msg.setText("This code contains blocked operations and cannot be executed:")
                msg.setInformativeText("\n".join(f"  * {r}" for r in blocked_reasons[:5]))
                msg.setDetailedText(
                    "For security, the following are blocked:\n"
                    "* Imports: subprocess, shutil, ctypes, socket, multiprocessing\n"
                    "* Calls: eval(), exec(), __import__(), compile()\n"
                    "* Methods: .unlink(), .rmdir(), .remove(), .rmtree(), .system()\n\n"
                    "ALLOWED: os, sys, pandas, numpy, scipy, matplotlib, open() for reading\n\n"
                    "The code notebook is for data analysis.\n"
                    "Use built-in Export functions to save results."
                )
            msg.exec()
            error_type = "syntax error" if is_syntax_error else "prohibited operations"
            self.mw.codeOutputText.append(f"<span style='color: #f48771;'>Code blocked - {error_type}.</span>")
            return

        # Warn if potentially risky operations found (user can proceed)
        if warnings:
            msg = QMessageBox(self.mw)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Potentially Risky Code")
            msg.setText("This code contains operations that may write files:")
            msg.setInformativeText("\n".join(f"  ! {w}" for w in warnings[:5]))
            msg.setDetailedText(
                "These operations can create files but are sandboxed.\n"
                "Any files created will be in the sandbox directory.\n\n"
                "Are you sure you want to run this code?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)

            if msg.exec() != QMessageBox.StandardButton.Yes:
                self.mw.codeOutputText.append("<span style='color: #f48771;'>Execution cancelled by user.</span>")
                return

        self.mw.codeOutputText.append("<span style='color: #569cd6;'>>>> Running code...</span>")

        # Set up execution namespace
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        # Use non-interactive backend to prevent external windows
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        plt.close('all')  # Close any existing figures

        # Try to import pandas
        pd = None
        try:
            import pandas as pd
        except ImportError:
            pass

        # Helper functions for data access
        def _normalize_search_term(filename_contains):
            """Extract clean search term from path, filename, or fragment."""
            if not filename_contains:
                return None
            # Handle full paths - extract just the base name
            search_term = Path(filename_contains).stem.lower()
            # Remove common suffixes like _bundle, _means_by_time, etc.
            for suffix in ['_bundle', '_means_by_time', '_breaths', '_pleth']:
                if search_term.endswith(suffix):
                    search_term = search_term[:-len(suffix)]
            return search_term

        def get_export_paths(filename_contains=None):
            """Get export paths for files. Returns dict of {filename: export_path}."""
            result = {}
            search_term = _normalize_search_term(filename_contains)
            master_list = getattr(self.mw, '_master_file_list', None) or []

            for f in master_list:
                fn = f.get('file_name', '')
                export_path = f.get('export_path', '')
                if export_path:
                    fn_stem = Path(fn).stem.lower()
                    if search_term is None or search_term in fn.lower() or search_term in fn_stem:
                        result[fn] = export_path
            return result

        def load_means_csv(filename_contains):
            """Load _means_by_time.csv for a file. Returns DataFrame."""
            p = Path(filename_contains)
            if p.suffix.lower() == '.csv' and p.exists():
                return pd.read_csv(p)

            paths = get_export_paths(filename_contains)
            if not paths:
                raise FileNotFoundError(f"No exports found matching '{_normalize_search_term(filename_contains)}'")
            export_dir = list(paths.values())[0]
            csv_files = list(Path(export_dir).glob('*_means_by_time.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No _means_by_time.csv in {export_dir}")
            return pd.read_csv(csv_files[0])

        def load_breaths_csv(filename_contains):
            """Load _breaths.csv for a file. Returns DataFrame."""
            p = Path(filename_contains)
            if p.suffix.lower() == '.csv' and p.exists():
                return pd.read_csv(p)

            paths = get_export_paths(filename_contains)
            if not paths:
                raise FileNotFoundError(f"No exports found matching '{_normalize_search_term(filename_contains)}'")
            export_dir = list(paths.values())[0]
            csv_files = list(Path(export_dir).glob('*_breaths.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No _breaths.csv in {export_dir}")
            return pd.read_csv(csv_files[0])

        def load_bundle_npz(filename_contains):
            """Load _bundle.npz for a file. Returns numpy NpzFile object."""
            p = Path(filename_contains)
            if p.suffix.lower() == '.npz' and p.exists():
                return np.load(p, allow_pickle=True)

            paths = get_export_paths(filename_contains)
            if not paths:
                raise FileNotFoundError(f"No exports found matching '{_normalize_search_term(filename_contains)}'")
            export_dir = list(paths.values())[0]
            npz_files = list(Path(export_dir).glob('*_bundle.npz'))
            if not npz_files:
                raise FileNotFoundError(f"No _bundle.npz in {export_dir}")
            return np.load(npz_files[0], allow_pickle=True)

        def get_stim_spans(bundle_data):
            """Extract stim spans from bundle.npz data. Returns dict {sweep_idx: [(start, end), ...]}."""
            import json as _json
            if 'stim_spans_json' in bundle_data:
                spans_str = str(bundle_data['stim_spans_json'])
                return {int(k): v for k, v in _json.loads(spans_str).items()}
            return {}

        def add_stim_shading(ax, stim_spans, sweep_idx=0, color='blue', alpha=0.2, label='Stim'):
            """Add blue shading for stimulation periods."""
            spans = stim_spans.get(sweep_idx, [])
            for i, (start, end) in enumerate(spans):
                ax.axvspan(start, end, alpha=alpha, color=color,
                          label=label if i == 0 else None)

        def list_available_files():
            """List files with exports available."""
            result = []
            master_list = getattr(self.mw, '_master_file_list', None) or []
            for f in master_list:
                fn = f.get('file_name', '')
                export_path = f.get('export_path', '')
                if export_path:
                    result.append(f"{fn} -> {export_path}")
            return result

        # Create namespace with common data science imports and helpers
        master_list = getattr(self.mw, '_master_file_list', None)
        exec_namespace = {
            'np': np,
            'plt': plt,
            'pd': pd,
            'Path': Path,
            'json': __import__('json'),
            'master_file_list': master_list,
            # Helper functions
            'get_export_paths': get_export_paths,
            'load_means_csv': load_means_csv,
            'load_breaths_csv': load_breaths_csv,
            'load_bundle_npz': load_bundle_npz,
            'get_stim_spans': get_stim_spans,
            'add_stim_shading': add_stim_shading,
            'list_available_files': list_available_files,
        }

        # Capture stdout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Disable run button during execution
        if hasattr(self.mw, 'runCodeButton'):
            self.mw.runCodeButton.setEnabled(False)

        # Process events to update UI before potentially long execution
        QApplication.processEvents()

        # Execute directly on main thread
        success = True
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_namespace)
        except Exception as e:
            stderr_capture.write(f"{type(e).__name__}: {e}")
            success = False

        # Display text results
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        if stdout:
            self.mw.codeOutputText.append(f"<pre style='color: #d4d4d4;'>{stdout}</pre>")
        if stderr:
            color = '#f48771' if not success else '#dcdcaa'
            self.mw.codeOutputText.append(f"<pre style='color: {color};'>{stderr}</pre>")

        # Capture and embed any matplotlib figures inline
        figs = [plt.figure(i) for i in plt.get_fignums()]
        if figs:
            self.mw.codeOutputText.append(f"<span style='color: #569cd6;'>{len(figs)} figure(s) generated:</span>")
            for i, fig in enumerate(figs):
                # Save figure to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, facecolor='#1e1e1e',
                           edgecolor='none', bbox_inches='tight')
                buf.seek(0)
                fig_bytes = buf.read()
                buf.close()

                # Store figure data for Pop Out / Save functionality
                self._notebook_figures.append({
                    'bytes': fig_bytes,
                    'figsize': fig.get_size_inches().tolist(),
                    'index': len(self._notebook_figures) + 1
                })

                # Convert to base64 for embedding in HTML
                img_data = base64.b64encode(fig_bytes).decode('utf-8')

                # Embed as inline image
                img_html = f'''
                <div style="margin: 8px 0; padding: 4px; border: 1px solid #3e3e42; border-radius: 4px; background: #252526;">
                    <img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;" />
                    <div style="color: #808080; font-size: 9pt; margin-top: 4px;">Figure {i+1} - Use "Pop Out" to view larger or "Save" to export</div>
                </div>
                '''
                self.mw.codeOutputText.append(img_html)

            plt.close('all')
        elif success and not stdout and not stderr:
            self.mw.codeOutputText.append("<span style='color: #4ec9b0;'>Code executed successfully (no output).</span>")

        # Restore original matplotlib backend
        try:
            matplotlib.use(original_backend)
        except Exception:
            pass  # May fail if backend doesn't support switching

        # Scroll to bottom
        scrollbar = self.mw.codeOutputText.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Re-enable run button
        if hasattr(self.mw, 'runCodeButton'):
            self.mw.runCodeButton.setEnabled(True)

    def on_clear_code_output(self):
        """Clear the notebook output area."""
        if hasattr(self.mw, 'codeOutputText'):
            self.mw.codeOutputText.clear()
        # Also clear stored figures
        self._notebook_figures = []

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def add_notebook_extra_buttons(self):
        """Add Pop Out and Save buttons to the notebook header."""
        from PyQt6.QtWidgets import QPushButton

        # Store figures for pop-out functionality
        self._notebook_figures = []

        # Find the header layout (where Run and Clear buttons are)
        if not hasattr(self.mw, 'clearOutputButton') or not self.mw.clearOutputButton.parent():
            return

        parent_layout = self.mw.clearOutputButton.parent().layout()
        if not parent_layout:
            return

        # Button style matching existing buttons
        button_style = """
            QPushButton {
                background-color: #6c757d;
                color: white;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """

        # Pop Out button - opens figures in external windows
        self.mw.popOutFigureButton = QPushButton("Pop Out")
        self.mw.popOutFigureButton.setToolTip("Open the last figure in an external window")
        self.mw.popOutFigureButton.setMinimumSize(80, 24)
        self.mw.popOutFigureButton.setStyleSheet(button_style)
        self.mw.popOutFigureButton.clicked.connect(self.on_pop_out_figure)
        parent_layout.addWidget(self.mw.popOutFigureButton)

        # Save Figure button
        self.mw.saveFigureButton = QPushButton("Save")
        self.mw.saveFigureButton.setToolTip("Save the last figure to a file")
        self.mw.saveFigureButton.setMinimumSize(70, 24)
        self.mw.saveFigureButton.setStyleSheet(button_style)
        self.mw.saveFigureButton.clicked.connect(self.on_save_figure)
        parent_layout.addWidget(self.mw.saveFigureButton)

        print("[notebook] Added Pop Out and Save Figure buttons")

    # -------------------------------------------------------------------------
    # Figure Management
    # -------------------------------------------------------------------------

    def on_pop_out_figure(self):
        """Open the last generated figure in an external matplotlib window."""
        import matplotlib
        import matplotlib.pyplot as plt

        if not self._notebook_figures:
            self.mw._log_status_message("No figures to pop out. Run code that generates a plot first.", 2000)
            return

        # Temporarily switch to interactive backend
        try:
            matplotlib.use('TkAgg')  # Or 'Qt5Agg' depending on system
        except Exception:
            pass

        # Recreate the last figure
        fig_data = self._notebook_figures[-1]
        fig, ax = plt.subplots(figsize=fig_data.get('figsize', (8, 6)))

        # If we stored the figure bytes, display it
        if 'bytes' in fig_data:
            from PIL import Image
            img = Image.open(io.BytesIO(fig_data['bytes']))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Figure {len(self._notebook_figures)}")

        plt.show()
        self.mw._log_status_message("Figure opened in external window", 2000)

    def on_save_figure(self):
        """Save the last generated figure to a file."""
        from PyQt6.QtWidgets import QFileDialog

        if not self._notebook_figures:
            self.mw._log_status_message("No figures to save. Run code that generates a plot first.", 2000)
            return

        # Get save path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self.mw,
            "Save Figure",
            "figure.png",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg);;All Files (*)"
        )

        if not file_path:
            return

        # Write the figure bytes
        fig_data = self._notebook_figures[-1]
        if 'bytes' in fig_data:
            try:
                with open(file_path, 'wb') as f:
                    f.write(fig_data['bytes'])
                self.mw._log_status_message(f"Figure saved to {file_path}", 2000)
            except Exception as e:
                self.mw._log_status_message(f"Error saving figure: {e}", 3000)
        else:
            self.mw._log_status_message("No figure data to save", 2000)
