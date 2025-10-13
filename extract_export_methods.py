"""
Script to extract export methods from main.py and create export_manager.py
"""
import re
from pathlib import Path

def extract_methods():
    main_py = Path("main.py")
    content = main_py.read_text(encoding='utf-8')
    lines = content.splitlines()

    # Methods to extract (method_name, start_line_approx)
    methods_to_extract = [
        '_load_save_dialog_history',
        '_update_save_dialog_history',
        '_sanitize_token',
        '_suggest_stim_string',
        'on_save_analyzed_clicked',
        'on_view_summary_clicked',
        '_metric_keys_in_order',
        '_compute_metric_trace',
        '_get_stim_masks',
        '_nanmean_sem',
        '_export_all_analyzed_data',
        '_mean_sem_1d',
        '_save_metrics_summary_pdf',
        '_show_summary_preview_dialog',
        '_sigh_sample_indices',
    ]

    # Find all method definitions
    method_starts = {}
    for i, line in enumerate(lines):
        if re.match(r'^    def \w+\(', line):
            method_name = re.search(r'def (\w+)\(', line).group(1)
            method_starts[method_name] = i

    # Extract each method
    extracted_methods = {}
    for method_name in methods_to_extract:
        if method_name not in method_starts:
            print(f"Warning: Method {method_name} not found")
            continue

        start_idx = method_starts[method_name]

        # Find end of method (next method at same indent or end of class)
        end_idx = len(lines)
        for j in range(start_idx + 1, len(lines)):
            line = lines[j]
            # Check for next method at same level or less indented
            if re.match(r'^    def \w+\(', line) or re.match(r'^\w', line):
                end_idx = j
                break

        method_lines = lines[start_idx:end_idx]
        extracted_methods[method_name] = method_lines
        print(f"Extracted {method_name}: {len(method_lines)} lines")

    return extracted_methods

def transform_to_manager_class(methods):
    """Transform extracted methods into ExportManager class format"""
    output_lines = []

    # Add header
    output_lines.extend([
        '"""',
        'ExportManager - Handles all data export operations.',
        '',
        'Extracted from main.py for better maintainability and easier customization',
        'for different experiment types.',
        '"""',
        '',
        'import re',
        'import csv',
        'import json',
        'import time',
        'import numpy as np',
        'import pandas as pd',
        'from pathlib import Path',
        'from PyQt6.QtWidgets import QMessageBox, QFileDialog, QDialog, QProgressDialog, QApplication',
        'from PyQt6.QtCore import Qt',
        'from core import metrics',
        'from dialogs import SaveMetaDialog',
        '',
        '',
        'class ExportManager:',
        '    """Manages all data export operations for the main window."""',
        '',
        '    # metrics we won\'t include in CSV exports and PDFs',
        '    _EXCLUDE_FOR_CSV = {"d1", "d2", "eupnic", "apnea", "regularity"}',
        '',
        '    def __init__(self, main_window):',
        '        """',
        '        Initialize the ExportManager.',
        '',
        '        Args:',
        '            main_window: Reference to MainWindow instance',
        '        """',
        '        self.window = main_window',
        '',
    ])

    # Add each method
    for method_name, method_lines in methods.items():
        # Transform the method
        transformed = []
        for line in method_lines:
            # Change indentation from 4 to 4 (keep same since we're in a class)
            # Change self. to self.window. (except for self.window, self._methods)
            if line.strip():
                # Replace self. with self.window. but be careful
                new_line = line

                # Replace self.state with self.window.state
                new_line = re.sub(r'\bself\.state\b', 'self.window.state', new_line)
                # Replace self.settings with self.window.settings
                new_line = re.sub(r'\bself\.settings\b', 'self.window.settings', new_line)
                # Replace self.statusbar with self.window.statusbar
                new_line = re.sub(r'\bself\.statusbar\b', 'self.window.statusbar', new_line)
                # Replace self.navigation_manager with self.window.navigation_manager
                new_line = re.sub(r'\bself\.navigation_manager\b', 'self.window.navigation_manager', new_line)
                # Replace other self._method calls with self._method (keep internal calls)
                # But replace self.other_methods with self.window.other_methods
                # This is tricky - we need to keep self._export_all_analyzed_data but change self._compute_stim_for_current_sweep

                # For calls to other main window methods not in ExportManager
                if '_compute_stim_for_current_sweep' in new_line:
                    new_line = re.sub(r'\bself\._compute_stim_for_current_sweep', 'self.window._compute_stim_for_current_sweep', new_line)
                if '_save_dir' in new_line or '_save_base' in new_line or '_save_meta' in new_line:
                    new_line = re.sub(r'\bself\._save_dir\b', 'self.window._save_dir', new_line)
                    new_line = re.sub(r'\bself\._save_base\b', 'self.window._save_base', new_line)
                    new_line = re.sub(r'\bself\._save_meta\b', 'self.window._save_meta', new_line)

                transformed.append(new_line)
            else:
                transformed.append(line)

        output_lines.extend(transformed)
        output_lines.append('')  # blank line between methods

    return '\n'.join(output_lines)

if __name__ == '__main__':
    print("Extracting export methods from main.py...")
    methods = extract_methods()

    print(f"\nTotal methods extracted: {len(methods)}")
    print(f"Total lines: {sum(len(m) for m in methods.values())}")

    print("\nTransforming to ExportManager class...")
    output = transform_to_manager_class(methods)

    output_path = Path("export/export_manager.py")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(output, encoding='utf-8')

    print(f"\nCreated {output_path} ({len(output.splitlines())} lines)")
    print("Done!")
