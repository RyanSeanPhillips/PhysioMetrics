"""
Script to extract consolidation methods from main.py and create consolidation_manager.py
"""
import re
from pathlib import Path

def extract_methods():
    main_py = Path("main.py")
    content = main_py.read_text(encoding='utf-8')
    lines = content.splitlines()

    # Methods to extract
    methods_to_extract = [
        '_propose_consolidated_filename',
        'on_consolidate_save_data_clicked',
        '_consolidate_breaths_histograms',
        '_consolidate_events',
        '_consolidate_stimulus',
        '_try_load_npz_v2',
        '_extract_timeseries_from_npz',
        '_consolidate_from_npz_v2',
        '_consolidate_means_files',
        '_consolidate_breaths_sighs',
        '_save_consolidated_to_excel',
        '_add_events_charts',
        '_add_sighs_chart',
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
    """Transform extracted methods into ConsolidationManager class format"""
    output_lines = []

    # Add header
    output_lines.extend([
        '"""',
        'ConsolidationManager - Handles multi-file data consolidation and Excel export.',
        '',
        'Extracted from main.py for better maintainability and easier customization',
        'for different experiment types.',
        '"""',
        '',
        'import re',
        'import numpy as np',
        'import pandas as pd',
        'from pathlib import Path',
        'from PyQt6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog, QApplication',
        'from PyQt6.QtCore import Qt',
        '',
        '',
        'class ConsolidationManager:',
        '    """Manages all multi-file consolidation operations for the main window."""',
        '',
        '    def __init__(self, main_window):',
        '        """',
        '        Initialize the ConsolidationManager.',
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
            if line.strip():
                new_line = line

                # Replace self.state with self.window.state
                new_line = re.sub(r'\bself\.state\b', 'self.window.state', new_line)
                # Replace self.settings with self.window.settings
                new_line = re.sub(r'\bself\.settings\b', 'self.window.settings', new_line)
                # Replace self.statusbar with self.window.statusbar
                new_line = re.sub(r'\bself\.statusbar\b', 'self.window.statusbar', new_line)

                # UI widgets that need self.window prefix
                widget_names = [
                    'FilestoConsolidateList',
                    'ConsolidateGroupComboBox',
                ]
                for widget in widget_names:
                    new_line = re.sub(rf'\bself\.{widget}\b', f'self.window.{widget}', new_line)

                # Methods that need self.window prefix (external to ConsolidationManager)
                external_methods = [
                    '_get_processed_for',
                    '_parse_float',
                    '_compute_stim_for_current_sweep',
                ]
                for method in external_methods:
                    new_line = re.sub(rf'\bself\.{method}\b', f'self.window.{method}', new_line)

                transformed.append(new_line)
            else:
                transformed.append(line)

        output_lines.extend(transformed)
        output_lines.append('')  # blank line between methods

    return '\n'.join(output_lines)

if __name__ == '__main__':
    print("Extracting consolidation methods from main.py...")
    methods = extract_methods()

    print(f"\nTotal methods extracted: {len(methods)}")
    print(f"Total lines: {sum(len(m) for m in methods.values())}")

    print("\nTransforming to ConsolidationManager class...")
    output = transform_to_manager_class(methods)

    output_path = Path("consolidation/consolidation_manager.py")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(output, encoding='utf-8')

    print(f"\nCreated {output_path} ({len(output.splitlines())} lines)")
    print("Done!")
