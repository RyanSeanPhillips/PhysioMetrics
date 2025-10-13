"""
Script to find potential missing self.window references in export_manager.py
"""
import re
from pathlib import Path

def check_references():
    export_file = Path("export/export_manager.py")
    content = export_file.read_text(encoding='utf-8')

    # Find all self.XXX references (excluding self.window, self._method_in_export_manager)
    pattern = r'\bself\.([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(pattern, content)

    # Known methods/properties that SHOULD be in ExportManager
    export_manager_methods = {
        'window',  # The main window reference
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
    }

    # Find unique references
    unique_refs = set(matches)

    # Filter to potential problems (not in export_manager_methods)
    potential_issues = unique_refs - export_manager_methods

    print(f"Found {len(unique_refs)} unique self.XXX references")
    print(f"\n{len(export_manager_methods)} are expected ExportManager methods/properties")
    print(f"\n{len(potential_issues)} potential issues (may need self.window.):\n")

    # Count occurrences of each
    issue_counts = {}
    for ref in potential_issues:
        count = len(re.findall(r'\bself\.' + re.escape(ref) + r'\b', content))
        issue_counts[ref] = count

    # Sort by count (most frequent first)
    for ref, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  self.{ref:<40} ({count:3} occurrences)")

    print("\nNOTE: Some of these may be legitimate (e.g., local variables).")
    print("Review each one to determine if it should be self.window.XXX")

    # Also check for Qt dialog/widget patterns that need parent=self.window
    print("\n" + "="*60)
    print("Checking for Qt widgets/dialogs that may need self.window parent:")
    print("="*60)

    qt_patterns = [
        (r'QMessageBox\.(information|warning|critical|question)\(self,', 'QMessageBox with self parent'),
        (r'QFileDialog\.[a-zA-Z]+\(self,', 'QFileDialog with self parent'),
        (r'QDialog\(self\)', 'QDialog with self parent'),
        (r'QProgressDialog\([^)]*self\)', 'QProgressDialog with self parent'),
    ]

    for pattern, description in qt_patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"\n[!] Found {len(matches)} instances of: {description}")
            # Show line numbers
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    print(f"   Line {i}: {line.strip()[:80]}")
        else:
            print(f"[OK] No issues: {description}")

if __name__ == '__main__':
    check_references()
