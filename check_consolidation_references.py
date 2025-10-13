"""
Verification script to check for missing self.window references in consolidation_manager.py
"""
import re
from pathlib import Path

def check_references():
    """Check for potential missing self.window references."""

    consolidation_file = Path("consolidation/consolidation_manager.py")
    content = consolidation_file.read_text(encoding='utf-8')
    lines = content.splitlines()

    # Patterns to check
    issues = []

    # 1. Check for bare self.state (should be self.window.state)
    for i, line in enumerate(lines, 1):
        if re.search(r'\bself\.state\b', line) and 'self.window.state' not in line:
            issues.append({
                'line': i,
                'type': 'state',
                'content': line.strip()
            })

    # 2. Check for bare self.settings (should be self.window.settings)
    for i, line in enumerate(lines, 1):
        if re.search(r'\bself\.settings\b', line) and 'self.window.settings' not in line:
            issues.append({
                'line': i,
                'type': 'settings',
                'content': line.strip()
            })

    # 3. Check for bare self.statusbar (should be self.window.statusbar)
    for i, line in enumerate(lines, 1):
        if re.search(r'\bself\.statusbar\b', line) and 'self.window.statusbar' not in line:
            issues.append({
                'line': i,
                'type': 'statusbar',
                'content': line.strip()
            })

    # 4. Check for UI widgets (should have self.window prefix)
    ui_widgets = [
        'FilestoConsolidateList',
        'ConsolidateGroupComboBox',
    ]

    for widget in ui_widgets:
        for i, line in enumerate(lines, 1):
            if re.search(rf'\bself\.{widget}\b', line) and f'self.window.{widget}' not in line:
                issues.append({
                    'line': i,
                    'type': 'widget',
                    'widget': widget,
                    'content': line.strip()
                })

    # 5. Check for external methods that should have self.window prefix
    external_methods = [
        '_get_processed_for',
        '_parse_float',
        '_compute_stim_for_current_sweep',
    ]

    for method in external_methods:
        for i, line in enumerate(lines, 1):
            # Only flag if it's self.method() and not self.window.method()
            if re.search(rf'\bself\.{method}\b', line) and f'self.window.{method}' not in line:
                # Make sure it's not just a comment
                if not line.strip().startswith('#'):
                    issues.append({
                        'line': i,
                        'type': 'external_method',
                        'method': method,
                        'content': line.strip()
                    })

    # 6. Check for QDialog parent issues (should use self.window, not self)
    for i, line in enumerate(lines, 1):
        # Look for dialog constructors
        if any(x in line for x in ['QMessageBox', 'QFileDialog', 'QProgressDialog']):
            # Check if 'self)' is used as parent (should be 'self.window)')
            if re.search(r'(\w+)\(self\)', line) and 'self.window' not in line:
                # Extract just the context around 'self)'
                issues.append({
                    'line': i,
                    'type': 'dialog_parent',
                    'content': line.strip()
                })

    # 7. Check for internal method calls (these should stay as self.)
    # This is informational - we want to verify internal calls are NOT changed to self.window
    internal_methods = [
        '_propose_consolidated_filename',
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

    correct_internal_calls = []
    for method in internal_methods:
        for i, line in enumerate(lines, 1):
            if re.search(rf'\bself\.{method}\(', line) and 'self.window' not in line:
                correct_internal_calls.append({
                    'line': i,
                    'method': method,
                    'content': line.strip()
                })

    # Print results
    print("=" * 80)
    print("CONSOLIDATION REFERENCE VERIFICATION")
    print("=" * 80)

    if not issues:
        print("\n[OK] No issues found! All references appear correct.")
    else:
        print(f"\n[WARNING] Found {len(issues)} potential issues:\n")

        # Group by type
        by_type = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(issue)

        for issue_type, type_issues in by_type.items():
            print(f"\n{issue_type.upper()} issues ({len(type_issues)}):")
            print("-" * 80)
            for issue in type_issues:
                print(f"  Line {issue['line']}: {issue['content']}")
                if issue_type == 'widget':
                    print(f"    -> Should be: self.window.{issue['widget']}")
                elif issue_type == 'external_method':
                    print(f"    -> Should be: self.window.{issue['method']}")
                elif issue_type == 'dialog_parent':
                    print(f"    -> Should use: self.window as parent")
            print()

    # Print internal call verification
    if correct_internal_calls:
        print("\n" + "=" * 80)
        print(f"INTERNAL METHOD CALLS (should be self., not self.window.): {len(correct_internal_calls)}")
        print("=" * 80)
        print("These are CORRECT - internal calls should stay as self.method():")
        for call in correct_internal_calls[:10]:  # Show first 10
            print(f"  Line {call['line']}: {call['content']}")
        if len(correct_internal_calls) > 10:
            print(f"  ... and {len(correct_internal_calls) - 10} more")

    print("\n" + "=" * 80)

    return len(issues)

if __name__ == '__main__':
    num_issues = check_references()

    if num_issues == 0:
        print("\n[OK] Verification passed! ConsolidationManager appears correctly transformed.")
    else:
        print(f"\n[WARNING] Please review and fix the {num_issues} issues above.")
        print("\nNote: The extraction script should have handled most transformations.")
        print("      Manual fixes may be needed for edge cases.")
