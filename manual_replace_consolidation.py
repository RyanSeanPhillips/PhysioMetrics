"""
Manually replace consolidation methods with thin delegation wrappers.
This version is simpler and more reliable than the automated script.
"""
import re
from pathlib import Path

# Read main.py
main_py = Path("main.py")
lines = main_py.read_text(encoding='utf-8').splitlines()

# Methods to replace (in order they appear)
methods_info = [
    ("_propose_consolidated_filename", 2149, "files"),
    ("on_consolidate_save_data_clicked", 2243, ""),
    ("_consolidate_breaths_histograms", 2445, "files"),
    ("_consolidate_events", 2663, "files"),
    ("_consolidate_stimulus", 2709, "files"),
    ("_try_load_npz_v2", 2771, "npz_path"),
    ("_extract_timeseries_from_npz", 2801, "npz_data, metric, variant"),
    ("_consolidate_from_npz_v2", 2827, "npz_data_by_root, files, metrics"),
    ("_consolidate_means_files", 2992, "files"),
    ("_consolidate_breaths_sighs", 3361, "files"),
    ("_save_consolidated_to_excel", 3437, "consolidated, save_path"),
    ("_add_events_charts", 3957, "ws, header_row"),
    ("_add_sighs_chart", 4120, "ws, header_row"),
]

# Find the end of each method by looking for the next method at same indent level
def find_method_end(start_line):
    """Find where this method ends (next method or class-level code)."""
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        # Check for next method at same indentation
        if re.match(r'^    def \w+\(', line):
            return i
        # Check for class-level code (dedent)
        if line and not line.startswith(' ') and not line.startswith('\t'):
            return i
    return len(lines)

# Build replacement plan
replacements = []
for method_name, start_line_num, params in methods_info:
    start_idx = start_line_num - 1  # Convert to 0-indexed
    end_idx = find_method_end(start_idx)

    # Get method signature from original
    signature = lines[start_idx]

    # Build delegation method
    if params:
        delegation = [
            signature,
            f'        """Delegate to ConsolidationManager."""',
            f'        return self.consolidation_manager.{method_name}({params})',
            ''
        ]
    else:
        delegation = [
            signature,
            f'        """Delegate to ConsolidationManager."""',
            f'        return self.consolidation_manager.{method_name}()',
            ''
        ]

    replacements.append({
        'name': method_name,
        'start': start_idx,
        'end': end_idx,
        'original_lines': end_idx - start_idx,
        'delegation': delegation
    })

    print(f"{method_name}: lines {start_line_num}-{end_idx} ({end_idx - start_idx} lines) -> {len(delegation)} lines")

# Apply replacements in REVERSE order to preserve line numbers
replacements.reverse()

for r in replacements:
    # Delete old method
    del lines[r['start']:r['end']]
    # Insert delegation
    for i, line in enumerate(r['delegation']):
        lines.insert(r['start'] + i, line)

# Add import at top
import_line = 'from consolidation import ConsolidationManager'
import_idx = 0
for i, line in enumerate(lines):
    if line.startswith('from ') or line.startswith('import '):
        import_idx = i + 1
    elif import_idx > 0 and line.strip() == '':
        break

lines.insert(import_idx, import_line)
print(f"\nAdded import at line {import_idx + 1}")

# Add consolidation_manager initialization in __init__
init_pattern = r'^\s+def __init__\('
init_found = False
for i, line in enumerate(lines):
    if re.match(init_pattern, line):
        # Look for super().__init__() or similar
        for j in range(i + 1, min(i + 100, len(lines))):
            if 'super().__init__()' in lines[j] or 'QMainWindow.__init__' in lines[j]:
                # Insert after this line
                lines.insert(j + 1, '')
                lines.insert(j + 2, '        # Initialize managers')
                lines.insert(j + 3, '        self.consolidation_manager = ConsolidationManager(self)')
                print(f"Added consolidation_manager initialization at line {j + 3}")
                init_found = True
                break
        break

if not init_found:
    print("WARNING: Could not find __init__ to add consolidation_manager initialization")

# Write output
output = '\n'.join(lines)
main_py.write_text(output, encoding='utf-8')

original_total = sum(r['original_lines'] for r in replacements)
delegation_total = sum(len(r['delegation']) for r in replacements)

print(f"\n=== Summary ===")
print(f"Replaced {len(replacements)} methods")
print(f"Original: {original_total} lines")
print(f"Delegation: {delegation_total} lines")
print(f"Reduction: {original_total - delegation_total} lines")
print("\nDone! main.py has been updated.")
