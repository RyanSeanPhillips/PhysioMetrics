"""
Script to replace consolidation methods in main.py with delegation calls.
"""
import re
from pathlib import Path

def replace_methods():
    main_py = Path("main.py")
    content = main_py.read_text(encoding='utf-8')
    lines = content.splitlines()

    # Methods to replace
    methods_to_replace = [
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

    # Find method starts
    method_starts = {}
    for i, line in enumerate(lines):
        if re.match(r'^    def \w+\(', line):
            method_name = re.search(r'def (\w+)\(', line).group(1)
            method_starts[method_name] = i

    # Replace each method
    modifications = []
    for method_name in methods_to_replace:
        if method_name not in method_starts:
            print(f"Warning: Method {method_name} not found")
            continue

        start_idx = method_starts[method_name]

        # Extract method signature
        signature_line = lines[start_idx]

        # Find end of method
        end_idx = len(lines)
        for j in range(start_idx + 1, len(lines)):
            line = lines[j]
            if re.match(r'^    def \w+\(', line) or re.match(r'^\w', line):
                end_idx = j
                break

        # Extract original docstring if present
        docstring_lines = []
        check_idx = start_idx + 1
        if check_idx < len(lines) and '"""' in lines[check_idx]:
            # Multi-line docstring
            docstring_lines.append(lines[check_idx])
            check_idx += 1
            while check_idx < end_idx and '"""' not in lines[check_idx]:
                docstring_lines.append(lines[check_idx])
                check_idx += 1
            if check_idx < end_idx and '"""' in lines[check_idx]:
                docstring_lines.append(lines[check_idx])

        # Create delegation method
        delegation_lines = [signature_line]
        if docstring_lines:
            delegation_lines.extend(docstring_lines)

        # Extract parameters from signature (everything between parentheses)
        params_match = re.search(r'def \w+\((.*?)\):', signature_line)
        if params_match:
            params = params_match.group(1)
            # Remove 'self,' and whitespace
            params_clean = re.sub(r'^\s*self\s*,?\s*', '', params)
            params_clean = params_clean.strip()

            # Remove type annotations (e.g., "param: Type" -> "param")
            # Split by comma, strip type hints, rejoin
            if params_clean:
                param_list = []
                for param in params_clean.split(','):
                    param = param.strip()
                    # Remove type annotation (everything after :)
                    if ':' in param:
                        param = param.split(':')[0].strip()
                    # Remove default value (everything after =)
                    if '=' in param:
                        param = param.split('=')[0].strip()
                    if param:
                        param_list.append(param)
                call_params = ', '.join(param_list)
            else:
                call_params = ''
        else:
            call_params = ''

        # Add delegation call
        delegation_lines.append(f'        """Delegate to ConsolidationManager."""')
        if call_params:
            delegation_lines.append(f'        return self.consolidation_manager.{method_name}({call_params})')
        else:
            delegation_lines.append(f'        return self.consolidation_manager.{method_name}()')

        modifications.append({
            'name': method_name,
            'start': start_idx,
            'end': end_idx,
            'original_lines': end_idx - start_idx,
            'new_lines': delegation_lines
        })

        print(f"Will replace {method_name}: {end_idx - start_idx} lines -> {len(delegation_lines)} lines")

    # Apply modifications in reverse order (to preserve line numbers)
    modifications.sort(key=lambda x: x['start'], reverse=True)

    for mod in modifications:
        # Remove old method
        del lines[mod['start']:mod['end']]
        # Insert new delegation method
        for i, new_line in enumerate(reversed(mod['new_lines'])):
            lines.insert(mod['start'], new_line)

    # Add import at top of file (after existing imports)
    import_line = 'from consolidation import ConsolidationManager'

    # Find where to insert (after other 'from' imports)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            insert_idx = i + 1
        elif insert_idx > 0 and line.strip() == '':
            # Found blank line after imports
            break

    lines.insert(insert_idx, import_line)
    print(f"\nAdded import at line {insert_idx}: {import_line}")

    # Find __init__ method and add consolidation_manager initialization
    init_found = False
    for i, line in enumerate(lines):
        if re.match(r'^\s+def __init__\(', line):
            # Find a good place to add initialization (after super().__init__() or similar)
            for j in range(i + 1, min(i + 50, len(lines))):
                if 'super().__init__()' in lines[j] or 'QMainWindow.__init__' in lines[j]:
                    # Insert after this line
                    lines.insert(j + 1, '')
                    lines.insert(j + 2, '        # Initialize ConsolidationManager')
                    lines.insert(j + 3, '        self.consolidation_manager = ConsolidationManager(self)')
                    print(f"Added consolidation_manager initialization at line {j + 2}")
                    init_found = True
                    break
            break

    if not init_found:
        print("WARNING: Could not find __init__ method to add consolidation_manager initialization")

    # Write modified content
    output = '\n'.join(lines)
    main_py.write_text(output, encoding='utf-8')

    original_line_count = sum(m['original_lines'] for m in modifications)
    new_line_count = sum(len(m['new_lines']) for m in modifications)

    print(f"\n=== Summary ===")
    print(f"Modified main.py:")
    print(f"  - Replaced {len(modifications)} methods")
    print(f"  - Removed {original_line_count} lines")
    print(f"  - Added {new_line_count} delegation lines")
    print(f"  - Net reduction: {original_line_count - new_line_count} lines")
    print(f"  - Added import statement")
    print(f"  - Added consolidation_manager initialization")

if __name__ == '__main__':
    print("Replacing consolidation methods in main.py with delegation calls...")
    replace_methods()
    print("\nDone! Please verify main.py compiles correctly.")
