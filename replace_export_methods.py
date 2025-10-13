"""
Replace export methods in main.py with delegation calls
"""
import re
from pathlib import Path

def replace_methods():
    # Read main.py
    content = Path('main.py').read_text(encoding='utf-8')
    lines = content.splitlines()

    # Methods to replace (method_name, replacement)
    replacements = {
        '_load_save_dialog_history': 'return self.export_manager._load_save_dialog_history()',
        '_update_save_dialog_history': 'return self.export_manager._update_save_dialog_history(vals)',
        '_sanitize_token': 'return self.export_manager._sanitize_token(s)',
        '_suggest_stim_string': 'return self.export_manager._suggest_stim_string()',
        'on_save_analyzed_clicked': 'return self.export_manager.on_save_analyzed_clicked()',
        'on_view_summary_clicked': 'return self.export_manager.on_view_summary_clicked()',
        '_metric_keys_in_order': 'return self.export_manager._metric_keys_in_order()',
        '_compute_metric_trace': 'return self.export_manager._compute_metric_trace(key, t, y, sr_hz, peaks, breaths)',
        '_get_stim_masks': 'return self.export_manager._get_stim_masks(s)',
        '_nanmean_sem': 'return self.export_manager._nanmean_sem(X, axis)',
        '_export_all_analyzed_data': 'return self.export_manager._export_all_analyzed_data(preview_only, progress_dialog)',
        '_mean_sem_1d': 'return self.export_manager._mean_sem_1d(arr)',
        '_save_metrics_summary_pdf': 'return self.export_manager._save_metrics_summary_pdf(pdf_path, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, meta, stim_zero, stim_dur)',
        '_show_summary_preview_dialog': 'return self.export_manager._show_summary_preview_dialog(t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, stim_zero, stim_dur)',
        '_sigh_sample_indices': 'return self.export_manager._sigh_sample_indices(s, pks)',
    }

    # Find and replace each method
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a method definition we want to replace
        matched = False
        for method_name, delegation in replacements.items():
            pattern = r'^    def ' + re.escape(method_name) + r'\('
            if re.match(pattern, line):
                # Found it - find the end of the method
                end_idx = i + 1
                for j in range(i + 1, len(lines)):
                    if re.match(r'^    def \w+\(', lines[j]) or re.match(r'^\w', lines[j]):
                        end_idx = j
                        break

                # Keep the signature line and docstring, replace body with delegation
                new_lines.append(line)  # def line

                # Check for docstring
                doc_start = i + 1
                if doc_start < len(lines) and '"""' in lines[doc_start]:
                    # Find end of docstring
                    doc_end = doc_start
                    if lines[doc_start].count('"""') >= 2:
                        # Single-line docstring
                        doc_end = doc_start + 1
                    else:
                        # Multi-line docstring
                        for k in range(doc_start + 1, len(lines)):
                            if '"""' in lines[k]:
                                doc_end = k + 1
                                break
                    # Keep docstring
                    for k in range(doc_start, doc_end):
                        new_lines.append(lines[k])
                    # Add delegation
                    new_lines.append(f'        {delegation}')
                    new_lines.append('')
                else:
                    # No docstring, just add delegation
                    new_lines.append(f'        """Delegate to ExportManager."""')
                    new_lines.append(f'        {delegation}')
                    new_lines.append('')

                i = end_idx
                matched = True
                print(f'Replaced {method_name} (lines {i+1}-{end_idx})')
                break

        if not matched:
            new_lines.append(line)
            i += 1

    # Write back
    Path('main.py').write_text('\n'.join(new_lines), encoding='utf-8')
    print(f'\nWrote {len(new_lines)} lines to main.py')

if __name__ == '__main__':
    replace_methods()
    print('Done!')
