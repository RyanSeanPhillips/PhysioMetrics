"""
Recovery Manager - Handles corrupted project file recovery and AI repair.

Extracted from main.py to improve maintainability.
Contains:
- Corruption analysis and detection
- Metadata extraction from corrupted files
- AI-assisted repair functionality
- Backup-based recovery
- Recovery summary dialogs
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple
import json

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QGroupBox, QMessageBox
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow


class RecoveryManager:
    """Manages project file recovery and corruption handling.

    Handles:
    - Analyzing corrupted project files
    - Extracting metadata from corrupted JSON
    - AI-assisted smart repair
    - Backup-based recovery
    - Recovery summary dialogs
    """

    def __init__(self, main_window: 'QMainWindow'):
        """Initialize RecoveryManager.

        Args:
            main_window: Reference to MainWindow for widget access
        """
        self.mw = main_window

    def get_configured_ai_client(self):
        """Get AI client if configured, or None if not set up."""
        try:
            settings = QSettings("PhysioMetrics", "BreathAnalysis")
            provider = settings.value("ai/provider", "claude")
            api_key = settings.value(f"ai/{provider}_api_key", "")

            if not api_key:
                return None

            from core.ai_client import AIClient
            return AIClient(provider=provider, api_key=api_key)
        except Exception as e:
            print(f"[ai] Could not initialize AI client: {e}")
            return None

    def apply_recovered_metadata(self) -> int:
        """Apply recovered metadata from corrupted file to rescanned files.

        Returns:
            Count of applied fields
        """
        if not hasattr(self.mw, '_pending_recovery_metadata') or not self.mw._pending_recovery_metadata:
            return 0

        recovered = self.mw._pending_recovery_metadata
        applied_count = 0
        fields_applied = 0

        for task in self.mw._master_file_list:
            file_path = task.get('file_path')
            if not file_path:
                continue

            # Try to match by filename or relative path
            file_name = Path(file_path).name

            # Look for a match in recovered metadata
            matched_meta = None
            for recovered_path, meta in recovered.items():
                if recovered_path.endswith(file_name) or file_name in recovered_path:
                    matched_meta = meta
                    break

            if matched_meta:
                file_updated = False
                # Apply recovered metadata (only if not already set)
                for key in ['strain', 'animal_id', 'power', 'sex', 'channel', 'channels']:
                    if matched_meta.get(key) and not task.get(key):
                        task[key] = matched_meta[key]
                        fields_applied += 1
                        file_updated = True
                if file_updated:
                    applied_count += 1

        # Clear pending metadata
        self.mw._pending_recovery_metadata = None

        if applied_count > 0:
            # Refresh the table to show recovered data
            self.mw._rebuild_table_from_master_list()
            self.mw._log_status_message(f"Recovered metadata for {applied_count} files", 3000)
            print(f"[recovery] Applied {fields_applied} metadata fields to {applied_count} files")

        return fields_applied

    def apply_recovered_metadata_and_show_summary(self):
        """Apply recovered metadata and show summary dialog."""
        fields_applied = self.apply_recovered_metadata()

        # Get recovery stats
        stats = getattr(self.mw, '_pending_recovery_stats', {
            'method': 'rescan',
            'files_recovered': len(self.mw._master_file_list),
            'metadata_fields_recovered': fields_applied,
            'ai_tokens_used': 0,
            'errors': [],
        })
        stats['metadata_fields_recovered'] = fields_applied
        stats['files_recovered'] = len(self.mw._master_file_list)

        # Get project name
        project_name = getattr(self.mw, '_current_project_name', '') or "Unknown"

        # Clear pending stats
        self.mw._pending_recovery_stats = None

        # Show summary
        self.show_recovery_summary(stats, project_name)

    def show_recovery_summary(self, stats: dict, project_name: str):
        """Show a summary dialog after project recovery."""
        dialog = QDialog(self.mw)
        dialog.setWindowTitle("Recovery Complete")
        dialog.setMinimumSize(450, 300)
        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel(f"<b>Project '{project_name}' recovered successfully!</b>")
        layout.addWidget(header)

        # Build summary text
        summary_lines = []

        # Method used
        method_names = {
            'backup_merge': 'Backup restore with change merge',
            'rescan': 'Directory rescan with metadata recovery',
            'unknown': 'Recovery',
        }
        summary_lines.append(f"<b>Recovery method:</b> {method_names.get(stats.get('method', 'unknown'), 'Unknown')}")
        summary_lines.append("")

        # Stats
        summary_lines.append("<b>Results:</b>")
        summary_lines.append(f"  - Files in project: {stats.get('files_recovered', 0)}")
        summary_lines.append(f"  - Metadata fields recovered: {stats.get('metadata_fields_recovered', 0)}")

        if stats.get('ai_tokens_used', 0) > 0:
            summary_lines.append(f"  - AI tokens used: {stats['ai_tokens_used']}")

        # Errors
        if stats.get('errors'):
            summary_lines.append("")
            summary_lines.append("<b>Warnings/Errors:</b>")
            for error in stats['errors']:
                summary_lines.append(f"  ! {error}")

        # What was lost (if rescan method)
        if stats.get('method') == 'rescan':
            summary_lines.append("")
            summary_lines.append("<b>Note:</b> Recovery via rescan may have lost some metadata")
            summary_lines.append("(channels, custom fields) that wasn't in the corrupted portion.")
            summary_lines.append("Check your project and re-enter any missing information.")

        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setHtml("<br>".join(summary_lines))
        layout.addWidget(summary_text)

        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        dialog.exec()

    def extract_metadata_from_corrupted(self, project_path: Path) -> dict:
        """Extract whatever metadata we can from a corrupted JSON file."""
        try:
            with open(project_path, 'r', encoding='utf-8', errors='ignore') as f:
                corrupted_content = f.read()

            import re
            recovered_metadata = {}

            # First, try to find file_path entries
            file_path_pattern = r'"file_path"\s*:\s*"([^"]+)"'
            file_paths = re.findall(file_path_pattern, corrupted_content)

            for fp in set(file_paths):  # dedupe
                if fp:
                    recovered_metadata[fp] = {}

            # Now extract individual fields for each file
            # We search for each field near its file_path
            field_patterns = {
                'strain': r'"strain"\s*:\s*"([^"]*)"',
                'animal_id': r'"animal_id"\s*:\s*"([^"]*)"',
                'power': r'"power"\s*:\s*"([^"]*)"',
                'sex': r'"sex"\s*:\s*"([^"]*)"',
                'channel': r'"channel"\s*:\s*"([^"]*)"',
                'channels': r'"channels"\s*:\s*"([^"]*)"',
                'notes': r'"notes"\s*:\s*"([^"]*)"',
            }

            # Find each file entry block and extract fields
            # Look for blocks that start with file_path
            entry_pattern = r'\{\s*"file_path"\s*:\s*"([^"]+)"([^}]*)\}'
            entry_matches = re.findall(entry_pattern, corrupted_content, re.DOTALL)

            for file_path, entry_content in entry_matches:
                if file_path not in recovered_metadata:
                    recovered_metadata[file_path] = {}

                # Extract each field from this entry's content
                for field_name, field_pattern in field_patterns.items():
                    match = re.search(field_pattern, entry_content)
                    if match and match.group(1):
                        recovered_metadata[file_path][field_name] = match.group(1)

            # Count how many entries have actual metadata
            entries_with_data = sum(1 for meta in recovered_metadata.values() if any(meta.values()))

            print(f"[recovery] Extracted {len(recovered_metadata)} file entries ({entries_with_data} with metadata)")
            return recovered_metadata

        except Exception as e:
            print(f"[recovery] Failed to extract data: {e}")
            return {}

    def perform_project_recovery(self, project_path: Path, project_name: str,
                                  data_directory: Path, recovered_metadata: dict,
                                  use_ai: bool = False):
        """Perform the actual project recovery with optional AI enhancement."""
        import copy

        backup_path = project_path.with_suffix('.physiometrics.bak')
        backup_data = None

        # First, try to load the backup file as our baseline
        if backup_path.exists():
            try:
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)
                print(f"[recovery] Loaded backup file with {len(backup_data.get('files', []))} files")
                self.mw._log_status_message("Found valid backup - comparing with corrupted file...", 2000)
            except Exception as e:
                print(f"[recovery] Backup file also corrupted: {e}")
                backup_data = None

        # If we have both backup and AI, do smart diff-based recovery
        if use_ai and backup_data and recovered_metadata:
            self.mw._log_status_message("Using AI to recover changes since last save...", 3000)
            ai_client = self.get_configured_ai_client()
            if ai_client:
                try:
                    # Build a map of backup file metadata for comparison
                    backup_files = {f.get('file_path', ''): f for f in backup_data.get('files', [])}

                    # Find what's different in corrupted vs backup
                    changes = []
                    for file_path, meta in recovered_metadata.items():
                        backup_entry = backup_files.get(file_path, {})
                        # Check if this file has different metadata than backup
                        if meta != {k: backup_entry.get(k, '') for k in ['strain', 'animal_id', 'power', 'sex']}:
                            changes.append({
                                'file': file_path,
                                'backup_had': {k: backup_entry.get(k, '') for k in ['strain', 'animal_id', 'power', 'sex'] if backup_entry.get(k)},
                                'corrupted_has': {k: v for k, v in meta.items() if v}
                            })

                    if changes:
                        # Read the corrupted section for context
                        try:
                            with open(project_path, 'r', errors='ignore') as f:
                                corrupted_content = f.read()
                            # Find where corruption likely occurred (truncated JSON)
                            last_complete = corrupted_content.rfind('},')
                            corrupted_tail = corrupted_content[last_complete:] if last_complete > 0 else corrupted_content[-500:]
                        except:
                            corrupted_tail = "(could not read)"

                        prompt = f"""A JSON project file was corrupted during save. I have:
1. A valid backup from before the corruption
2. The corrupted file with partial data

The user was editing {len(changes)} file entries when corruption occurred.

Files that changed (comparing backup vs corrupted):
{json.dumps(changes[:10], indent=2)}  # Limit to 10 for token efficiency

The corrupted section ends with:
{corrupted_tail[:500]}

Based on the partial data in 'corrupted_has', what were the likely intended values?
Return ONLY a JSON object mapping file_path to the corrected metadata fields.
Only include fields you can confidently infer from the partial data."""

                        response = ai_client.complete(prompt)

                        # Parse AI suggestions
                        try:
                            content = response.content
                            json_start = content.find('{')
                            json_end = content.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                ai_fixes = json.loads(content[json_start:json_end])
                                ai_fixed = 0
                                for fp, meta in ai_fixes.items():
                                    if fp in recovered_metadata:
                                        for key, value in meta.items():
                                            if value:
                                                recovered_metadata[fp][key] = value
                                                ai_fixed += 1
                                print(f"[recovery] AI repaired {ai_fixed} fields from diff analysis")
                        except json.JSONDecodeError:
                            print("[recovery] Could not parse AI diff-repair response")
                    else:
                        print("[recovery] No changes detected between backup and corrupted file")

                except Exception as e:
                    print(f"[recovery] AI diff-based recovery failed: {e}")

        # If we have backup data but no AI, merge backup metadata into recovered
        elif backup_data and not use_ai:
            backup_files = {f.get('file_path', ''): f for f in backup_data.get('files', [])}
            merged_count = 0
            for file_path in recovered_metadata:
                if file_path in backup_files:
                    backup_entry = backup_files[file_path]
                    # Use backup values for any missing fields
                    for key in ['strain', 'animal_id', 'power', 'sex']:
                        if backup_entry.get(key) and not recovered_metadata[file_path].get(key):
                            recovered_metadata[file_path][key] = backup_entry[key]
                            merged_count += 1
            if merged_count > 0:
                print(f"[recovery] Merged {merged_count} fields from backup file")

        # Fallback: AI without backup - just analyze file paths for patterns
        elif use_ai and recovered_metadata and not backup_data:
            self.mw._log_status_message("No backup available - AI analyzing file patterns...", 3000)
            ai_client = self.get_configured_ai_client()
            if ai_client:
                try:
                    # Only send file paths (not full metadata) for efficiency
                    file_paths = list(recovered_metadata.keys())[:20]  # Limit for tokens
                    prompt = f"""Analyze these file paths from a neuroscience experiment and suggest metadata.

File paths:
{json.dumps(file_paths, indent=2)}

Look for patterns like:
- Animal IDs (numbers like 25729)
- Power levels (e.g., "10mW")
- Strain names (e.g., "VgatCre", "C57")

Return ONLY a JSON object mapping file_path to suggested {{strain, animal_id, power}} fields.
Only include fields where you're confident."""

                    response = ai_client.complete(prompt)
                    content = response.content
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        ai_suggestions = json.loads(content[json_start:json_end])
                        for fp, meta in ai_suggestions.items():
                            if fp in recovered_metadata:
                                for key, value in meta.items():
                                    if value and not recovered_metadata[fp].get(key):
                                        recovered_metadata[fp][key] = value
                except Exception as e:
                    print(f"[recovery] AI pattern analysis failed: {e}")

        # Track recovery stats for summary
        recovery_stats = {
            'method': 'unknown',
            'files_recovered': 0,
            'metadata_fields_recovered': 0,
            'ai_tokens_used': getattr(self.mw, '_last_ai_tokens', 0),
            'errors': [],
        }

        # STRATEGY 1: If we have valid backup data, use it directly (preserves ALL metadata)
        if backup_data:
            self.mw._log_status_message("Restoring from backup with recovered changes...", 2000)
            recovery_stats['method'] = 'backup_merge'

            try:
                # Start with backup data as our base
                # Deep copy to avoid modifying original
                project_data = copy.deepcopy(backup_data)

                # Build filename-based map for matching (backup has relative paths, recovered has absolute)
                # Map filename -> index in project_data['files']
                backup_by_filename = {}
                for i, f in enumerate(project_data.get('files', [])):
                    fp = f.get('file_path', '')
                    if fp:
                        filename = Path(fp).name
                        backup_by_filename[filename] = i

                # Merge any recovered metadata changes from corrupted file
                # recovered_metadata has absolute paths as keys
                for recovered_path, meta in recovered_metadata.items():
                    filename = Path(recovered_path).name
                    if filename in backup_by_filename:
                        idx = backup_by_filename[filename]
                        # Only update fields that have values in recovered data
                        # AND are different from backup (these are the user's recent changes)
                        for key, value in meta.items():
                            if value:  # Only apply non-empty values
                                old_value = project_data['files'][idx].get(key, '')
                                if value != old_value:
                                    project_data['files'][idx][key] = value
                                    recovery_stats['metadata_fields_recovered'] += 1
                                    print(f"[recovery] Merged change: {filename}.{key} = '{value}'")

                recovery_stats['files_recovered'] = len(project_data.get('files', []))

                # Convert relative paths back to absolute for save_project
                # (save_project expects absolute paths and converts them to relative)
                for file_entry in project_data.get('files', []):
                    fp = file_entry.get('file_path', '')
                    if fp and not Path(fp).is_absolute():
                        abs_path = (data_directory / fp).resolve()
                        file_entry['file_path'] = abs_path

                # Save the merged data
                self.mw.project_manager.save_project(
                    project_name=project_name,
                    data_directory=data_directory,
                    files_data=project_data.get('files', []),
                    experiments=project_data.get('experiments', []),
                    notes_directory=project_data.get('notes_directory'),
                    notes_files=project_data.get('notes_files', [])
                )

                # Load the recovered project
                saved_data = self.mw.project_manager.load_project(project_path)
                self.mw._populate_ui_from_project(saved_data)

                print(f"[recovery] Restored from backup with {recovery_stats['metadata_fields_recovered']} field updates")
                print(f"[recovery] Total files in project: {recovery_stats['files_recovered']}")

            except Exception as e:
                recovery_stats['errors'].append(f"Backup merge failed: {e}")
                print(f"[recovery] Backup merge failed, falling back to rescan: {e}")
                import traceback
                traceback.print_exc()
                backup_data = None  # Fall through to rescan

        # STRATEGY 2: No valid backup - must rescan and apply recovered metadata
        if not backup_data:
            recovery_stats['method'] = 'rescan'

            # Delete the corrupted file
            try:
                project_path.unlink()
                print(f"[recovery] Deleted corrupted file: {project_path}")
            except Exception as e:
                print(f"[recovery] Could not delete corrupted file: {e}")
                recovery_stats['errors'].append(f"Could not delete corrupted file: {e}")

            # Set up the project
            self.mw._current_project_name = project_name
            if hasattr(self.mw, 'projectNameCombo'):
                self.mw.projectNameCombo.setCurrentText(project_name)
            self.mw._project_directory = str(data_directory)
            self.mw.directoryPathEdit.setText(self.mw._project_directory)
            self.mw._master_file_list = []
            self.mw._discovered_files_data = []

            # Store recovered metadata and stats to apply after rescan
            self.mw._pending_recovery_metadata = recovered_metadata
            self.mw._pending_recovery_stats = recovery_stats

            # Trigger rescan
            self.mw._log_status_message(f"Rescanning '{project_name}'...", 3000)
            self.mw._scan_manager.on_project_scan_files()

            # Apply recovered metadata after scan completes (with callback for summary)
            if recovered_metadata:
                QTimer.singleShot(3000, self.apply_recovered_metadata_and_show_summary)
            return  # Summary will be shown after rescan completes

        # Show recovery summary for backup-based recovery
        self.show_recovery_summary(recovery_stats, project_name)

    def get_text_diff(self, backup_path: Path, corrupted_path: Path) -> dict:
        """Get a proper text diff between backup and corrupted file."""
        import difflib

        result = {
            'has_diff': False,
            'diff_lines': [],
            'diff_summary': '',
            'additions': [],      # Lines added in corrupted
            'deletions': [],      # Lines removed from backup
            'backup_content': '',
            'corrupted_content': '',
        }

        try:
            with open(backup_path, 'r', encoding='utf-8', errors='ignore') as f:
                backup_lines = f.readlines()
                result['backup_content'] = ''.join(backup_lines)

            with open(corrupted_path, 'r', encoding='utf-8', errors='ignore') as f:
                corrupted_lines = f.readlines()
                result['corrupted_content'] = ''.join(corrupted_lines)

            # Generate unified diff
            diff = list(difflib.unified_diff(
                backup_lines, corrupted_lines,
                fromfile='backup', tofile='corrupted',
                lineterm=''
            ))

            result['diff_lines'] = diff
            result['has_diff'] = len(diff) > 0

            # Categorize changes
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    result['additions'].append(line[1:].strip())
                elif line.startswith('-') and not line.startswith('---'):
                    result['deletions'].append(line[1:].strip())

            # Create human-readable summary
            result['diff_summary'] = '\n'.join(diff[:50])  # First 50 lines of diff

            print(f"[diff] Found {len(result['additions'])} additions, {len(result['deletions'])} deletions")

        except Exception as e:
            print(f"[diff] Error comparing files: {e}")

        return result

    def ai_smart_repair(self, backup_path: Path, corrupted_path: Path,
                        diff_info: dict, error_msg: str) -> Tuple[Optional[str], str]:
        """
        Use AI to intelligently repair a corrupted JSON file.

        Strategy: Start with backup (valid JSON), have AI identify valid new data
        from the diff, then merge those changes into the backup.

        Returns (repaired_content, explanation) or (None, error_message).
        """
        ai_client = self.get_configured_ai_client()
        if not ai_client:
            return None, "AI not configured"

        # Load the backup as our base (it's valid JSON)
        try:
            backup_data = json.loads(diff_info['backup_content'])
        except:
            return None, "Backup file is not valid JSON"

        # Extract just the additions from the diff (new data in corrupted file)
        additions = diff_info.get('additions', [])
        if not additions:
            return None, "No additions found in diff"

        # Format additions for AI analysis
        additions_text = '\n'.join(additions[:100])  # Limit for tokens

        prompt = f"""Analyze changes made to a JSON project file that caused corruption.

## JSON Parse Error:
{error_msg}

## Lines ADDED to the file (these are the changes since last backup):
```
{additions_text}
```

## Your Task:
These additions contain BOTH:
1. **VALID NEW DATA** - intentional edits like strain="VgatCre", animal_id="25729", etc.
2. **CORRUPTION** - accidental changes that broke the JSON (extra newlines, broken paths, typos)

Analyze each addition and categorize it:
- VALID: Looks like intentional metadata (field values, new entries)
- CORRUPT: Looks like an error (broken syntax, truncated text, random characters)

## Output Format:
Return a JSON object with this structure:
```json
{{
  "analysis": "Brief explanation of what you found",
  "valid_changes": [
    {{"file_pattern": "filename or pattern", "field": "strain", "value": "VgatCre"}},
    {{"file_pattern": "filename", "field": "animal_id", "value": "25729"}}
  ],
  "corrupted_changes": [
    {{"description": "what was corrupted and why"}}
  ]
}}
```

Only include changes you're confident about. For valid_changes, use file_pattern to match which file entry should be updated (can be partial filename)."""

        try:
            self.mw._log_status_message("AI analyzing changes...", 5000)
            response = ai_client.complete(prompt)
            content = response.content.strip()

            # Extract JSON from response
            json_content = None
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.rfind('```')
                json_content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.rfind('```')
                json_content = content[start:end].strip()
            else:
                json_start = content.find('{')
                if json_start >= 0:
                    json_content = content[json_start:]

            if not json_content:
                return None, "AI did not return analysis"

            # Parse AI's analysis
            try:
                ai_analysis = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"[ai-repair] Could not parse AI analysis: {e}")
                return None, f"Could not parse AI analysis: {e}"

            explanation = ai_analysis.get('analysis', 'AI analyzed the changes')
            valid_changes = ai_analysis.get('valid_changes', [])
            corrupted_changes = ai_analysis.get('corrupted_changes', [])

            if not valid_changes:
                explanation += "\n\nNo valid changes identified - restoring from backup only."
                repaired_json = json.dumps(backup_data, indent=2)
                return repaired_json, explanation

            # Record original file count to ensure we don't accidentally change structure
            original_file_count = len(backup_data.get('files', []))

            # Group changes by file pattern to apply consistently to multi-channel files
            changes_by_pattern = {}
            for change in valid_changes:
                file_pattern = change.get('file_pattern', '')
                field = change.get('field', '')
                value = change.get('value', '')
                channel = change.get('channel', None)  # Optional channel specifier

                if not file_pattern or not field:
                    continue

                if file_pattern not in changes_by_pattern:
                    changes_by_pattern[file_pattern] = []
                changes_by_pattern[file_pattern].append({
                    'field': field,
                    'value': value,
                    'channel': channel
                })

            # Apply changes to backup data
            applied_count = 0
            files_updated = set()

            for file_pattern, changes in changes_by_pattern.items():
                # Find ALL matching file entries (handles multi-channel files)
                matching_indices = []
                for i, file_entry in enumerate(backup_data.get('files', [])):
                    file_path = file_entry.get('file_path', '')
                    filename = Path(file_path).name.lower()
                    pattern_lower = file_pattern.lower()

                    # Match by filename (more precise than substring)
                    if filename == pattern_lower or pattern_lower in filename:
                        matching_indices.append(i)

                if not matching_indices:
                    print(f"[ai-repair] No match found for pattern: {file_pattern}")
                    continue

                # Apply changes to matching entries
                for change in changes:
                    field = change['field']
                    value = change['value']
                    channel = change.get('channel')

                    for idx in matching_indices:
                        file_entry = backup_data['files'][idx]
                        entry_channel = file_entry.get('channel', file_entry.get('channels', ''))

                        # If channel specified, only update matching channel
                        if channel and str(channel) not in str(entry_channel):
                            continue

                        # Apply the change
                        file_entry[field] = value
                        applied_count += 1
                        files_updated.add(Path(file_entry.get('file_path', '')).name)
                        print(f"[ai-repair] Applied: {Path(file_entry.get('file_path', '')).name}[{entry_channel}].{field} = '{value}'")

            # Verify we haven't changed the file structure
            final_file_count = len(backup_data.get('files', []))
            if final_file_count != original_file_count:
                print(f"[ai-repair] WARNING: File count changed from {original_file_count} to {final_file_count}")
                return None, f"Repair would change file count from {original_file_count} to {final_file_count} - aborting"

            explanation += f"\n\nApplied {applied_count} changes to {len(files_updated)} files."
            if corrupted_changes:
                explanation += f"\nIgnored {len(corrupted_changes)} corrupted changes."
            explanation += f"\nFile structure preserved ({original_file_count} entries)."

            # Generate repaired JSON
            repaired_json = json.dumps(backup_data, indent=2)

            # Validate it
            try:
                json.loads(repaired_json)
                file_count = len(backup_data.get('files', []))
                print(f"[ai-repair] Produced valid JSON with {file_count} files, {applied_count} changes applied")
                return repaired_json, explanation
            except json.JSONDecodeError as e:
                return None, f"Generated invalid JSON: {e}"

        except Exception as e:
            print(f"[ai-repair] AI repair failed: {e}")
            import traceback
            traceback.print_exc()
            return None, f"AI repair failed: {e}"

    def analyze_corruption_cause(self, project_path: Path, error_msg: str) -> dict:
        """Analyze the corrupted file to determine the likely cause of corruption."""
        result = {
            'type': 'unknown',
            'description': 'Unknown corruption',
            'likely_cause': 'Unable to determine cause',
            'corrupted_section': '',
        }

        try:
            with open(project_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            file_size = len(content)

            # Check for truncation (file ends abruptly)
            content_stripped = content.rstrip()
            if not content_stripped.endswith('}'):
                result['type'] = 'truncation'
                result['description'] = 'File appears truncated (incomplete JSON)'
                result['likely_cause'] = 'Save operation was interrupted (app crash, power loss, or switching projects during autosave)'
                # Get the last 200 chars to show where it cut off
                result['corrupted_section'] = content_stripped[-200:] if len(content_stripped) > 200 else content_stripped

            # Check for invalid control characters
            elif 'invalid control character' in error_msg.lower():
                result['type'] = 'invalid_chars'
                result['description'] = 'File contains invalid control characters'
                result['likely_cause'] = 'Binary data or encoding issue during save (possibly disk error or memory corruption)'
                # Find the problematic area
                import re
                bad_chars = re.findall(r'[\x00-\x1f]', content)
                if bad_chars:
                    result['corrupted_section'] = f"Found {len(bad_chars)} invalid characters"

            # Check for missing brackets/braces
            elif content.count('{') != content.count('}'):
                result['type'] = 'unbalanced_braces'
                open_count = content.count('{')
                close_count = content.count('}')
                result['description'] = f'Unbalanced braces ({{ {open_count} vs }} {close_count})'
                result['likely_cause'] = 'Partial write - save was interrupted before completion'
                result['corrupted_section'] = content[-200:] if len(content) > 200 else content

            # Check for duplicate keys or malformed structure
            elif 'expecting' in error_msg.lower():
                result['type'] = 'malformed_json'
                result['description'] = 'Malformed JSON structure'
                result['likely_cause'] = 'Data serialization error or concurrent write conflict'
                # Extract line number from error if available
                import re
                line_match = re.search(r'line (\d+)', error_msg)
                if line_match:
                    line_num = int(line_match.group(1))
                    lines = content.split('\n')
                    if line_num <= len(lines):
                        start = max(0, line_num - 3)
                        end = min(len(lines), line_num + 2)
                        result['corrupted_section'] = '\n'.join(f"{i+1}: {lines[i]}" for i in range(start, end))

            else:
                # Generic corruption
                result['type'] = 'generic'
                result['description'] = 'JSON parsing error'
                result['likely_cause'] = 'File was modified or corrupted'
                result['corrupted_section'] = content[-200:] if len(content) > 200 else content

        except Exception as e:
            result['likely_cause'] = f'Could not analyze file: {e}'

        return result

    def handle_corrupted_project(self, project_path: Path, project_name: str, error_msg: str):
        """Handle a corrupted project file with a recovery preview dialog."""
        data_directory = project_path.parent
        backup_path = project_path.with_suffix('.physiometrics.bak')
        has_backup = backup_path.exists()

        # First, analyze what can be recovered
        self.mw._log_status_message("Analyzing corrupted file...", 2000)

        # Analyze the corruption cause
        corruption_info = self.analyze_corruption_cause(project_path, error_msg)

        # Get proper text diff if backup exists
        diff_info = None
        if has_backup:
            diff_info = self.get_text_diff(backup_path, project_path)

        # Extract metadata from corrupted file
        recovered_metadata = self.extract_metadata_from_corrupted(project_path)

        # Check if AI is configured
        has_ai = self.get_configured_ai_client() is not None

        # Build recovery preview dialog
        dialog = QDialog(self.mw)
        dialog.setWindowTitle("Project Recovery")
        dialog.setMinimumSize(600, 500)
        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel(f"<b>The project '{project_name}' is corrupted.</b>")
        layout.addWidget(header)

        # Check if backup is valid (can be loaded as JSON)
        backup_valid = False
        backup_file_count = 0
        backup_data = None
        if has_backup:
            try:
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)
                backup_valid = True
                backup_file_count = len(backup_data.get('files', []))
            except:
                backup_valid = False

        # Show corruption diagnosis
        summary_parts = []
        summary_parts.append(f"<b>Corruption type:</b> {corruption_info['description']}")
        summary_parts.append(f"<b>Likely cause:</b> {corruption_info['likely_cause']}")
        summary_parts.append("")

        # Recovery options summary
        if has_backup and backup_valid:
            summary_parts.append(f"+ Valid backup found ({backup_file_count} files)")
        elif has_backup:
            summary_parts.append("! Backup file exists but is also corrupted")
        else:
            summary_parts.append("x No backup file found")

        summary_parts.append(f"+ Found {len(recovered_metadata)} file entries in corrupted data")

        if has_ai:
            if backup_valid:
                summary_parts.append("+ AI configured (will compare backup vs corrupted to find changes)")
            else:
                summary_parts.append("+ AI configured (will analyze file patterns)")
        else:
            summary_parts.append("o AI not configured (optional)")

        # Explain recovery strategy
        summary_parts.append("")
        summary_parts.append("<b>Recovery strategy:</b>")
        if backup_valid and has_ai:
            summary_parts.append("1. Load backup as baseline")
            summary_parts.append("2. Find changes in corrupted file")
            summary_parts.append("3. AI repairs corrupted changes (token-efficient)")
            summary_parts.append("4. Merge with backup data")
        elif backup_valid:
            summary_parts.append("1. Load backup as baseline")
            summary_parts.append("2. Merge any recoverable changes from corrupted file")
        elif has_ai:
            summary_parts.append("1. Rescan directory for files")
            summary_parts.append("2. AI analyzes file paths for metadata patterns")
        else:
            summary_parts.append("1. Rescan directory for files")
            summary_parts.append("2. Apply any metadata found in corrupted file")

        summary_label = QLabel("\n".join(summary_parts))
        layout.addWidget(summary_label)

        # Show REAL text diff between backup and corrupted
        if diff_info and diff_info['has_diff']:
            preview_group = QGroupBox(f"Text Diff: {len(diff_info['additions'])} additions, {len(diff_info['deletions'])} deletions")
        elif has_backup:
            preview_group = QGroupBox("Detected Changes (Backup -> Corrupted)")
        else:
            preview_group = QGroupBox("Recovered Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setMaximumHeight(200)
        preview_text.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")

        preview_lines = []

        # Use actual text diff if available
        if diff_info and diff_info['has_diff']:
            preview_lines.append("Changes detected (- = backup, + = corrupted):")
            preview_lines.append("")
            # Show first 30 diff lines
            for line in diff_info['diff_lines'][:30]:
                preview_lines.append(line.rstrip())
            if len(diff_info['diff_lines']) > 30:
                preview_lines.append(f"... and {len(diff_info['diff_lines']) - 30} more lines")

        elif diff_info and not diff_info['has_diff']:
            preview_lines.append("+ No text differences between backup and corrupted file.")
            preview_lines.append("  This shouldn't happen - the files appear identical.")

        elif has_backup and not diff_info:
            preview_lines.append("Could not compute diff between files.")

        elif recovered_metadata:
            # No backup - just show what we extracted
            preview_lines.append("No backup available for comparison.")
            preview_lines.append(f"Extracted {len(recovered_metadata)} file entries from corrupted data:")
            preview_lines.append("")
            for file_path, meta in list(recovered_metadata.items())[:8]:
                meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if v)
                preview_lines.append(f"- {Path(file_path).name}: {meta_str or '(no metadata)'}")
            if len(recovered_metadata) > 8:
                preview_lines.append(f"... and {len(recovered_metadata) - 8} more files")
        else:
            preview_lines.append("No metadata could be extracted from the corrupted file.")
            preview_lines.append("A fresh scan will discover all files but user-entered data will be lost.")

        preview_text.setPlainText("\n".join(preview_lines))
        preview_layout.addWidget(preview_text)
        layout.addWidget(preview_group)

        # AI enhancement option (if not configured)
        if not has_ai and recovered_metadata:
            ai_note = QLabel(
                "<i>Tip: Configure an AI API key in Settings -> AI Integration to help\n"
                "recover additional metadata from file path patterns.</i>"
            )
            ai_note.setStyleSheet("color: #888;")
            layout.addWidget(ai_note)

        # Buttons
        button_layout = QHBoxLayout()

        # AI Smart Repair button - most powerful option when AI + backup are available
        ai_repair_btn = None
        if has_ai and has_backup and diff_info and diff_info['has_diff']:
            ai_repair_btn = QPushButton("AI Smart Repair")
            ai_repair_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px 16px;")
            ai_repair_btn.setToolTip("AI analyzes the diff to fix corruption while preserving your new data")
            button_layout.addWidget(ai_repair_btn)

        recover_btn = QPushButton("Recover Project")
        recover_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        if has_ai and has_backup:
            recover_btn.setToolTip("Restore from backup and merge any recoverable changes")

        # Only show backup restore option if backup exists AND is valid
        backup_btn = None
        if backup_valid:
            backup_btn = QPushButton("Restore Backup Only")
            backup_btn.setToolTip(f"Restore from backup ({backup_file_count} files, loses ALL recent changes)")
            button_layout.addWidget(backup_btn)

        cancel_btn = QPushButton("Cancel")

        button_layout.addStretch()
        button_layout.addWidget(recover_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        # Button handlers
        result = {'action': None}

        def on_ai_repair():
            result['action'] = 'ai_repair'
            dialog.accept()

        def on_recover():
            result['action'] = 'recover'
            dialog.accept()

        def on_backup():
            result['action'] = 'backup'
            dialog.accept()

        def on_cancel():
            result['action'] = 'cancel'
            dialog.reject()

        if ai_repair_btn:
            ai_repair_btn.clicked.connect(on_ai_repair)
        recover_btn.clicked.connect(on_recover)
        cancel_btn.clicked.connect(on_cancel)
        if backup_btn:
            backup_btn.clicked.connect(on_backup)

        dialog.exec()

        # Handle result
        if result['action'] == 'ai_repair' and diff_info:
            # AI Smart Repair - let AI analyze diff and fix the file
            repaired_json, explanation = self.ai_smart_repair(
                backup_path, project_path, diff_info, error_msg
            )

            if repaired_json:
                # Show what AI found and ask for confirmation
                confirm = QMessageBox.question(
                    self.mw, "AI Repair Complete",
                    f"AI Analysis:\n{explanation}\n\nApply this repair?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if confirm == QMessageBox.StandardButton.Yes:
                    try:
                        # Write the repaired JSON
                        with open(project_path, 'w') as f:
                            f.write(repaired_json)
                        print(f"[ai-repair] Wrote repaired file: {project_path}")

                        # Load it
                        project_data = self.mw.project_manager.load_project(project_path)
                        self.mw._populate_ui_from_project(project_data)
                        self.mw._log_status_message(f"AI repaired project '{project_name}'", 3000)
                    except Exception as e:
                        self.mw._show_error("Repair Failed", f"Could not apply AI repair:\n{e}")
            else:
                self.mw._show_error("AI Repair Failed", f"AI could not repair the file:\n{explanation}")

        elif result['action'] == 'recover':
            self.perform_project_recovery(
                project_path, project_name, data_directory,
                recovered_metadata, use_ai=has_ai
            )

        elif result['action'] == 'backup' and backup_valid:
            # Restore from backup file (we already validated it can be loaded)
            try:
                import shutil
                shutil.copy2(backup_path, project_path)
                print(f"[recovery] Restored from backup: {backup_path}")

                # Load the restored file (should work since we validated it)
                project_data = self.mw.project_manager.load_project(project_path)
                self.mw._populate_ui_from_project(project_data)
                self.mw._log_status_message(f"Restored project '{project_name}' from backup ({backup_file_count} files)", 3000)
            except Exception as e:
                self.mw._show_error("Restore Failed", f"Could not restore from backup:\n{e}")

        # else: Cancel - do nothing
