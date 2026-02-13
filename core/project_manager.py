"""
Project Manager - Save/load PhysioMetrics project files.

Project files are saved in the data directory itself for portability.
Recent projects are tracked in AppData for quick access.

Experiment Schema (v2):
    experiment = {
        'id': str,                    # Unique identifier (UUID)
        'name': str,                  # Display name
        'condition': str,             # Experimental condition
        'notes': str,                 # Free-form notes

        # Analysis metadata (pre-fills SaveMetaDialog)
        'metadata': {
            'strain': str,            # Mouse strain (e.g., "VgatCre")
            'virus': str,             # Virus (e.g., "ConFoff-ChR2")
            'location': str,          # Recording location (e.g., "preBotC")
            'stim_type': str,         # Stimulation parameters (e.g., "30Hz10s15ms")
            'power': str,             # Laser power (e.g., "10mW")
            'sex': str,               # Animal sex (M/F/Unknown)
            'experiment_type': str,   # Export strategy (30hz_stim/hargreaves/licking)
        },

        # Analysis tasks (one per file+channel+animal combination)
        'tasks': [
            {
                'id': str,            # Unique task ID
                'file_path': str,     # Relative path to ABF file
                'channel': str,       # Channel to analyze (e.g., "AD0")
                'animal_id': str,     # Animal identifier
                'status': str,        # pending/in_progress/completed/skipped
                'output_folder': str, # Where exports were saved (if completed)
                'last_analyzed': str, # ISO timestamp
            }
        ],

        # Associated note files
        'note_files': [str],          # Paths to Excel/TXT/Word files with notes
    }
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import os


# Project file format version
PROJECT_FORMAT_VERSION = 2


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to be JSON serializable.

    Handles:
    - Path objects -> str
    - datetime objects -> ISO format string
    - sets -> lists
    - nested dicts and lists
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        # Return as-is for basic types (str, int, float, bool, None)
        return obj


class ProjectManager:
    """Manages project save/load operations and recent projects tracking."""

    def __init__(self):
        """Initialize project manager with AppData config."""
        self.config_dir = self._get_app_config_dir()
        self.config_file = self.config_dir / "config.json"
        self._ensure_config_exists()

    def _get_app_config_dir(self) -> Path:
        """Get platform-specific application config directory."""
        if os.name == 'nt':  # Windows
            base = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')))
            config_dir = base / "PhysioMetrics"
        else:  # Linux/Mac
            config_dir = Path.home() / ".physiometrics"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _ensure_config_exists(self):
        """Create config file if it doesn't exist."""
        if not self.config_file.exists():
            self._save_config({"recent_projects": []})

    def _load_config(self) -> Dict:
        """Load app configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[project-manager] Error loading config: {e}")
            return {"recent_projects": []}

    def _save_config(self, config: Dict):
        """Save app configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[project-manager] Error saving config: {e}")

    @staticmethod
    def create_experiment(name: str = "", condition: str = "", notes: str = "",
                          metadata: Dict = None) -> Dict:
        """
        Create a new experiment with default structure.

        Args:
            name: Experiment display name
            condition: Experimental condition description
            notes: Free-form notes
            metadata: Analysis metadata dict (strain, virus, location, etc.)

        Returns:
            Experiment dict with all required fields
        """
        default_metadata = {
            'strain': '',
            'virus': '',
            'location': '',
            'stim_type': '',
            'power': '',
            'sex': '',
            'experiment_type': '30hz_stim',  # Default export strategy
        }
        if metadata:
            default_metadata.update(metadata)

        return {
            'id': str(uuid.uuid4()),
            'name': name,
            'condition': condition,
            'notes': notes,
            'metadata': default_metadata,
            'tasks': [],
            'note_files': [],
        }

    @staticmethod
    def create_analysis_task(file_path: str, channel: str = "", animal_id: str = "") -> Dict:
        """
        Create a new analysis task for a file.

        Args:
            file_path: Path to ABF file (will be stored as relative path)
            channel: Channel to analyze (e.g., "AD0")
            animal_id: Animal identifier

        Returns:
            Task dict with pending status
        """
        return {
            'id': str(uuid.uuid4()),
            'file_path': file_path,
            'channel': channel,
            'animal_id': animal_id,
            'status': 'pending',
            'output_folder': '',
            'last_analyzed': '',
        }

    @staticmethod
    def migrate_experiment_v1_to_v2(old_experiment: Dict) -> Dict:
        """
        Migrate an old experiment format (v1) to the new format (v2).

        V1 experiments only had: name (str), condition (str), notes (str), files (list of paths)
        V2 adds: id, metadata, tasks, note_files
        """
        # Create new experiment with default metadata
        new_exp = ProjectManager.create_experiment(
            name=old_experiment.get('name', ''),
            condition=old_experiment.get('condition', ''),
            notes=old_experiment.get('notes', ''),
        )

        # Convert old file list to tasks
        old_files = old_experiment.get('files', [])
        for file_path in old_files:
            if isinstance(file_path, str):
                task = ProjectManager.create_analysis_task(file_path)
                new_exp['tasks'].append(task)
            elif isinstance(file_path, dict) and 'file_path' in file_path:
                # Already partial dict format
                task = ProjectManager.create_analysis_task(
                    file_path=str(file_path.get('file_path', '')),
                    channel=file_path.get('channel', ''),
                    animal_id=file_path.get('animal_id', ''),
                )
                task['status'] = file_path.get('status', 'pending')
                new_exp['tasks'].append(task)

        return new_exp

    def save_project(self, project_name: str, data_directory: Path,
                     files_data: List[Dict], experiments: List[Dict] = None,
                     notes_directory: str = None, notes_files: List[Dict] = None) -> Path:
        """
        Save project file to data directory.

        Args:
            project_name: Name of the project
            data_directory: Root directory containing the data
            files_data: List of file metadata dicts (with 'file_path', 'protocol', etc.)
            experiments: List of experiment definitions (optional)
            notes_directory: Path to notes folder (optional)
            notes_files: List of notes file metadata dicts (optional)

        Returns:
            Path to saved project file
        """
        if experiments is None:
            experiments = []
        if notes_files is None:
            notes_files = []

        # Create project file path
        project_filename = f"{self._sanitize_filename(project_name)}.physiometrics"
        project_path = data_directory / project_filename

        # Convert file paths to relative paths and ensure JSON serializable
        files_relative = []
        for file_data in files_data:
            # Deep copy and convert all non-serializable types (Path, datetime, set, etc.)
            file_data_copy = _make_json_serializable(file_data)

            if 'file_path' in file_data_copy and file_data_copy['file_path']:
                # Convert absolute path to relative path from data_directory
                abs_path = Path(file_data_copy['file_path'])
                try:
                    rel_path = abs_path.relative_to(data_directory)
                    file_data_copy['file_path'] = str(rel_path)
                except ValueError:
                    # File is outside data_directory, keep absolute
                    file_data_copy['file_path'] = str(abs_path)
            files_relative.append(file_data_copy)

        # Convert experiment task file paths to relative and ensure JSON serializable
        experiments_relative = []
        for exp in experiments:
            # Deep copy and convert all non-serializable types
            exp_copy = _make_json_serializable(exp)

            if 'tasks' in exp_copy:
                for task_copy in exp_copy['tasks']:
                    if 'file_path' in task_copy and task_copy['file_path']:
                        abs_path = Path(task_copy['file_path'])
                        try:
                            rel_path = abs_path.relative_to(data_directory)
                            task_copy['file_path'] = str(rel_path)
                        except ValueError:
                            task_copy['file_path'] = str(abs_path)
            experiments_relative.append(exp_copy)

        # Process notes files - make JSON serializable
        notes_files_serializable = _make_json_serializable(notes_files)

        # Create project data structure
        project_data = {
            "version": PROJECT_FORMAT_VERSION,
            "project_name": project_name,
            "data_directory": ".",  # Relative to project file location
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "file_count": len(files_data),
            "files": files_relative,
            "experiments": experiments_relative,
            "notes_directory": str(notes_directory) if notes_directory else None,
            "notes_files": notes_files_serializable
        }

        # Create backup of existing file before saving
        # This provides a "one undo" safety net - if current file corrupts, .bak has previous version
        import shutil
        backup_path = project_path.with_suffix('.physiometrics.bak')
        if project_path.exists():
            try:
                shutil.copy2(project_path, backup_path)  # copy2 preserves metadata
                print(f"[project-manager] Created backup: {backup_path.name}")
            except Exception as e:
                print(f"[project-manager] Warning: Could not create backup: {e}")

        # Save to JSON atomically (write to temp file, then rename)
        # This prevents corruption if the write is interrupted
        import tempfile
        try:
            # Write to temp file in same directory (ensures same filesystem for atomic rename)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp',
                prefix='physiometrics_',
                dir=data_directory
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(project_data, f, indent=2)

                # Atomic rename (replace existing file)
                temp_path = Path(temp_path)
                temp_path.replace(project_path)
                print(f"[project-manager] Saved project to: {project_path}")
            except:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise
        except Exception as e:
            raise Exception(f"Failed to save project: {e}")

        # Add to recent projects
        self._add_to_recent_projects(project_name, project_path)

        return project_path

    def load_project(self, project_path: Path) -> Dict:
        """
        Load project from file.

        Args:
            project_path: Path to .physiometrics file

        Returns:
            Dictionary with:
                'project_name': str
                'data_directory': Path (absolute)
                'files': List[Dict] with absolute file paths
                'experiments': List[Dict]
                'created': str
                'last_modified': str

        Raises:
            FileNotFoundError: If project file doesn't exist
            Exception: If project file is corrupted
        """
        if not project_path.exists():
            raise FileNotFoundError(f"Project file not found: {project_path}")

        try:
            with open(project_path, 'r') as f:
                project_data = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load project (corrupted file?): {e}")

        # Get data directory (relative to project file)
        project_dir = project_path.parent
        data_directory = project_dir  # Since we save "." in the project file

        # Check version and migrate if necessary
        file_version = project_data.get('version', 1)
        if file_version < PROJECT_FORMAT_VERSION:
            print(f"[project-manager] Migrating project from v{file_version} to v{PROJECT_FORMAT_VERSION}")

        # Convert relative paths back to absolute
        files_absolute = []
        for file_data in project_data.get('files', []):
            file_data_copy = file_data.copy()
            if 'file_path' in file_data_copy:
                rel_path = Path(file_data_copy['file_path'])
                if not rel_path.is_absolute():
                    # Convert relative to absolute
                    abs_path = (data_directory / rel_path).resolve()
                    file_data_copy['file_path'] = abs_path
                else:
                    file_data_copy['file_path'] = Path(file_data_copy['file_path'])
            files_absolute.append(file_data_copy)

        # Process experiments - migrate v1 to v2 if needed and convert paths
        experiments = project_data.get('experiments', [])
        experiments_processed = []
        for exp in experiments:
            # Migrate old format if needed
            if file_version < 2 or 'id' not in exp:
                exp = self.migrate_experiment_v1_to_v2(exp)

            # Convert task file paths to absolute
            if 'tasks' in exp:
                for task in exp['tasks']:
                    if 'file_path' in task and task['file_path']:
                        rel_path = Path(task['file_path'])
                        if not rel_path.is_absolute():
                            abs_path = (data_directory / rel_path).resolve()
                            task['file_path'] = str(abs_path)

            experiments_processed.append(exp)

        # Update last_modified
        project_data['version'] = PROJECT_FORMAT_VERSION
        project_data['last_modified'] = datetime.now().isoformat()
        project_data['data_directory'] = data_directory
        project_data['files'] = files_absolute
        project_data['experiments'] = experiments_processed

        # Include notes data
        project_data['notes_directory'] = project_data.get('notes_directory', None)
        project_data['notes_files'] = project_data.get('notes_files', [])

        # Update recent projects
        self._add_to_recent_projects(project_data['project_name'], project_path)

        print(f"[project-manager] Loaded project: {project_data['project_name']}")
        print(f"[project-manager] Files: {len(files_absolute)}, Experiments: {len(experiments_processed)}")
        if project_data['notes_directory']:
            print(f"[project-manager] Notes directory: {project_data['notes_directory']}")

        return project_data

    def get_recent_projects(self) -> List[Dict]:
        """
        Get list of recent projects.

        Returns:
            List of dicts with 'name', 'path', 'last_opened'
        """
        config = self._load_config()
        return config.get('recent_projects', [])

    def _add_to_recent_projects(self, project_name: str, project_path: Path):
        """Add or update project in recent projects list."""
        config = self._load_config()
        recent = config.get('recent_projects', [])

        # Remove if already exists
        recent = [p for p in recent if p['path'] != str(project_path)]

        # Add to front
        recent.insert(0, {
            'name': project_name,
            'path': str(project_path),
            'last_opened': datetime.now().isoformat()
        })

        # Keep only last 20 projects
        recent = recent[:20]

        config['recent_projects'] = recent
        self._save_config(config)

    def update_recent_project_path(self, old_path: Path, new_path: Path):
        """
        Update path for a recent project (when user relocates it).

        Args:
            old_path: Old project file path
            new_path: New project file path
        """
        config = self._load_config()
        recent = config.get('recent_projects', [])

        for project in recent:
            if project['path'] == str(old_path):
                project['path'] = str(new_path)
                project['last_opened'] = datetime.now().isoformat()
                break

        config['recent_projects'] = recent
        self._save_config(config)
        print(f"[project-manager] Updated project path: {old_path} -> {new_path}")

    def remove_recent_project(self, project_path: Path):
        """Remove project from recent projects list."""
        config = self._load_config()
        recent = config.get('recent_projects', [])
        recent = [p for p in recent if p['path'] != str(project_path)]
        config['recent_projects'] = recent
        self._save_config(config)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert project name to valid filename."""
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("PROJECT MANAGER TEST")
    print("=" * 60)

    pm = ProjectManager()

    # Create test project data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create some fake files
        (data_dir / "file1.abf").touch()
        (data_dir / "file2.abf").touch()

        files_data = [
            {
                'file_path': data_dir / "file1.abf",
                'file_name': "file1.abf",
                'protocol': "Test Protocol 1",
                'file_size_mb': 1.5
            },
            {
                'file_path': data_dir / "file2.abf",
                'file_name': "file2.abf",
                'protocol': "Test Protocol 2",
                'file_size_mb': 2.3
            }
        ]

        # Test save
        print("\n1. Saving project...")
        project_path = pm.save_project("Test Project", data_dir, files_data)
        print(f"   Saved to: {project_path}")

        # Test load
        print("\n2. Loading project...")
        loaded = pm.load_project(project_path)
        print(f"   Project name: {loaded['project_name']}")
        print(f"   Files: {len(loaded['files'])}")
        print(f"   Data directory: {loaded['data_directory']}")

        # Test recent projects
        print("\n3. Recent projects:")
        recent = pm.get_recent_projects()
        for p in recent:
            print(f"   - {p['name']} ({p['path']})")

        print("\n✓ All tests passed!")
