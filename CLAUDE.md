# PhysioMetrics - Development Guide

> **Version**: v1.0.15-beta.3 (Pre-release) | **Next**: v1.0.15 stable
> **License**: MIT | **DOI**: 10.5281/zenodo.17575911

> **Session Skills**:
> - **Start of conversation**: Suggest running `/session-start` to load current priorities
> - **End of conversation**: Suggest running `/session-end` to save accomplishments

## Debugging: Check Logs First

When debugging issues or if the user mentions crashes/errors, check these files:

1. **Startup log** (phase timing for every launch):
   ```
   %APPDATA%\PhysioMetrics\startup.log
   ```
   Timestamped phase markers (imports, UI load, ML models, DB load). Rotating, max 3MB. Use `--console` flag on .exe to also print to a visible console. In-app: Ctrl+Shift+L opens live log viewer dialog.

2. **Persistent debug log** (survives app restart - check this first!):
   ```
   %APPDATA%\PhysioMetrics\debug_log.json
   ```
   Contains last 100 user actions across sessions with timestamps. Read this to understand what the user has been doing.

3. **Local crash reports**:
   ```
   %APPDATA%\PhysioMetrics\crash_reports\*.json
   ```
   Read the most recent crash report(s) to see stack traces and action history.

4. **GitHub issues**:
   - https://github.com/RyanSeanPhillips/PhysioMetrics/issues
   - Look for recent crash reports or feature requests submitted by users

5. **Heartbeat file** (only exists while app is running or after kill):
   ```
   %APPDATA%\PhysioMetrics\.heartbeat
   ```
   Shows last 20 user actions with timestamps - useful for understanding what led to a crash/kill.

See `_internal/docs/CORE/ERROR_REPORTING.md` for full documentation.

## Fixing GitHub Issues — Required Workflow

When fixing bugs or feature requests from GitHub issues:

1. **Grep first** — find all instances of the relevant pattern across the codebase. Don't just fix the first match; there may be duplicates in other dialogs or code paths.
2. **Codeindex impact analysis** — use `get_impact` and `get_context` on the symbols you're changing to understand the full data flow. Ask: "what downstream code depends on this?" This reveals hidden costs (memory, serialization, performance) that grep alone cannot surface.
3. **Reason about WHETHER to change**, not just WHERE — some feature requests sound reasonable but have scaling risks. Add soft warnings for edge cases rather than hard limits or unbounded changes.
4. **Write a codeindex review** — save to `../codeindex/reviews/YYYY-MM-DD_<topic>.md` comparing codeindex vs grep for the session. See the codeindex CLAUDE.md for the template.
5. **Run `ruff check`** on all modified files before finishing.

## Quick Start

**IMPORTANT: Always use the `plethapp` conda environment!**

**Environment location**: `C:\Users\rphil2\AppData\Local\miniforge3\envs\plethapp`
**Direct Python path**: `/c/Users/rphil2/AppData/Local/miniforge3/envs/plethapp/python.exe`

```bash
# Option 1: Activate environment
conda activate plethapp

# Option 2: Use direct path (for scripts/bash)
/c/Users/rphil2/AppData/Local/miniforge3/envs/plethapp/python.exe script.py

# Development
python run_debug.py

# Build executable
python build_executable.py

# Run LabIndex tests
cd ../LabIndex && python test_mvvm.py
```

## Hot Reload (Ctrl+R)

The app supports hot reloading for faster development iteration:

- **Press Ctrl+R** to reload dialog modules and refresh the main plot
- **Styling changes** (colors, titles) appear immediately after Ctrl+R
- **Dialog changes** take effect when you close and reopen the dialog
- **Main window/UI changes** still require app restart

**Modules reloaded**: `core/photometry.py`, `core/peaks.py`, `core/plot_themes.py`, all `dialogs/photometry/*.py`, `plotting/*.py`, `export/export_manager.py`, and more.

See `_internal/docs/CORE/DEVELOPMENT_WORKFLOW.md` for full documentation.

## Current Development Priorities

1. **MVVM Refactoring + Batch Analysis** (Active)
   - Decomposing monolithic `main.py` into MVVM architecture (services, viewmodels, domain models)
   - Projects tab rework (Phases 1-2 complete: file tree, filtering, batch dry run)
   - Batch analysis review workflow with parallel execution and per-row progress
   - **Refactor plan**: `.claude/plans/refactor-plan-v2.md` (6 steps)
     - Step 1 (Startup logging): DONE
     - Step 2 (UI Cleanup Phase 1 — controls to right-click menu): DONE
     - Step 3 (UI Cleanup Phase 2 — minimap nav, compact editing): DONE
     - Step 4A (GMMManager → GMMService + GMMViewModel): DONE
     - Step 4B (ClassifierManager → ClassifierService + ClassifierViewModel): DONE
     - Step 4E (ExportManager → ExportService, refs 126→15): PARTIALLY DONE
     - Step 4C (ScanManager, 95 self.mw refs): TODO
     - Step 4D (RecoveryManager, 37 self.mw refs): TODO
     - Step 4F (ProjectBuilderManager, 168 self.mw refs): TODO
     - Step 4G (Remaining MainWindow methods): TODO
     - Step 5 (HDF5 Migration): TODO
     - Step 6 (SessionManager): TODO
   - Track changes in `_internal/docs/UNRELEASED_CHANGES.md`

2. **Photometry Integration** (Mostly Implemented)
   - Multi-channel fiber photometry import, processing, and visualization
   - Primary dialog: `dialogs/photometry_import_dialog_v3.py` + `dialogs/photometry/` submodule
   - See: `_internal/docs/PLANNING/PHOTOMETRY_*.md`

3. **Next Release** (v1.0.15)
   - After batch analysis workflow is stable
   - Track changes in `_internal/docs/UNRELEASED_CHANGES.md`

4. **Claude Code Lab Assistant — MCP Integration** (Future)
   - Replace current chatbot with Claude Code terminal (popup or embedded)
   - MCP server bridges Claude Code ↔ running app (load files, inspect channels, trigger processing)
   - Two-phase workflow: (1) Interactive data organization with user, (2) Autonomous batch processing with flagging
   - Claude uses built-in tools for file discovery/notes reading; MCP tools for app control
   - See: `_internal/docs/PLANNING/CLAUDE_CODE_NOTEBOOK_INTEGRATION_PLAN.md`

## Project Structure

```
physiometrics/
├── main.py                    # Main application (~10K lines)
├── core/                      # Signal processing, state, ML, I/O
│   ├── state.py              # AppState dataclass
│   ├── peaks.py              # Peak detection algorithms
│   ├── metrics.py            # 60+ breathing metrics
│   ├── channel_manager.py    # Multi-channel display config
│   ├── event_markers.py      # Event marker system
│   ├── event_types.py        # Event type registry
│   ├── ml_training.py        # RF, XGBoost, MLP training
│   ├── ml_prediction.py      # Model inference
│   ├── photometry.py         # Fiber photometry processing
│   ├── parallel_utils.py     # Parallel processing
│   └── io/                   # File loaders (ABF, SMRX, EDF, MAT)
├── dialogs/                   # 24 dialog windows
│   ├── advanced_peak_editor_dialog.py  # 3D UMAP visualization
│   ├── gmm_clustering_dialog.py
│   ├── peak_detection_dialog.py
│   └── photometry/           # Photometry subsystem (5 files)
├── editing/                   # Peak/event editing modes
│   ├── editing_modes.py
│   └── event_marking_mode.py
├── plotting/                  # Visualization
│   ├── plot_manager.py       # Main plot orchestration
│   └── pyqtgraph_backend.py  # Fast plotting (experimental)
├── export/                    # Data export
│   ├── export_manager.py     # Main export (~5K lines)
│   └── strategies/           # Export strategies
├── consolidation/             # Multi-file consolidation
└── ui/                        # Qt Designer .ui files
    └── pleth_app_layout_05_horizontal.ui  # Current layout
```

## Key Features

- **Peak Detection**: Multi-level fallback algorithms with auto-thresholding
- **ML Classification**: RF, XGBoost, MLP for breath type (eupnea/sniffing/sigh)
- **GMM Clustering**: Unsupervised classification fallback
- **Channel Manager**: Multi-channel display with type detection
- **Event Markers**: Drag-and-drop region marking system
- **Parallel Processing**: 4-8x speedup for detection/export
- **File Formats**: ABF, SMRX (Spike2), EDF, MAT, photometry CSV

## Documentation

See `_internal/docs/INDEX.md` for complete documentation map.

**Session management:**
- `CURRENT_STATUS.md` - Current priorities, recent work (~50 lines)
- Use `/session-start` and `/session-end` skills

**Essential docs (in `_internal/docs/CORE/`):**
- `ARCHITECTURE.md` - Software design and code organization
- `ALGORITHMS.md` - Signal processing algorithms
- `FEATURE_BACKLOG.md` - Todo list and roadmap
- `UNRELEASED_CHANGES.md` - Changelog staging

**Current work:**
- See `CURRENT_STATUS.md` for latest session details
- `PLANNING/PHOTOMETRY_*.md` - Photometry integration (ACTIVE)

## File Organization

| Location | Purpose |
|----------|---------|
| Root `.md` files | Public docs (README, paper.md, BUILD_INSTRUCTIONS) |
| `_internal/docs/` | Development docs (NOT on GitHub) |
| `_internal/scripts/` | Development utilities (NOT on GitHub) |

## Standalone Tools

### Metadata Extraction Browser
**Location**: `_internal/scripts/metadata_extraction_app.py`

A standalone PyQt6 application for exploring file systems and extracting metadata from research documents. Features include:

- **Interactive Graph Visualization**: Tree, radial, spring, and circular layouts
- **File System Browser**: Navigate folders with file categorization
- **Live Preview Overlay**: Real-time visualization during scanning
- **Search & Highlighting**: Find files with visual highlighting in graph
- **Right-Click Context Menu**: Open files, folders, copy paths

```bash
# Launch the app
python _internal/scripts/metadata_extraction_app.py

# Or use the batch launcher (Windows)
_internal/scripts/launch_metadata_browser.bat
```

See `_internal/scripts/METADATA_EXTRACTION_APP.md` for full documentation.

**Planned Features**: LLM-assisted metadata extraction, cross-referencing with data folders, potential standalone GitHub release.

## Development Workflow

### Tracking Changes
Add entries to `_internal/docs/UNRELEASED_CHANGES.md` as you work:
```markdown
## Session: YYYY-MM-DD
### Bug Fixes / New Features / Performance
- **Description**: What changed
- **Location**: `file.py:line_number`
```

### When Releasing
1. Copy UNRELEASED_CHANGES.md to CHANGELOG.md
2. Update version in `version_info.py`
3. Clear UNRELEASED_CHANGES.md
4. Build and test executable

## Architecture: MVVM Migration (In Progress)

**All new features MUST follow MVVM architecture.** Do NOT add code directly to `main.py` or create managers that reference `self.mw` (MainWindow).

### MVVM Pattern (Required for New Code)

Follow the event marker system as the reference implementation:

```
Domain Model (pure Python, no Qt)     → core/domain/*/models.py
Service (business logic, no Qt)       → core/services/*_service.py
ViewModel (QObject + signals)         → viewmodels/*_viewmodel.py
View (Qt widgets, binds to ViewModel) → views/*/ or dialogs/
```

**Rules:**
1. **Domain models** — pure dataclasses in `core/domain/`. No PyQt6 imports. Serializable with `to_dict()`/`from_dict()`
2. **Services** — pure Python classes in `core/services/`. No PyQt6 imports. Contain all business logic, algorithms, I/O coordination
3. **ViewModels** — `QObject` subclasses in `viewmodels/`. Expose `pyqtSignal`s for data changes. Wrap services. No direct widget manipulation
4. **Views/Dialogs** — receive ViewModels, bind to signals. No business logic. No `self.mw` references
5. **No `self.mw` pattern** — managers that reference MainWindow directly are legacy. New code must use dependency injection (callbacks, signals, or interfaces)

### Reference Implementation

```
core/domain/events/models.py          ← EventMarker dataclass (pure Python)
core/services/event_marker_service.py  ← EventMarkerService (undo/redo, CRUD)
viewmodels/event_marker_viewmodel.py   ← EventMarkerViewModel (QObject + signals)
views/events/plot_integration.py       ← View layer (binds ViewModel to plot)
views/events/marker_renderer.py        ← Rendering logic
views/events/context_menu.py           ← Context menu handling
```

### Legacy Architecture (Being Migrated)

- **`main.py`** (~9K lines) — monolithic MainWindow, being gradually decomposed
- **`core/state.py`** — `AppState` god-object with 100+ fields. `plotting_backend` now defaults to `'pyqtgraph'` (matplotlib backend is disabled). New features should extract relevant state into smaller dataclasses (e.g., `FilterState`, `NavigationState`)
- **Extracted managers** (`core/*_manager.py`) — use `self.mw` pattern. When touching these, prefer decoupling toward MVVM over adding more `self.mw` references
- **Already extracted**: GMMService, ClassifierService, StimService, AnalysisService, NavigationService + ViewModels. EditingModes decoupled via dependency injection (callbacks, no self.mw)
- **Remaining**: ScanManager (95 refs), RecoveryManager (37 refs), ExportManager (15 refs remaining), ProjectBuilderManager (168 refs)
- **Dual peak arrays**: `all_peaks_by_sweep` (master) + `peaks_by_sweep` (filtered)
- **ML pipeline**: Model 1 (breath vs noise) → Model 2 (breath type)
- **Editing undo**: `EditingModes._undo_stack` — Ctrl+Z for peak edits, sigh, omit, merge, move point
- **Bottom toolbar**: Editing icons + nav controls on minimap bar; display/Y2/view in right-click menu; exports in status bar

### Startup Performance

- Heavy imports must be **lazy-loaded** (imported inside functions, not at module top level)
- Dialogs like `AdvancedPeakEditorDialog`, `SpectralAnalysisDialog`, `PhotometryImportDialog` are lazy-imported where used
- `ExportManager` uses a lazy property (`@property` on MainWindow)
- ML model loading is deferred via `QTimer.singleShot(200, ...)`
- Never add heavy imports (scipy, sklearn, matplotlib, pyqtgraph.opengl) to module-level in `main.py`

## Testing

```bash
# Manual testing
python run_debug.py

# Test built executable on clean machine
# No automated tests yet (pytest planned)
```

## Workflow Skills

Custom skills in `.claude/skills/` for common workflows:

| Skill | When to Use |
|-------|-------------|
| `/session-start` | **Start of conversation** - Load current priorities (~300 tokens) |
| `/session-end` | **End of conversation** - Update CURRENT_STATUS.md with accomplishments |
| `/release` | Preparing a release - changelog, version bump, git commands |
| `/doc-update` | Check if documentation is in sync with code |
| `/feature-status` | Cross-reference FEATURE_BACKLOG.md with codebase |
| `/photometry` | Photometry development status and files |

**Suggested workflow**:
1. Start session → Run `/session-start`
2. Work on features...
3. End session → Run `/session-end`

## External Resources

- [PyQt6 Docs](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [SciPy Signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [scikit-learn](https://scikit-learn.org/stable/)

---

**Full documentation index**: `_internal/docs/INDEX.md`
