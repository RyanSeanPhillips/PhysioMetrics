"""
App Bridge Service — exposes running app state to external MCP tools.

Runs a lightweight JSON-over-TCP server inside the app process, giving
MCP tools direct access to AppState, managers, and the loaded file.

Architecture:
    app_mcp.py (stdio MCP) ←→ TCP ←→ AppBridgeService (inside app) ←→ AppState

The bridge runs on a background thread with a simple request/response
protocol: newline-delimited JSON messages on localhost.

Usage (in main.py or app startup):
    bridge = AppBridgeService(main_window)
    bridge.start()  # Starts TCP listener on localhost:PORT
    # ... app runs ...
    bridge.stop()
"""

import json
import socket
import threading
from typing import Any, Dict, Optional, Callable

# Default port — can be overridden via environment variable
DEFAULT_PORT = 19847
_ENV_PORT_KEY = "PHYSIOMETRICS_BRIDGE_PORT"


class _MainThreadDispatcher:
    """Helper to dispatch callables to the main Qt thread via signal."""
    _instance = None

    @classmethod
    def get(cls, parent=None):
        if cls._instance is None and parent is not None:
            from PyQt6.QtCore import QObject, pyqtSignal
            class _Dispatcher(QObject):
                run_on_main = pyqtSignal(object)
                def __init__(self, p):
                    super().__init__(p)
                    self.run_on_main.connect(self._execute)
                def _execute(self, fn):
                    fn()
            cls._instance = _Dispatcher(parent)
        return cls._instance


class AppBridgeService:
    """
    TCP server that runs inside the PhysioMetrics app, exposing state to MCP tools.

    Handlers are registered with @bridge.handler("command_name") or
    bridge.register("command_name", callable).
    """

    def __init__(self, get_state_fn: Optional[Callable] = None, port: int = 0,
                 main_window=None):
        """
        Args:
            get_state_fn: Callable that returns the current AppState object.
            port: TCP port (0 = use DEFAULT_PORT or env var).
            main_window: Reference to the MainWindow for screenshot capture.
        """
        import os
        self._get_state = get_state_fn
        self._main_window = main_window
        self._port = port or int(os.environ.get(_ENV_PORT_KEY, DEFAULT_PORT))
        self._handlers: Dict[str, Callable] = {}
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Register built-in handlers
        self._register_builtins()

    def register(self, name: str, handler: Callable):
        """Register a command handler. handler(args: dict) -> dict."""
        self._handlers[name] = handler

    def handler(self, name: str):
        """Decorator to register a command handler."""
        def decorator(fn):
            self._handlers[name] = fn
            return fn
        return decorator

    def start(self):
        """Start the TCP listener on a background thread."""
        if self._running:
            return

        # Initialize the main-thread dispatcher (must happen on main thread)
        if self._main_window is not None:
            _MainThreadDispatcher.get(self._main_window)

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # Allow periodic check of _running flag
        self._server_socket.bind(("127.0.0.1", self._port))
        self._server_socket.listen(2)
        self._running = True

        # Write port to a discoverable location
        self._write_port_file()

        self._thread = threading.Thread(target=self._serve_loop, daemon=True, name="AppBridge")
        self._thread.start()

    def stop(self):
        """Stop the TCP listener."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=3)
        self._cleanup_port_file()

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # TCP server
    # ------------------------------------------------------------------

    def _serve_loop(self):
        """Accept connections and handle requests."""
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            # Handle each connection in a short-lived thread
            threading.Thread(
                target=self._handle_connection, args=(conn,), daemon=True,
            ).start()

    def _handle_connection(self, conn: socket.socket):
        """Handle a single client connection (one request, one response)."""
        try:
            conn.settimeout(10.0)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if not data:
                return

            request = json.loads(data.decode("utf-8").strip())
            command = request.get("command", "")
            args = request.get("args", {})

            handler = self._handlers.get(command)
            if handler:
                try:
                    result = handler(args)
                    response = {"status": "ok", "result": result}
                except Exception as e:
                    response = {"status": "error", "error": str(e)}
            else:
                response = {
                    "status": "error",
                    "error": f"Unknown command: {command}",
                    "available": sorted(self._handlers.keys()),
                }

            conn.sendall((json.dumps(response, default=str) + "\n").encode("utf-8"))

        except Exception as e:
            try:
                conn.sendall((json.dumps({"status": "error", "error": str(e)}) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Port file (so MCP server can discover the port)
    # ------------------------------------------------------------------

    def _write_port_file(self):
        """Write port number to %APPDATA%/PhysioMetrics/.bridge_port."""
        import os
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            from pathlib import Path
            port_file = Path(appdata) / "PhysioMetrics" / ".bridge_port"
            port_file.parent.mkdir(parents=True, exist_ok=True)
            port_file.write_text(str(self._port))

    def _cleanup_port_file(self):
        """Remove the port file on shutdown."""
        import os
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            from pathlib import Path
            port_file = Path(appdata) / "PhysioMetrics" / ".bridge_port"
            try:
                port_file.unlink(missing_ok=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Built-in handlers
    # ------------------------------------------------------------------

    def _register_builtins(self):
        """Register the default set of app bridge commands."""

        @self.handler("ping")
        def ping(args):
            return {"alive": True, "port": self._port}

        @self.handler("get_state")
        def get_state(args):
            st = self._get_state() if self._get_state else None
            if st is None:
                return {"loaded": False}
            return {
                "loaded": True,
                "file_path": getattr(st, "current_file_path", ""),
                "file_name": getattr(st, "file_name", ""),
                "sweep_count": getattr(st, "sweep_count", 0),
                "current_sweep": getattr(st, "current_sweep_index", 0),
                "analyze_chan": getattr(st, "analyze_chan", ""),
                "sample_rate": getattr(st, "sample_rate", 0),
                "is_photometry": getattr(st, "is_photometry", False),
                "channel_count": len(getattr(st, "channel_names", [])),
                "channel_names": list(getattr(st, "channel_names", [])),
            }

        @self.handler("get_selection")
        def get_selection(args):
            """Get currently selected files in the project table."""
            st = self._get_state() if self._get_state else None
            if st is None:
                return {"selected": []}
            # The selection would come from the project table's selection model
            # For now, return the current file
            return {
                "current_file": getattr(st, "current_file_path", ""),
                "selected": [],  # Populated when project table selection is wired
            }

        @self.handler("navigate")
        def navigate(args):
            """Navigate to a specific sweep or time position."""
            # This requires calling back to the main thread via signal
            # For now, return the requested navigation as acknowledgment
            return {
                "acknowledged": True,
                "file": args.get("file_path", ""),
                "sweep": args.get("sweep"),
                "time": args.get("time"),
                "note": "Navigation requires MainWindow integration — wire via QMetaObject.invokeMethod",
            }

        @self.handler("list_commands")
        def list_commands(args):
            return {"commands": sorted(self._handlers.keys())}

        @self.handler("switch_tab")
        def switch_tab(args):
            """Switch to a named tab in the main window.

            Args:
                tab: "project"|"analysis"|"curation"|"data_files"|"consolidate"
            """
            tab_name = args.get("tab", "")
            if not self._main_window:
                return {"error": "No main window"}

            result_holder = {}
            event = threading.Event()

            def _switch():
                try:
                    mw = self._main_window
                    # Main tabs (leftColumnTabs)
                    TAB_MAP = {
                        "data_files": "tableContainer",
                        "consolidate": "consolidationContainer",
                    }
                    # Top-level tool box pages
                    TOP_TAB_MAP = {
                        "project": "ProjectBuilderTab",
                        "analysis": "Analysis",
                        "curation": "Curation",
                    }

                    widget_name = TAB_MAP.get(tab_name)
                    if widget_name and hasattr(mw, 'leftColumnTabs'):
                        tabs = mw.leftColumnTabs
                        for i in range(tabs.count()):
                            w = tabs.widget(i)
                            if w and w.objectName() == widget_name:
                                tabs.setCurrentIndex(i)
                                result_holder["switched"] = tab_name
                                return

                    top_name = TOP_TAB_MAP.get(tab_name)
                    # Try the top-level Tabs QTabWidget (named 'Tabs' in .ui)
                    top_tabs = getattr(mw, 'Tabs', None) or getattr(mw, 'mainToolBox', None)
                    if top_name and top_tabs:
                        for i in range(top_tabs.count()):
                            w = top_tabs.widget(i)
                            if w and w.objectName() == top_name:
                                top_tabs.setCurrentIndex(i)
                                result_holder["switched"] = tab_name
                                return

                    result_holder["error"] = f"Unknown tab: {tab_name}"
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            if dispatcher is None:
                raise RuntimeError("No main thread dispatcher")
            dispatcher.run_on_main.emit(_switch)
            event.wait(timeout=5)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("click_button")
        def click_button(args):
            """Click a named button in the UI.

            Args:
                button: Button object name (e.g. "scanFilesButton", "saveProjectButton")
            """
            button_name = args.get("button", "")
            if not self._main_window:
                return {"error": "No main window"}

            result_holder = {}
            event = threading.Event()

            def _click():
                try:
                    mw = self._main_window
                    btn = getattr(mw, button_name, None)
                    if btn is None:
                        result_holder["error"] = f"Button not found: {button_name}"
                        return
                    btn.click()
                    result_holder["clicked"] = button_name
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            dispatcher.run_on_main.emit(_click)
            event.wait(timeout=5)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("open_dialog")
        def open_dialog(args):
            """Open a dialog by name.

            Args:
                dialog: "peak_detection"|"export"|"analysis_options"|"spectral"|"advanced_peak_editor"|"help"
            """
            dialog_name = args.get("dialog", "")
            if not self._main_window:
                return {"error": "No main window"}

            DIALOG_MAP = {
                "peak_detection": "_show_peak_detection_dialog",
                "analysis_options": "_show_analysis_options_dialog",
                "help": "_show_help_dialog",
            }

            method_name = DIALOG_MAP.get(dialog_name)
            if not method_name:
                return {"error": f"Unknown dialog: {dialog_name}. Available: {list(DIALOG_MAP.keys())}"}

            result_holder = {}
            event = threading.Event()

            def _open():
                try:
                    mw = self._main_window
                    method = getattr(mw, method_name, None)
                    if method:
                        method()
                        result_holder["opened"] = dialog_name
                    else:
                        result_holder["error"] = f"Method not found: {method_name}"
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            dispatcher.run_on_main.emit(_open)
            event.wait(timeout=5)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("close_dialog")
        def close_dialog(args):
            """Close the active modal dialog."""
            if not self._main_window:
                return {"error": "No main window"}

            result_holder = {}
            event = threading.Event()

            def _close():
                try:
                    from PyQt6.QtWidgets import QApplication
                    active = QApplication.activeModalWidget()
                    if active:
                        active.close()
                        result_holder["closed"] = True
                    else:
                        result_holder["closed"] = False
                        result_holder["note"] = "No modal dialog open"
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            dispatcher.run_on_main.emit(_close)
            event.wait(timeout=5)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("get_ui_state")
        def get_ui_state(args):
            """Get current UI state: active tab, open dialogs, selected rows."""
            if not self._main_window:
                return {"error": "No main window"}

            result_holder = {}
            event = threading.Event()

            def _get():
                try:
                    from PyQt6.QtWidgets import QApplication
                    mw = self._main_window

                    # Active tab
                    active_tab = ""
                    if hasattr(mw, 'leftColumnTabs'):
                        idx = mw.leftColumnTabs.currentIndex()
                        w = mw.leftColumnTabs.widget(idx)
                        if w:
                            active_tab = w.objectName()

                    # Open dialogs
                    dialogs = []
                    for tlw in QApplication.topLevelWidgets():
                        if tlw is not mw and tlw.isVisible():
                            dialogs.append({
                                "title": tlw.windowTitle(),
                                "class": type(tlw).__name__,
                            })

                    # Selected rows in file table
                    selected_rows = []
                    if hasattr(mw, 'discoveredFilesTable'):
                        sel_model = mw.discoveredFilesTable.selectionModel()
                        if sel_model:
                            for idx in sel_model.selectedRows():
                                selected_rows.append(idx.row())

                    result_holder.update({
                        "active_tab": active_tab,
                        "open_dialogs": dialogs,
                        "selected_rows": selected_rows,
                    })
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            dispatcher.run_on_main.emit(_get)
            event.wait(timeout=5)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("open_source_review")
        def open_source_review(args):
            """Open the source review dialog for a table row.

            Args:
                row: Row index in the data files table.
            """
            row = args.get("row", 0)
            result_holder = {}
            event = threading.Event()

            def _do():
                try:
                    mw = self._main_window
                    if hasattr(mw, '_open_source_review'):
                        mw._open_source_review(row)
                        result_holder["opened"] = True
                    else:
                        result_holder["error"] = "Method not available"
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            if dispatcher is None:
                raise RuntimeError("No main thread dispatcher")
            dispatcher.run_on_main.emit(_do)
            event.wait(timeout=10)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("load_project")
        def load_project(args):
            """Load a project by selecting it in the project combo.

            Args:
                index: Combo index to select (1 = most recent project).
                       If not provided, selects index 1 (most recent).
            """
            idx = args.get("index", 1)
            result_holder = {}
            event = threading.Event()

            def _do_load():
                try:
                    mw = self._main_window
                    combo = getattr(mw, 'projectNameCombo', None)
                    if combo is None:
                        result_holder["error"] = "projectNameCombo not found"
                        return
                    if idx >= combo.count():
                        result_holder["error"] = f"Index {idx} out of range (max {combo.count() - 1})"
                        return
                    combo.setCurrentIndex(idx)
                    result_holder["loaded"] = combo.currentText()
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            if dispatcher is None:
                raise RuntimeError("No main thread dispatcher")
            dispatcher.run_on_main.emit(_do_load)
            event.wait(timeout=10)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("reload")
        def reload(args):
            """Trigger hot reload (same as Ctrl+R) from MCP."""
            result_holder = {}
            event = threading.Event()

            def _do_reload():
                try:
                    mw = self._main_window
                    if hasattr(mw, '_on_hot_reload'):
                        mw._on_hot_reload()
                        result_holder["reloaded"] = True
                    else:
                        result_holder["error"] = "Hot reload not available"
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            if dispatcher is None:
                raise RuntimeError("No main thread dispatcher")
            dispatcher.run_on_main.emit(_do_reload)
            event.wait(timeout=10)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("refresh_project")
        def refresh_project(args):
            """Reload the experiment table from the SQLite DB.

            Use after MCP tools modify experiment metadata so the app
            table reflects the latest data without a full restart.
            """
            result_holder = {}
            event = threading.Event()

            def _do_refresh():
                try:
                    mw = self._main_window
                    if hasattr(mw, '_load_experiments_from_db'):
                        mw._load_experiments_from_db()
                        result_holder["refreshed"] = True
                        # Get new count
                        if hasattr(mw, '_master_file_list'):
                            result_holder["experiment_count"] = len(mw._master_file_list)
                    else:
                        result_holder["error"] = "_load_experiments_from_db not found"
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            dispatcher = _MainThreadDispatcher.get(self._main_window)
            if dispatcher is None:
                raise RuntimeError("No main thread dispatcher")
            dispatcher.run_on_main.emit(_do_refresh)
            event.wait(timeout=15)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])
            return result_holder

        @self.handler("screenshot")
        def screenshot(args):
            """Capture a screenshot of the app window or a specific widget.

            Args:
                target: "main" (default), "project", "plot", or "full" (entire window)
                output_path: Optional path to save PNG (auto-generates if omitted)
            """
            if self._main_window is None:
                return {"error": "No main window reference available"}

            target = args.get("target", "main")
            output_path = args.get("output_path", "")

            # Must run on the main Qt thread — use a thread-safe result holder
            result_holder = {}
            event = threading.Event()

            def _capture():
                try:
                    from PyQt6.QtWidgets import QApplication
                    import tempfile, os

                    mw = self._main_window
                    widget = None

                    if target == "dialog":
                        # Capture the active modal/modeless dialog
                        active = QApplication.activeModalWidget() or QApplication.activeWindow()
                        if active and active is not mw:
                            widget = active
                        else:
                            # Check for any visible top-level dialog
                            for tlw in QApplication.topLevelWidgets():
                                if tlw is not mw and tlw.isVisible():
                                    widget = tlw
                                    break
                        if widget is None:
                            result_holder["error"] = "No dialog is currently open"
                            return
                    elif target == "project":
                        # Try to find the project builder tab/widget
                        for attr in ('project_builder_manager', '_project_builder'):
                            mgr = getattr(mw, attr, None)
                            if mgr:
                                tab_widget = getattr(mgr, 'tab_widget', None) or getattr(mgr, '_tab', None)
                                if tab_widget:
                                    widget = tab_widget
                                    break
                        if widget is None:
                            widget = mw
                    elif target == "plot":
                        # Try to find the plot area
                        for attr in ('plot_widget', 'graphics_view', '_plot_manager'):
                            w = getattr(mw, attr, None)
                            if w:
                                widget = getattr(w, 'widget', w) if hasattr(w, 'widget') else w
                                break
                        if widget is None:
                            widget = mw
                    else:
                        widget = mw

                    # Capture
                    pixmap = widget.grab()

                    # Save
                    if not output_path:
                        tmp_dir = os.path.join(tempfile.gettempdir(), "physiometrics_screenshots")
                        os.makedirs(tmp_dir, exist_ok=True)
                        from datetime import datetime
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(tmp_dir, f"screenshot_{target}_{ts}.png")
                    else:
                        save_path = output_path

                    pixmap.save(save_path, "PNG")
                    result_holder["path"] = save_path
                    result_holder["width"] = pixmap.width()
                    result_holder["height"] = pixmap.height()
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    event.set()

            # Dispatch to main Qt thread via signal (thread-safe)
            dispatcher = _MainThreadDispatcher.get(self._main_window)
            if dispatcher is None:
                raise RuntimeError("No main thread dispatcher available")
            dispatcher.run_on_main.emit(_capture)

            # Wait for result from main thread
            event.wait(timeout=10)
            if "error" in result_holder:
                raise RuntimeError(result_holder["error"])

            return {
                "screenshot_path": result_holder.get("path", ""),
                "width": result_holder.get("width", 0),
                "height": result_holder.get("height", 0),
                "target": target,
            }


# === Client helper (used by app_mcp.py) ===

def bridge_call(command: str, args: Optional[Dict] = None, port: int = 0, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Send a command to the running app bridge and return the response.

    Args:
        command: Command name (e.g., "get_state").
        args: Optional arguments dict.
        port: Bridge port (0 = auto-discover from port file).
        timeout: Socket timeout in seconds.

    Returns:
        Response dict with "status" and "result" or "error".
    """
    import os
    if port == 0:
        # Try to read port from file
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            from pathlib import Path
            port_file = Path(appdata) / "PhysioMetrics" / ".bridge_port"
            if port_file.exists():
                port = int(port_file.read_text().strip())

    if port == 0:
        port = DEFAULT_PORT

    request = json.dumps({"command": command, "args": args or {}}) + "\n"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(("127.0.0.1", port))
        sock.sendall(request.encode("utf-8"))

        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        return json.loads(data.decode("utf-8").strip())

    except ConnectionRefusedError:
        return {"status": "error", "error": "App is not running (connection refused). Start PhysioMetrics first."}
    except socket.timeout:
        return {"status": "error", "error": "App bridge timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        sock.close()
