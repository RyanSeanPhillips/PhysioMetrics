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


class AppBridgeService:
    """
    TCP server that runs inside the PhysioMetrics app, exposing state to MCP tools.

    Handlers are registered with @bridge.handler("command_name") or
    bridge.register("command_name", callable).
    """

    def __init__(self, get_state_fn: Optional[Callable] = None, port: int = 0):
        """
        Args:
            get_state_fn: Callable that returns the current AppState object.
            port: TCP port (0 = use DEFAULT_PORT or env var).
        """
        import os
        self._get_state = get_state_fn
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
