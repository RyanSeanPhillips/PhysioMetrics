"""
Folder Tree Service — builds a tree structure from DB file paths.

Pure Python, no Qt dependencies. Used by FolderTreeViewModel.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import os
import sys


@dataclass
class FolderNode:
    """A node in the folder tree."""
    name: str
    full_path: str
    children: Dict[str, 'FolderNode'] = field(default_factory=dict)
    file_count: int = 0          # experiments in this exact folder
    total_count: int = 0         # experiments in this folder + descendants
    is_indexed: bool = True      # False for filesystem-discovered unindexed folders

    def sorted_children(self) -> List['FolderNode']:
        """Return children sorted alphabetically."""
        return sorted(self.children.values(), key=lambda n: n.name.lower())


# Cache for drive letter → UNC mappings (populated once)
_drive_map_cache: Optional[Dict[str, str]] = None


def _get_drive_mappings() -> Dict[str, str]:
    """Get mapped network drives: {'Z:': '\\\\helens\\active\\baertsch_n', ...}

    Windows only. Returns empty dict on other platforms or errors.
    """
    global _drive_map_cache
    if _drive_map_cache is not None:
        return _drive_map_cache

    _drive_map_cache = {}
    if sys.platform != 'win32':
        return _drive_map_cache

    try:
        import subprocess
        result = subprocess.run(
            ['net', 'use'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            # Lines like: OK  Z:  \\helens\active\baertsch_n  ...
            # or:         Z:  \\helens\active\baertsch_n  ...
            for i, part in enumerate(parts):
                if len(part) == 2 and part[1] == ':' and part[0].isalpha():
                    # Next part should be the UNC path
                    if i + 1 < len(parts) and parts[i + 1].startswith('\\\\'):
                        _drive_map_cache[part.upper()] = parts[i + 1]
                    break
    except Exception:
        pass
    return _drive_map_cache


def normalize_path(path: str) -> str:
    """Normalize a path, resolving mapped drive letters to UNC paths.

    E.g., Z:\\DATA\\file.abf → \\\\helens\\active\\baertsch_n\\DATA\\file.abf
    """
    norm = os.path.normpath(path)
    if sys.platform == 'win32':
        mappings = _get_drive_mappings()
        # Check if path starts with a mapped drive letter
        if len(norm) >= 2 and norm[1] == ':':
            drive = norm[:2].upper()
            if drive in mappings:
                unc_root = mappings[drive]
                rest = norm[2:]  # everything after "Z:"
                norm = unc_root + rest
    return norm


class FolderTreeService:
    """Builds a folder tree from a list of file paths (from the DB)."""

    def find_common_root(self, file_paths: List[str]) -> str:
        """Find the deepest common ancestor directory of all paths.

        Goes up 2 levels from the deepest common to show more context.
        Returns empty string if no paths or no common root.
        """
        if not file_paths:
            return ""

        # Get directory parts for each path (already normalized)
        parts_list = []
        for fp in file_paths:
            d = os.path.dirname(fp)
            parts_list.append(d.split(os.sep))

        if not parts_list:
            return ""

        # Find common prefix
        common = list(parts_list[0])
        for parts in parts_list[1:]:
            new_common = []
            for a, b in zip(common, parts):
                if a.lower() == b.lower():
                    new_common.append(a)
                else:
                    break
            common = new_common
            if not common:
                return ""

        # Count meaningful parts (skip empty strings from UNC \\server\share)
        meaningful = [p for p in common if p]
        if len(meaningful) < 2:
            return ""

        # Go up 2 levels for more context (but keep UNC prefix intact)
        if len(meaningful) > 3:
            common = common[:-2]

        return os.sep.join(common)

    def build_tree(self, file_paths: List[str], root_path: str = "") -> Optional[FolderNode]:
        """Build a folder tree from experiment file paths.

        Args:
            file_paths: List of absolute file paths from the DB.
            root_path: Root to use as tree base. If empty, auto-detected.

        Returns:
            Root FolderNode, or None if no paths.
        """
        if not file_paths:
            return None

        # Normalize all paths (resolve mapped drives to UNC)
        normalized = [normalize_path(fp) for fp in file_paths]

        if not root_path:
            root_path = self.find_common_root(normalized)

        if not root_path:
            # Still no common root — shouldn't happen after normalization,
            # but fall back to multi-root just in case
            return self._build_multi_root_tree(normalized)

        return self._build_single_root_tree(normalized, root_path)

    def _build_single_root_tree(self, file_paths: List[str], root_path: str) -> FolderNode:
        """Build tree under a single root path."""
        root_path = os.path.normpath(root_path)
        root = FolderNode(
            name=os.path.basename(root_path) or root_path,
            full_path=root_path,
        )

        for fp_norm in file_paths:
            dir_path = os.path.dirname(fp_norm)

            try:
                rel = os.path.relpath(dir_path, root_path)
            except ValueError:
                continue

            if rel == '.':
                root.file_count += 1
                continue

            parts = rel.split(os.sep)
            node = root
            current_path = root_path
            for part in parts:
                if not part or part == '.':
                    continue
                current_path = os.path.join(current_path, part)
                if part not in node.children:
                    node.children[part] = FolderNode(
                        name=part,
                        full_path=current_path,
                    )
                node = node.children[part]

            node.file_count += 1

        self._compute_totals(root)
        return root

    def _build_multi_root_tree(self, file_paths: List[str]) -> FolderNode:
        """Build tree when paths span multiple drives/shares (fallback)."""
        groups: Dict[str, List[str]] = {}
        for fp_norm in file_paths:
            parts = fp_norm.split(os.sep)
            if len(parts) >= 3 and parts[0] == '' and parts[1] == '':
                key = os.sep.join(parts[:4])  # \\server\share
            elif parts[0].endswith(':'):
                key = parts[0] + os.sep
            else:
                key = parts[0]
            groups.setdefault(key, []).append(fp_norm)

        virtual = FolderNode(name="All Locations", full_path="", is_indexed=True)

        for group_key, group_paths in sorted(groups.items()):
            group_root = self.find_common_root(group_paths)
            if group_root:
                subtree = self._build_single_root_tree(group_paths, group_root)
            else:
                subtree = FolderNode(
                    name=os.path.basename(group_key.rstrip(os.sep)) or group_key,
                    full_path=group_key,
                    file_count=len(group_paths),
                )
            if len(groups) > 1:
                location = group_key.rstrip(os.sep).rstrip('/')
                subtree.name = f"{location} / {subtree.name}"
            display_key = group_key.replace(os.sep, '/').strip('/')
            virtual.children[display_key] = subtree

        self._compute_totals(virtual)
        return virtual

    def _compute_totals(self, node: FolderNode):
        """Recursively compute total_count for each node."""
        total = node.file_count
        for child in node.children.values():
            self._compute_totals(child)
            total += child.total_count
        node.total_count = total

    def discover_unindexed(
        self, folder_path: str, indexed_children: Set[str],
        extensions: Set[str] = None
    ) -> List[str]:
        """Discover subfolders on disk not in the indexed set.

        Read-only: single os.scandir(), no recursion.
        Catches OSError for offline/inaccessible drives.

        Args:
            folder_path: Absolute path to scan.
            indexed_children: Set of child folder names already in the DB tree.
            extensions: Unused, reserved for future file counting.

        Returns:
            List of unindexed subfolder names.
        """
        try:
            unindexed = []
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in indexed_children and not entry.name.startswith('.'):
                                unindexed.append(entry.name)
                    except OSError:
                        continue
            return sorted(unindexed, key=str.lower)
        except OSError:
            return []
