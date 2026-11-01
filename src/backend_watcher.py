from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple


class BackendFileWatcher:
    """Lightweight backend document folder watcher.

    It does not run a separate thread. Streamlit calls scan() on each rerun
    with a throttle interval in app.py. This keeps the app simple and reliable
    on Windows.

    Returns:
        current: all currently visible supported files
        added_or_modified: files whose size/mtime signature changed
        removed: files present in previous snapshot but not present now
    """

    def __init__(self, documents_dir: Path, supported_extensions: Iterable[str]):
        self.documents_dir = Path(documents_dir)
        self.supported_extensions = {ext.lower() for ext in supported_extensions}
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def _is_supported_file(self, path: Path) -> bool:
        if not path.is_file():
            return False

        name = path.name
        if name.startswith(".") or name.startswith("~$"):
            return False
        if name.endswith(".tmp") or name.endswith(".crdownload") or name.endswith(".part"):
            return False

        return path.suffix.lower() in self.supported_extensions

    def _signature(self, path: Path) -> Tuple[int, int]:
        stat = path.stat()
        return int(stat.st_size), int(stat.st_mtime_ns)

    def _relative_key(self, path: Path) -> str:
        return path.relative_to(self.documents_dir).as_posix()

    def scan(self, previous_snapshot: Dict[str, tuple] | None = None):
        previous_snapshot = previous_snapshot or {}

        current: Dict[str, dict] = {}
        added_or_modified: Dict[str, dict] = {}

        for path in sorted(self.documents_dir.rglob("*")):
            if not self._is_supported_file(path):
                continue

            rel = self._relative_key(path)
            signature = self._signature(path)

            meta = {
                "path": str(path),
                "signature": signature,
                "size": signature[0],
                "mtime_ns": signature[1],
                "extension": path.suffix.lower(),
            }
            current[rel] = meta

            if previous_snapshot.get(rel) != signature:
                added_or_modified[rel] = meta

        removed = sorted(set(previous_snapshot.keys()) - set(current.keys()))
        return current, added_or_modified, removed
