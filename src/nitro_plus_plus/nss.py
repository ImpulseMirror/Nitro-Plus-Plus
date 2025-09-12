import json
from pathlib import Path
from typing import List


def collect_nss_strings(start: str | Path,
                        recursive: bool = False,
                        encoding: str = "shift_jis",
                        errors: str = "replace") -> List[str]:
    """Read `.nss` files under `start` and return their decoded contents as a list of strings."""
    root = Path(start)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern = "**/*.nss" if recursive else "*.nss"
    texts: List[str] = []

    for fp in sorted(root.glob(pattern)):
        with fp.open("r", encoding=encoding, errors=errors) as f:
            texts.append(f.read())

    return texts


def nss_to_json_array(start: str | Path,
                      out: str | Path | None = None,
                      recursive: bool = False,
                      encoding: str = "shift_jis",
                      errors: str = "replace") -> str:
    """Return a JSON array (UTF-8) of `.nss` contents; optionally write to `out`."""
    arr = collect_nss_strings(start, recursive=recursive, encoding=encoding, errors=errors)
    blob = json.dumps(arr, ensure_ascii=False, indent=2)
    if out:
        Path(out).write_text(blob, encoding="utf-8")
    return blob


def nss_to_json_map(start, recursive=False, encoding="shift_jis", errors="replace"):
    """Return a JSON object mapping filename → { original } for `.nss` files."""
    root = Path(start)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern = "**/*.nss" if recursive else "*.nss"
    out = {}
    for fp in sorted(root.glob(pattern)):
        with fp.open("r", encoding=encoding, errors=errors) as f:
            out[fp.name] = {"original": f.read()}
    return json.dumps(out, ensure_ascii=False, indent=2)
