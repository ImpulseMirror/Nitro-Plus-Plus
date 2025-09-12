import os, sys, subprocess, shutil, glob
from pathlib import Path

from .config import PROJECT_ROOT, NIPA_REPO


def _ensure_nipa_repo() -> Path:
    """Clone or update the `external/nipa` repository and return its path."""
    sub = PROJECT_ROOT / "external" / "nipa"
    if sub.exists():
        # If it's a git repo, lightly update (best-effort).
        try:
            if (sub / ".git").is_dir():
                subprocess.run(["git", "-C", str(sub), "pull", "--ff-only"], check=False)
        except Exception:
            pass
        return sub

    sub.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", NIPA_REPO, str(sub)], check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to clone nipa repo: {e}")
    return sub


def _ensure_nipa_built() -> str:
    """Return a path to `nipa.exe`, attempting to build if not found."""
    # Prefer prebuilt binary if present under bin/
    pre = PROJECT_ROOT / "bin" / "nipa.exe"
    if pre.is_file():
        return str(pre)

    sub = _ensure_nipa_repo()  # clone if missing, pull if present

    # If a build already exists under external/nipa/vc10/nipa/Release/nipa.exe, use it.
    cand = sub / "vc10" / "nipa" / "Release" / "nipa.exe"
    if cand.is_file():
        return str(cand)

    # Attempt to build via MSBuild if available
    sln = sub / "vc10" / "nipa.sln"
    if sln.is_file():
        # Try common MSBuild locations
        msbuild_cmds = [
            ["msbuild", str(sln), "/p:Configuration=Release"],
            ["C:/Program Files/Microsoft Visual Studio/2022/BuildTools/MSBuild/Current/Bin/MSBuild.exe", str(sln), "/p:Configuration=Release"],
        ]
        for cmd in msbuild_cmds:
            try:
                r = subprocess.run(cmd, cwd=str(sln.parent), check=False)
                if r.returncode == 0 and cand.is_file():
                    return str(cand)
            except Exception:
                pass

    # Last resort: search for any nipa.exe built under the repo
    hits = glob.glob(str(sub / "**/nipa.exe"), recursive=True)
    if hits:
        return hits[0]

    raise RuntimeError("nipa.exe not found and build failed. Please install/build it manually.")


def _find_nipa_exe() -> str:
    env = os.environ.get("NIPA_EXE")
    if env and Path(env).is_file():
        return env
    p = PROJECT_ROOT / "bin" / "nipa.exe"
    if p.is_file():
        return str(p)
    return _ensure_nipa_built()


def nipa_extract(archive_path: str, game_id: str | None = None, cwd: str | None = None) -> str:
    """Extract a Nitro+ `.npa` archive using `nipa.exe`; return output folder path."""
    """
    Run: nipa -x <archive>  or  nipa -xg <archive> <GameID>
    Works around non-ASCII paths by copying to an ASCII temp file.
    Returns the output folder path (based on the original archive name).
    """
    import time, shutil as _shutil

    exe = _find_nipa_exe()
    archive = Path(archive_path)
    work = Path(cwd) if cwd else Path.cwd()
    work.mkdir(parents=True, exist_ok=True)

    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    orig_stem = archive.stem
    use_path = archive
    tmp_dir = None
    if not _is_ascii(str(archive)):
        tmp_dir = work / f"nipa_job_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        use_path = tmp_dir / "input.npa"
        _shutil.copy2(archive, use_path)

    variants = []
    if game_id:
        variants += [
            [exe, "-xg", str(use_path), game_id],
            [exe, "-x", str(use_path), "-g", game_id],
            [exe, "-g", game_id, "-x", str(use_path)],
        ]
    else:
        variants = [[exe, "-x", str(use_path)]]

    def _decode(b: bytes) -> str:
        for enc in ("utf-8", "cp932", sys.getfilesystemencoding() or "", "latin-1"):
            if not enc:
                continue
            try:
                return b.decode(enc)
            except Exception:
                pass
        return b.decode("utf-8", "replace")

    last = None
    for args in variants:
        last = subprocess.run(
            args,
            cwd=str(work),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            shell=False,
        )
        if last.returncode == 0:
            break

    if tmp_dir:
        try:
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if not last or last.returncode != 0:
        raise RuntimeError(
            f"nipa failed ({last.returncode if last else 'n/a'}).\n"
            f"--- stdout ---\n{_decode(last.stdout) if last else ''}\n"
            f"--- stderr ---\n{_decode(last.stderr) if last else ''}"
        )

    produced = work / Path(use_path).with_suffix("").name
    desired = work / orig_stem
    if produced != desired and produced.exists():
        if desired.exists():
            try:
                _shutil.rmtree(desired)
            except Exception:
                pass
        produced.rename(desired)

    return str(desired)
