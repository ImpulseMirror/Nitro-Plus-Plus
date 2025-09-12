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


def _vswhere() -> str | None:
    """Return MSBuild.exe path discovered via vswhere, or None if unavailable."""
    vswhere = r"C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
    if not Path(vswhere).is_file():
        return None
    try:
        msbuild = subprocess.check_output(
            [vswhere, "-latest", "-requires", "Microsoft.Component.MSBuild",
             "-find", r"MSBuild\\**\\Bin\\MSBuild.exe"],
            text=True
        ).strip()
        return msbuild or None
    except Exception:
        return None


def _ensure_nipa_built() -> str:
    """
    Ensure we have bin/nipa.exe:
      1) if PROJECT_ROOT/bin/nipa.exe exists -> use it
      2) else clone/update repo, then build external/nipa/vc10 with modern toolset
         (force PlatformToolset=v143/v142, Platform=Win32), copy result to bin/.
    Returns absolute path to the exe.
    """
    out_exe = PROJECT_ROOT / "bin" / "nipa.exe"
    if out_exe.is_file():
        return str(out_exe)

    sub = _ensure_nipa_repo()  # clone if missing, pull if present
    msbuild = _vswhere()
    if not msbuild:
        raise RuntimeError("MSBuild not found. Install Visual Studio Build Tools (C++ workload).")

    # Find a solution/project under vc10/
    vc10 = sub / "vc10"
    candidates = [vc10 / "nipa.sln", vc10 / "nipa.vcxproj"] + \
                 list(vc10.glob("*.sln")) + list(vc10.glob("*.vcxproj"))
    if not candidates:
        raise FileNotFoundError("No Visual Studio solution/project under external/nipa/vc10/")
    sln = next(p for p in candidates if p.exists())

    # Build Release|Win32 and force a modern toolset; v143 (VS2022) then v142 (VS2019)
    # Some older solutions don’t have x64 configs; stick to Win32.
    for toolset in ("v143", "v142"):
        args = [
            msbuild, str(sln), "/m",
            "/p:Configuration=Release",
            "/p:Platform=Win32",
            f"/p:PlatformToolset={toolset}",
        ]
        print("MSBuild:", " ".join(args))
        proc = subprocess.run(args, cwd=str(sln.parent), text=True)
        # Look for produced exe anywhere in the repo (Release folders, etc.)
        hits = glob.glob(str(sub / "**" / "nipa.exe"), recursive=True)
        if proc.returncode == 0 and hits:
            exe = Path(hits[0])
            out_exe.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exe, out_exe)
            return str(out_exe)

    raise RuntimeError(
        "Could not build nipa with modern toolset. "
        "You can either install the VS2010 (v100) toolset OR set NIPA_EXE to a prebuilt binary."
    )


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
