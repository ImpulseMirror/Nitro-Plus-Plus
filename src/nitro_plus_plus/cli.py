import argparse, json, torch, re, os
from pathlib import Path
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSeq2SeqLM, AutoModelForCausalLM)
from huggingface_hub import snapshot_download
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List, Optional
import datetime
import sys, subprocess
import shutil, glob

NIPA_REPO = "https://github.com/Wilhansen/nipa"


# Resolve project root: <repo>/src/nitro_plus_plus/cli.py -> root is 3 up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

class BalancedJsonStopper(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tok = tokenizer
        self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        # Only inspect newly generated tokens
        gen_ids = input_ids[0][self.prompt_len:].tolist()
        if not gen_ids:
            return False
        text = self.tok.decode(gen_ids, skip_special_tokens=False)
        depth = 0
        started = False
        for ch in text:
            if ch == '{':
                depth += 1
                started = True
            elif ch == '}':
                depth -= 1
                if started and depth == 0:
                    return True
        return False

def _safe_ctx_len(tok, default=8192):
    m = getattr(tok, "model_max_length", None)
    return m if isinstance(m, int) and m > 0 and m <= 100_000 else default


def get_local_model(repo_id: str, local_dir: Path = MODELS_DIR) -> str:
    """
    Ensure model is available locally under <repo>/models/<repo_id>.
    If missing, download it. Returns the local path.
    """
    target_dir = Path(local_dir) / repo_id.replace("/", "_")
    if not target_dir.exists():
        print(f"Downloading model {repo_id} to {target_dir}...")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id,
            revision="main",
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # store real files
        )
    else:
        print(f"Using cached model at {target_dir}")
    return str(target_dir)

def load_model(repo_or_path, load_4bit=False, load_8bit=False):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert torch.cuda.is_available(), "PyTorch CUDA build not installed."

    cfg = AutoConfig.from_pretrained(repo_or_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id


    # cap absurd model_max_length values to avoid Rust tokenizer overflow on Windows
    max_ctx = getattr(tok, "model_max_length", None)
    if not isinstance(max_ctx, int) or max_ctx > 100_000:
        tok.model_max_length = 8192


    common = dict(
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    # if load_4bit: common["load_in_4bit"] = True
    # if load_8bit: common["load_in_8bit"] = True

    if getattr(cfg, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_or_path, **common)
        kind = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(repo_or_path, **common)
        kind = "causal"

    print("cuda:", torch.cuda.get_device_name(0))
    print("model dtype:", next(model.parameters()).dtype)
    return tok, model, kind

def extract_json_balanced(s: str, fallback):
    # find the last well-formed {...} block by brace balancing
    start = None
    depth = 0
    best = None
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    best = s[start:i+1]
    if best:
        try:
            json.loads(best)  # validate
            return best
        except Exception:
            pass
    return json.dumps(fallback, ensure_ascii=False)


def translate(repo_or_path, text, max_new=256, temp=0.0, load_4bit=False, load_8bit=False):
    if not Path(repo_or_path).exists():
        repo_or_path = get_local_model(repo_or_path)

    tok, model, kind = load_model(repo_or_path, load_4bit, load_8bit)

    inputs = build_inputs(tok, text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    stopper = StoppingCriteriaList([BalancedJsonStopper(tok, prompt_len)])

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)

    out = model.generate(
        **inputs,
        max_new_tokens=min(max_new, 256),
        do_sample=False,                 # force greedy for determinism
        temperature=None,
        repetition_penalty=1.0,
        eos_token_id=eos_ids or None,
        pad_token_id=pad_id,
        stopping_criteria=stopper
    )
    
    new_tokens = out[0][prompt_len:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)

    raw = extract_json_balanced(decoded, {"source": text, "translation": decoded.strip(), "notes": ""})
    return _denest_and_normalize(raw, text)



def build_inputs(tok, jp_text: str):
    sys = (
        "You are a professional JP→EN translator.\n"
        "Return EXACTLY ONE compact JSON object with keys in this order: "
        '{"source","translation","notes"}.\n'
        "Hard rules:\n"
        "1) Output ONLY the JSON object (no code fences, no prose).\n"
        "2) Do NOT wrap the JSON in quotes or stringify it.\n"
        "3) Each field value must be plain text. The value of \"translation\" "
        "   MUST NOT contain '{' or '}' and MUST NOT contain another JSON object/stringified JSON.\n"
        "4) Use standard English names for well-known proper nouns; otherwise Hepburn romaji. "
        "   Prefer romaji over literal morpheme translations (e.g., 東海→Toukai; 東海地方→Toukai region).\n"
        "5) Keep the translation concise; put brief clarifications in \"notes\" only."
    )

    # Positive few-shots (anchor style + the 東海 case)
    ex1_src = "日本語のテキスト"
    ex1_out = {"source": ex1_src, "translation": "Japanese Text", "notes": ""}

    ex2_src = "東海では局地的に非常に激しい雨が降っています。"
    ex2_out = {
        "source": ex2_src,
        "translation": "In the Toukai region, localized torrential rain is falling.",
        "notes": "「東海」 = Japan’s Toukai region (Aichi, Shizuoka, Mie, Gifu), not 'East Sea'."
    }

    ex1_json = json.dumps(ex1_out, ensure_ascii=False)
    ex2_json = json.dumps(ex2_out, ensure_ascii=False)

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": ex1_src},
        {"role": "assistant", "content": ex1_json},
        {"role": "user", "content": ex2_src},
        {"role": "assistant", "content": ex2_json},
        {"role": "user", "content": jp_text.strip()},
    ]

    ctx = _safe_ctx_len(tok)
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", truncation=True, max_length=ctx)
    else:
        prompt = (
            f"<system>{sys}</system>\n"
            f"<user>{ex1_src}</user>\n<assistant>{ex1_json}</assistant>\n"
            f"<user>{ex2_src}</user>\n<assistant>{ex2_json}</assistant>\n"
            f"<user>{jp_text.strip()}</user>\n<assistant>"
        )
        return tok(prompt, return_tensors="pt", truncation=True, max_length=ctx)


def _denest_and_normalize(raw_json: str, source_text: str) -> str:
    """
    If 'translation' contains stringified (or truncated) JSON, unwrap it and merge notes.
    Always return a flat {"source","translation","notes"} JSON string.
    """
    try:
        obj = json.loads(raw_json)
    except Exception:
        # Not even top-level JSON: return as-is
        return raw_json

    t = obj.get("translation", "")
    if isinstance(t, str):
        ts = t.strip()
        # Looks like the model stuffed JSON into the string
        if ts.startswith("{") and ("\"translation\"" in ts or "\"source\"" in ts):
            # 1) Try parsing the inner JSON normally
            try:
                inner = json.loads(ts)
                if isinstance(inner, dict):
                    if "translation" in inner:
                        obj["translation"] = inner.get("translation", "")
                    if not obj.get("notes") and inner.get("notes"):
                        obj["notes"] = inner["notes"]
            except Exception:
                # 2) Inner JSON is malformed/truncated -> regex fallback
                s = ts
                # Grab inner "translation": "..." (non-greedy), stopping at next field
                m = re.search(r'"translation"\s*:\s*"(.*?)"\s*,\s*"(?:notes|source)"', s, flags=re.S)
                if not m:
                    # Fall back to first closing , or } after the string
                    m = re.search(r'"translation"\s*:\s*"(.*?)"\s*[,}]', s, flags=re.S)
                if m:
                    raw_val = m.group(1)
                    try:
                        obj["translation"] = json.loads(f'"{raw_val}"')  # unescape
                    except Exception:
                        obj["translation"] = raw_val.replace('\\"', '"')

                # Try to grab inner notes if top-level notes is empty
                if not obj.get("notes"):
                    m2 = re.search(r'"notes"\s*:\s*"(.*?)"\s*[,}]', s, flags=re.S)
                    if m2:
                        raw_notes = m2.group(1)
                        try:
                            obj["notes"] = json.loads(f'"{raw_notes}"')
                        except Exception:
                            obj["notes"] = raw_notes.replace('\\"', '"')

    # Normalize + enforce fields
    out = {
        "source": source_text,
        "translation": str(obj.get("translation", "")).strip(),
        "notes": str(obj.get("notes", "")).strip(),
    }
    return json.dumps(out, ensure_ascii=False)




def translate_with_loaded(tok, model, kind, text, max_new=256, temp=0.0):
    inputs = build_inputs(tok, text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    stopper = StoppingCriteriaList([BalancedJsonStopper(tok, prompt_len)])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else None

    out = model.generate(
        **inputs,
        max_new_tokens=min(max_new, 96),
        do_sample=False,
        temperature=None,
        repetition_penalty=1.0,
        eos_token_id=eos_ids,
        pad_token_id=pad_id,
        stopping_criteria=stopper,
    )
    new_tokens = out[0][prompt_len:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    raw = extract_json_balanced(decoded, {"source": text, "translation": decoded.strip(), "notes": ""})
    return _denest_and_normalize(raw, text)



def kill_port(port: int) -> int:
    # Windows PowerShell approach
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            f"(Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction SilentlyContinue | "
            f"Select-Object -Expand OwningProcess -Unique) -join ' '"
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            print(f"No listener on port {port}.")
            return 0
        for pid in [p for p in out.split() if p.isdigit()]:
            subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
            print(f"Killed PID {pid} on port {port}.")
        return 0
    except Exception as e:
        print(f"Failed to kill port {port}: {e}", file=sys.stderr)
        return 1

def kill_port_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    args = ap.parse_args()
    sys.exit(kill_port(args.port))


def _ensure_nipa_repo() -> Path:
    sub = PROJECT_ROOT / "external" / "nipa"
    if sub.exists():
        # If it's a git repo, lightly update (best-effort).
        if (sub / ".git").exists():
            subprocess.run(["git", "-C", str(sub), "pull", "--ff-only"], check=False)
        return sub
    # clone (depth=1 to keep it fast)
    sub.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", NIPA_REPO, str(sub)], check=True)
    return sub

def _vswhere() -> Optional[str]:
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not Path(vswhere).is_file():
        return None
    try:
        msbuild = subprocess.check_output(
            [vswhere, "-latest", "-requires", "Microsoft.Component.MSBuild",
             "-find", r"MSBuild\**\Bin\MSBuild.exe"],
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
    # 1) env override
    env = os.environ.get("NIPA_EXE")
    if env and Path(env).is_file():
        return env
    # 2) project bin or build from source
    p = PROJECT_ROOT / "bin" / "nipa.exe"
    if p.is_file():
        return str(p)
    return _ensure_nipa_built()


def nipa_extract(archive_path: str, game_id: str | None = None, cwd: str | None = None) -> str:
    """
    Run: nipa -x <archive>  or  nipa -xg <archive> <GameID>
    Works around non-ASCII paths by copying to an ASCII temp file.
    Returns the output folder path (based on the *original* archive name).
    """
    import time, shutil as _shutil

    exe = _find_nipa_exe()
    archive = Path(archive_path)
    work = Path(cwd) if cwd else Path.cwd()
    work.mkdir(parents=True, exist_ok=True)

    # Use an ASCII temp name if the archive path contains non-ASCII chars
    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    orig_stem = archive.stem  # e.g., "nss"
    use_path = archive
    tmp_dir = None
    if not _is_ascii(str(archive)):
        tmp_dir = work / f"nipa_job_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        use_path = tmp_dir / "input.npa"  # ASCII name
        _shutil.copy2(archive, use_path)

    # Try a few CLI layouts that different nipa builds accept
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
        # Capture BYTES; we'll decode ourselves to avoid UnicodeDecodeError
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

    # Clean up temp copy if we made one
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

    # nipa outputs to a folder named after the archive's *filename* (without ext)
    produced = work / Path(use_path).with_suffix("").name  # "input" if we used the temp
    desired = work / orig_stem                              # "nss" from original path
    if produced != desired and produced.exists():
        if desired.exists():
            try:
                _shutil.rmtree(desired)
            except Exception:
                pass
        produced.rename(desired)

    return str(desired)


def collect_nss_strings(start: str | Path,
                        recursive: bool = False,
                        encoding: str = "shift_jis",
                        errors: str = "replace") -> List[str]:
    """
    Read all .nss files under `start` (non-recursive by default),
    decode as Shift-JIS, and return a JSON-ready list of strings.
    """
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
    """
    Build a JSON array (UTF-8) of Shift-JIS-decoded .nss file contents.
    If `out` is provided, write it to disk; always return the JSON string.
    """
    arr = collect_nss_strings(start, recursive=recursive, encoding=encoding, errors=errors)
    blob = json.dumps(arr, ensure_ascii=False, indent=2)
    if out:
        Path(out).write_text(blob, encoding="utf-8")
    return blob


def serve(args):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    # Load once; stays in memory while the server runs
    repo = args.model
    local = repo if Path(repo).exists() else get_local_model(repo)
    tok, model, kind = load_model(local, args.load_4bit, args.load_8bit)

    app = FastAPI()

    class Req(BaseModel):
        text: str
        max_new: int = 256
        temp: float = 0.0

        # ---- BATCH TRANSLATE ----
    class BatchReq(BaseModel):
        texts: List[str]
        max_new: int = 256
        temp: float = 0.0
        outfile: Optional[str] = None  # e.g. "outputs/run.jsonl"; default auto-named

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/translate")
    def translate_endpoint(r: Req):
        # Return a dict (FastAPI will serialize)
        return json.loads(translate_with_loaded(tok, model, kind, r.text, r.max_new, r.temp))

    @app.post("/batch_translate")
    def batch_translate(r: BatchReq):
        # Resolve output file path
        if r.outfile:
            out_path = Path(r.outfile)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = PROJECT_ROOT / f"outputs/batch_{stamp}.json"  # default to JSON array

        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for i, text in enumerate(r.texts, 1):
            res = json.loads(translate_with_loaded(tok, model, kind, text, r.max_new, r.temp))
            results.append({"index": i, **res})

        # Write depending on extension: .json => array, .jsonl => NDJSON
        if out_path.suffix.lower() == ".jsonl":
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in results:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(results, ensure_ascii=False, indent=2))
                fh.write("\n")

        return {
            "count": len(results),
            "outfile": str(out_path),
            "results": results  # keep or drop as you prefer
        }



    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, workers=1)



def _nss_to_json_map(start, recursive=False, encoding="shift_jis", errors="replace"):
    from pathlib import Path
    import json
    root = Path(start)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pattern = "**/*.nss" if recursive else "*.nss"
    out = {}
    for fp in sorted(root.glob(pattern)):
        with fp.open("r", encoding=encoding, errors=errors) as f:
            out[fp.name] = {"original": f.read()}
    return json.dumps(out, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # one-shot (current behavior)
    ap.add_argument("--model", help="HF repo id or local folder")
    ap.add_argument("--text", help="Japanese text")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")

    # server mode
    sv = sub.add_parser("serve")
    sv.add_argument("--model", required=True)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sv.add_argument("--load_4bit", action="store_true")
    sv.add_argument("--load_8bit", action="store_true")
    sv.add_argument("--reload", action="store_true")

    nx = sub.add_parser("nipa", help="Extract an NPA archive with nipa")
    nx.add_argument("archive", help="Path to .npa file (e.g., nss.npa)")
    nx.add_argument("-g", "--game-id", default=None, help="Optional game ID")
    nx.add_argument("--cwd", default=None, help="Working dir (default=current)")

    nj = sub.add_parser("nssjson", help="Emit JSON map of .nss contents decoded as Shift-JIS")
    nj.add_argument("start", help="Directory containing .nss files")
    nj.add_argument("--out", help="Write JSON to this file (UTF-8). If omitted, prints to stdout")
    nj.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    nj.add_argument("--encoding", default="shift_jis", help="Source encoding (default: shift_jis)")
    nj.add_argument("--errors", default="replace", choices=["strict", "ignore", "replace"],
                    help="Decoding error handling (default: replace)")

    args = ap.parse_args()

    if args.cmd == "serve":
        return serve(args)
    elif args.cmd == "nipa":
        out = nipa_extract(args.archive, args.game_id, args.cwd)
        print(out); return
    elif args.cmd == "nssjson":
        from pathlib import Path
        blob = _nss_to_json_map(args.start,
                                recursive=args.recursive,
                                encoding=args.encoding,
                                errors=args.errors)
        if args.out:
            Path(args.out).write_text(blob, encoding="utf-8")
        else:
            print(blob)
        return

    print(translate(args.model, args.text, args.max_new, args.temp, args.load_4bit, args.load_8bit))

if __name__ == "__main__":
    main()