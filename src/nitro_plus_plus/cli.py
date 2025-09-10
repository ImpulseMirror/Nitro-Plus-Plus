import argparse, json, torch, re, os
from pathlib import Path
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSeq2SeqLM, AutoModelForCausalLM)
from huggingface_hub import snapshot_download
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List, Optional
import datetime


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

    # cap absurd model_max_length...
    max_ctx = getattr(tok, "model_max_length", None)
    if not isinstance(max_ctx, int) or max_ctx > 100_000:
        tok.model_max_length = 8192


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

    return extract_json_balanced(decoded, {"source": text, "translation": decoded.strip(), "notes": ""})


def build_inputs(tok, jp_text: str):
    sys = (
        "You are a professional JP→EN visual-novel/news translator. "
        'Output ONLY valid JSON with keys exactly: {"source","translation","notes"}. '
        "Rules for proper nouns (people, places, organizations, regions, stations, universities): "
        "use the standard English name if it exists; otherwise use Hepburn-style romaji. "
        "Do NOT translate morphemes literally. "
        "Examples: 東海 → Toukai (region), 東海地方 → Toukai region, 東海大学 → Tokai University. "
        "If ambiguous, prefer romaji rather than literal meanings like 'East Sea'. "
        "Keep the translation concise; put brief explanations in 'notes' only."
    )

    # Few-shot: anchor 東海 usage in a weather context
    ex_src = "東海では局地的に非常に激しい雨が降っています。"
    ex_out = {
        "source": ex_src,
        "translation": "In the Toukai region, localized torrential rain is falling.",
        "notes": "「東海」 here refers to Japan’s Toukai region (Aichi, Shizuoka, Mie, Gifu), not 'East Sea'."
    }
    ex_json = json.dumps(ex_out, ensure_ascii=False)

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": ex_src},
        {"role": "assistant", "content": ex_json},
        {"role": "user", "content": jp_text.strip()},
    ]

    ctx = _safe_ctx_len(tok)
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", truncation=True, max_length=ctx)
    else:
        prompt = (
            f"<system>{sys}</system>\n"
            f"<user>{ex_src}</user>\n"
            f"<assistant>{ex_json}</assistant>\n"
            f"<user>{jp_text.strip()}</user>\n<assistant>"
        )
        return tok(prompt, return_tensors="pt", truncation=True, max_length=ctx)


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
    return extract_json_balanced(decoded, {"source": text, "translation": decoded.strip(), "notes": ""})


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
            out_path = PROJECT_ROOT / f"outputs/batch_{stamp}.jsonl"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        with out_path.open("w", encoding="utf-8") as fh:
            for i, text in enumerate(r.texts, 1):
                res = json.loads(translate_with_loaded(tok, model, kind, text, r.max_new, r.temp))
                rec = {"index": i, **res}
                results.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return {
            "count": len(results),
            "outfile": str(out_path),
            "results": results  # keep if you want the responses inline; remove to only return path
        }


    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, workers=1)



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

    args = ap.parse_args()

    if args.cmd == "serve":
        return serve(args)

    # default one-shot
    print(translate(args.model, args.text, args.max_new, args.temp, args.load_4bit, args.load_8bit))


if __name__ == "__main__":
    main()
