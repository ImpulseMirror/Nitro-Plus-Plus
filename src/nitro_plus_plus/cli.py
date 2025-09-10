import argparse, json, torch, re, os
from pathlib import Path
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSeq2SeqLM, AutoModelForCausalLM)
from huggingface_hub import snapshot_download
from transformers import StoppingCriteria, StoppingCriteriaList

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

    tok = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True, trust_remote_code=True)

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
    sys = ("You are a professional JP→EN visual-novel translator. "
           "Output ONLY valid JSON with keys exactly: "
           '{"source","translation","notes"}. '
           "No extra text, no code fences.")

    # one-shot example to anchor format & brevity
    example_user = "課長は根回しもしないで方針を変えたから、現場では「空気を読めない」って不満が爆発してる。"
    example_assistant = "{\"source\":\"課長は根回しもしないで方針を変えたから、現場では「空気を読めない」って不満が爆発してる。\",\"translation\":\"The manager changed the policy without any nemawashi, so the team on the ground is fuming that he can't read the room.\",\"notes\":\"「根回し」(nemawashi) = informal, behind-the-scenes consensus-building common in Japanese workplaces; 「空気を読めない」 = 'can't read the room', i.e., insensitive to shared, unspoken context.\"}"

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": jp_text.strip()},
    ]

    ctx = _safe_ctx_len(tok)
    
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", truncation=True, max_length=ctx)
    else:
        prompt = (f"<system>{sys}</system>\n"
                f"<user>{example_user}</user>\n"
                f"<assistant>{example_assistant}</assistant>\n"
                f"<user>{jp_text.strip()}</user>\n<assistant>")
        return tok(prompt, return_tensors="pt", truncation=True, max_length=ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local folder (e.g., CohereLabs/aya-23-8B)")
    ap.add_argument("--text", required=True, help="Japanese text")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    args = ap.parse_args()
    print(translate(args.model, args.text, args.max_new, args.temp, args.load_4bit, args.load_8bit))

if __name__ == "__main__":
    main()
