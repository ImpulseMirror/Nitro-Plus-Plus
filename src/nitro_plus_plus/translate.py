import os, re, json, torch
from pathlib import Path
from typing import List

from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList,
)
from huggingface_hub import snapshot_download

from .config import PROJECT_ROOT, MODELS_DIR


class BalancedJsonStopper(StoppingCriteria):
    """Stops generation once a balanced JSON object is closed."""
    def __init__(self, tokenizer, prompt_len: int):
        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
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
    return m if isinstance(m, int) and 0 < m <= 100_000 else default


def get_local_model(repo_id: str, local_dir: Path = MODELS_DIR) -> str:
    """Ensure HF `repo_id` is downloaded under models/ and return local path."""
    target_dir = Path(local_dir) / repo_id.replace("/", "_")
    if not target_dir.exists():
        print(f"Downloading model {repo_id} to {target_dir}...")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id,
            revision="main",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"Using cached model at {target_dir}")
    return str(target_dir)


def load_model(repo_or_path, load_4bit=False, load_8bit=False):
    """Load tokenizer/model on CUDA; returns `(tok, model, kind)` where kind is `seq2seq` or `causal`."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert torch.cuda.is_available(), "PyTorch CUDA build not installed."

    cfg = AutoConfig.from_pretrained(repo_or_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

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
            json.loads(best)
            return best
        except Exception:
            pass
    return json.dumps(fallback, ensure_ascii=False)


def build_inputs(tok, jp_text: str):
    """Build chat/prompt input tensors for the JP→EN translation task."""
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

    # Few-shots. Strings may contain non-ASCII; keep as-is.
    ex1_src = "日本語テキスト"
    ex1_out = {"source": ex1_src, "translation": "Japanese Text", "notes": ""}

    ex2_src = "東海では局地的に非常に激しい雨が降っています。"
    ex2_out = {
        "source": ex2_src,
        "translation": "In the Toukai region, localized torrential rain is falling.",
        "notes": "『東海』= Japan’s Toukai region (Aichi, Shizuoka, Mie, Gifu), not 'East Sea'.",
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
    """Flatten nested/quoted JSON into a stable {source,translation,notes} JSON string."""
    try:
        obj = json.loads(raw_json)
    except Exception:
        return raw_json

    t = obj.get("translation", "")
    if isinstance(t, str):
        ts = t.strip()
        if ts.startswith("{") and ("\"translation\"" in ts or "\"source\"" in ts):
            try:
                inner = json.loads(ts)
                if isinstance(inner, dict):
                    if "translation" in inner:
                        obj["translation"] = inner.get("translation", "")
                    if not obj.get("notes") and inner.get("notes"):
                        obj["notes"] = inner["notes"]
            except Exception:
                s = ts
                m = re.search(r'"translation"\s*:\s*"(.*?)"\s*,\s*"(?:notes|source)"', s, flags=re.S)
                if not m:
                    m = re.search(r'"translation"\s*:\s*"(.*?)"\s*[,}]', s, flags=re.S)
                if m:
                    raw_val = m.group(1)
                    try:
                        obj["translation"] = json.loads(f'"{raw_val}"')
                    except Exception:
                        obj["translation"] = raw_val.replace('\\"', '"')
                if not obj.get("notes"):
                    m2 = re.search(r'"notes"\s*:\s*"(.*?)"\s*[,}]', s, flags=re.S)
                    if m2:
                        raw_notes = m2.group(1)
                        try:
                            obj["notes"] = json.loads(f'"{raw_notes}"')
                        except Exception:
                            obj["notes"] = raw_notes.replace('\\"', '"')

    out = {
        "source": source_text,
        "translation": str(obj.get("translation", "")).strip(),
        "notes": str(obj.get("notes", "")).strip(),
    }
    return json.dumps(out, ensure_ascii=False)


def translate(repo_or_path, text, max_new=256, temp=0.0, load_4bit=False, load_8bit=False):
    """Translate a single Japanese string to JSON using the given model repo/path."""
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
        do_sample=False,
        temperature=None,
        repetition_penalty=1.0,
        eos_token_id=eos_ids or None,
        pad_token_id=pad_id,
        stopping_criteria=stopper,
    )

    new_tokens = out[0][prompt_len:]
    decoded = tok.decode(new_tokens, skip_special_tokens=True)

    raw = extract_json_balanced(decoded, {"source": text, "translation": decoded.strip(), "notes": ""})
    return _denest_and_normalize(raw, text)


def translate_with_loaded(tok, model, kind, text, max_new=256, temp=0.0):
    """Translate using preloaded `(tok, model, kind)` without reloading the model."""
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
