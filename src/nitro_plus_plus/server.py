import datetime, json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from .config import PROJECT_ROOT
from .translate import get_local_model, load_model, translate_with_loaded


def serve(args):
    """Start the FastAPI server and expose translate/batch endpoints."""
    # Load once; stays in memory while the server runs
    repo = args.model
    local = repo if Path(repo).exists() else get_local_model(repo)
    tok, model, kind = load_model(local, args.load_4bit, args.load_8bit)

    app = FastAPI()

    class Req(BaseModel):
        text: str
        max_new: int = 256
        temp: float = 0.0

    class BatchReq(BaseModel):
        texts: List[str]
        max_new: int = 256
        temp: float = 0.0
        outfile: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/translate")
    def translate_endpoint(r: Req):
        return json.loads(translate_with_loaded(tok, model, kind, r.text, r.max_new, r.temp))

    @app.post("/batch_translate")
    def batch_translate(r: BatchReq):
        if r.outfile:
            out_path = Path(r.outfile)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = PROJECT_ROOT / f"outputs/batch_{stamp}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for i, text in enumerate(r.texts, 1):
            res = json.loads(translate_with_loaded(tok, model, kind, text, r.max_new, r.temp))
            results.append({"index": i, **res})

        if out_path.suffix.lower() == ".jsonl":
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in results:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(results, ensure_ascii=False, indent=2))
                fh.write("\n")

        return {"count": len(results), "outfile": str(out_path), "results": results}

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, workers=1)
