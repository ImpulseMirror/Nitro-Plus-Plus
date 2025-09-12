import argparse
from pathlib import Path

from .translate import translate
from .nipa import nipa_extract
from .nss import nss_to_json_map


def main():
    """CLI entrypoint that parses args and dispatches to subcommands."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # one-shot translate
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

    # nipa extract
    nx = sub.add_parser("nipa", help="Extract an NPA archive with nipa")
    nx.add_argument("archive", help="Path to .npa file (e.g., nss.npa)")
    nx.add_argument("-g", "--game-id", default=None, help="Optional game ID")
    nx.add_argument("--cwd", default=None, help="Working dir (default=current)")

    # nss -> JSON
    nj = sub.add_parser("nssjson", help="Emit JSON map of .nss contents decoded as Shift-JIS")
    nj.add_argument("start", help="Directory containing .nss files")
    nj.add_argument("--out", help="Write JSON to this file (UTF-8). If omitted, prints to stdout")
    nj.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    nj.add_argument("--encoding", default="shift_jis", help="Source encoding (default: shift_jis)")
    nj.add_argument("--errors", default="replace", choices=["strict", "ignore", "replace"],
                    help="Decoding error handling (default: replace)")

    args = ap.parse_args()

    if args.cmd == "serve":
        from .server import serve  # defer FastAPI import unless needed
        return serve(args)
    elif args.cmd == "nipa":
        out = nipa_extract(args.archive, args.game_id, args.cwd)
        print(out)
        return
    elif args.cmd == "nssjson":
        blob = nss_to_json_map(args.start,
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
