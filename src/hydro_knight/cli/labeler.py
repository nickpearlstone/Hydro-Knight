"""Start the browser labeler on http://127.0.0.1:8765 (Ctrl+C to stop)."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--no-browser", action="store_true", help="don't open a browser tab"
    )
    args = ap.parse_args()
    from hydro_knight.annotate.server import main as serve

    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
