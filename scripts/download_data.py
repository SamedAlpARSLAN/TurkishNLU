"""Download + extract MASSIVE-tr (plan §6.1).

    python scripts/download_data.py                # -> data/raw/tr-TR.jsonl
    python scripts/download_data.py --locale de-DE # any other locale too
"""
import argparse

import _bootstrap  # noqa: F401
from src.data import download_massive, load_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--locale", default="tr-TR")
    args = ap.parse_args()

    path = download_massive(args.raw_dir, args.locale)
    for part in ("train", "dev", "test"):
        n = len(load_examples(path, part))
        print(f"  {part:<6}: {n:>6} examples")
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
