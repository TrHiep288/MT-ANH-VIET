"""Download the six original files and verify pinned hashes and split counts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, download, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source", choices=("stanford", "mirror"), default="stanford")
    args = parser.parse_args()
    download(args.data_dir / "raw", read_json(ROOT / "configs/data_sources.json"), args.source)
