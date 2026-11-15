"""Validate splits, normalize, deduplicate train, and preserve references."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, prepare, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/data_statistics.json")
    args = parser.parse_args()
    print(prepare(args.data_dir, args.report, read_json(ROOT / "configs/data_sources.json"),
                  read_json(ROOT / "configs/train_exclusions.json")))
