"""Learn 8,000 shared BPE merges using only the cleaned training split."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, write_json
from src.tokenizer import train_bpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--merges", type=int, default=8000)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/bpe_statistics.json")
    args = parser.parse_args()
    metadata = train_bpe(args.data_dir, args.merges)
    write_json(args.report, metadata)
    print(f"BPE complete: {metadata['learned_merges']} merges, {metadata['vocab_size']} tokens")
