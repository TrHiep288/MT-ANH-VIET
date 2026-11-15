"""Build per-run CSV and mean/sample-std tables without selecting on test."""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import read_json

FIELDS = ["model", "seed", "split", "decoding", "bleu", "parameters", "train_seconds",
          "decode_seconds", "sentences_per_second", "device", "signature", "reference_sha256",
          "data_manifest_sha256", "tokenizer_metadata_sha256", "checkpoint_sha256", "config_sha256"]
GROUP = ["model", "split", "decoding", "device", "signature", "reference_sha256",
         "data_manifest_sha256", "tokenizer_metadata_sha256", "config_sha256"]


def summarize(paths, output_dir):
    rows = [read_json(path) for path in paths]
    if not rows:
        raise ValueError("No run results supplied")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] not in ("dev", "test"):
            raise ValueError("Expected score output, not selection records")
        if row["split"] == "test" and not row.get("selection_sha256"):
            raise ValueError("Unfrozen test result")
        groups[tuple(row[key] for key in GROUP)].append(row)
    for members in groups.values():
        if len({r["seed"] for r in members}) != len(members):
            raise ValueError("Multiple results for the same model/seed/config; pass one preselected run per seed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Evaluation results", "", "BLEU is corpus SacreBLEU, case-sensitive, tokenize=13a.",
             "Groups keep split, decoding, device, config, data, tokenizer and metric identical.",
             "Standard deviation is the sample std; it is unavailable for one seed.", "",
             "| Model | Split | Decode | Seeds | BLEU mean | BLEU std | Device | Config |",
             "|---|---|---|---|---:|---:|---|---|"]
    for key, members in sorted(groups.items()):
        values = [r["bleu"] for r in members]
        std = f"{statistics.stdev(values):.2f}" if len(values) > 1 else "N/A"
        device = str(key[3]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {', '.join(str(r['seed']) for r in members)} | {statistics.mean(values):.2f} | {std} | {device} | {key[-1][:12]} |")
    lines += ["", "Per-run BLEU, parameter counts, training time and decoding speed: [runs.csv](runs.csv).", ""]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summarize(args.results, args.output_dir)
