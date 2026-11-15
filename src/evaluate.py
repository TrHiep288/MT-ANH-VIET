"""SacreBLEU scoring and dev-only selection records, independent of model code."""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import BLEU

from src.data import ROOT, read_json, read_lines, sha256, verify_manifest
from src.tokenizer import unmerge


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def corpus_bleu(hypotheses, references, bpe=False):
    if len(hypotheses) != len(references) or not references:
        raise ValueError(f"Expected {len(references)} hypotheses, got {len(hypotheses)}")
    if any(t in {"<pad>", "<bos>", "<eos>"} for h in hypotheses for t in h.split()):
        raise ValueError("Decode special IDs with SharedTokenizer.decode before scoring")
    if bpe:
        hypotheses = [unmerge(h) for h in hypotheses]
    elif any(t.endswith("@@") for h in hypotheses for t in h.split()):
        raise ValueError("Found BPE continuation markers; use --hyp-format bpe")
    metric = BLEU(tokenize="13a", lowercase=False, smooth_method="exp", effective_order=False)
    result = metric.corpus_score(hypotheses, [references])
    return {"bleu": result.score, "signature": str(metric.get_signature()),
            "sacrebleu_version": sacrebleu.__version__, "precisions": result.precisions,
            "brevity_penalty": result.bp, "hyp_tokens": result.sys_len, "ref_tokens": result.ref_len}


IDENTITY_FIELDS = ("model", "seed", "checkpoint_sha256", "config_sha256", "decoding",
                   "data_manifest_sha256", "tokenizer_metadata_sha256")


def check_selection(selection, identity):
    if selection.get("selected_on") != "dev":
        raise ValueError("Test scoring requires a selection made exclusively on dev")
    for field in IDENTITY_FIELDS:
        if selection.get(field) != identity.get(field):
            raise ValueError(f"Frozen selection mismatch: {field}")


def score(args):
    data = Path(args.data_dir)
    data_hash = verify_manifest(data)
    metadata = read_json(data / "bpe/metadata.json")
    if metadata["data_manifest_sha256"] != data_hash:
        raise ValueError("BPE was fitted on different prepared data")
    for name, expected in metadata["files"].items():
        if sha256(data / "bpe" / name) != expected:
            raise ValueError(f"BPE artifact changed: {name}")
    # Require a real JSON config and hash its exact bytes; do not execute it.
    if not isinstance(read_json(args.config), dict):
        raise ValueError("Run config must be a JSON object")
    identity = {"model": args.model, "seed": args.seed, "checkpoint_sha256": sha256(args.checkpoint),
                "config_sha256": sha256(args.config), "decoding": args.decoding,
                "data_manifest_sha256": data_hash,
                "tokenizer_metadata_sha256": sha256(data / "bpe/metadata.json")}
    selection_hash = None
    if args.split == "test":
        if not args.selection:
            raise ValueError("Test is locked: first run freeze using dev results and pass --selection")
        check_selection(read_json(args.selection), identity)
        selection_hash = sha256(args.selection)
    elif args.selection:
        raise ValueError("--selection is only used for final test evaluation")
    for name in ("train_seconds", "decode_seconds", "parameters"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    hypotheses = read_lines(args.hyp)
    reference_path = data / "references" / f"{args.split}.vi"
    result = {"schema_version": 1, **identity, "split": args.split,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "hyp_sha256": sha256(args.hyp), "hyp_format": args.hyp_format,
              "reference_sha256": sha256(reference_path), "selection_sha256": selection_hash,
              "sentences": len(hypotheses), "device": args.device,
              "parameters": args.parameters, "train_seconds": args.train_seconds,
              "decode_seconds": args.decode_seconds,
              "sentences_per_second": len(hypotheses) / args.decode_seconds if args.decode_seconds else None,
              **corpus_bleu(hypotheses, read_lines(reference_path), args.hyp_format == "bpe")}
    write_once(args.output, result)
    return result


def freeze(dev_results, output):
    if not dev_results:
        raise ValueError("Provide at least one dev result")
    candidates = [(Path(path), read_json(path)) for path in dev_results]
    for _, result in candidates:
        if result.get("split") != "dev":
            raise ValueError("Cannot select a checkpoint or configuration using test results")
        if not math.isfinite(result["bleu"]) or not 0 <= result["bleu"] <= 100 + 1e-9:
            raise ValueError("Invalid BLEU result")
    # Tune architectures independently, once per seed, on the same dev corpus/metric.
    for field in ("model", "seed", "data_manifest_sha256", "tokenizer_metadata_sha256", "reference_sha256", "signature"):
        if len({r[field] for _, r in candidates}) != 1:
            raise ValueError(f"Incomparable dev candidates: {field}")
    path, best = max(candidates, key=lambda item: item[1]["bleu"])
    selection = {"schema_version": 1, "selected_on": "dev", "criterion": "maximum corpus BLEU; first input wins ties",
                 **{key: best[key] for key in IDENTITY_FIELDS}, "dev_bleu": best["bleu"],
                 "created_utc": datetime.now(timezone.utc).isoformat(),
                 "selected_dev_result": path.name,
                 "selected_dev_result_sha256": sha256(path),
                 "candidate_results": [{"file": p.name, "sha256": sha256(p), "dev_bleu": r["bleu"]} for p, r in candidates]}
    write_once(output, selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--data-dir", type=Path, default=ROOT / "data")
    scoring.add_argument("--split", choices=("dev", "test"), required=True)
    scoring.add_argument("--hyp", type=Path, required=True)
    scoring.add_argument("--hyp-format", choices=("text", "bpe"), default="text")
    scoring.add_argument("--model", choices=("seq2seq", "transformer"), required=True)
    scoring.add_argument("--seed", type=int, required=True)
    scoring.add_argument("--checkpoint", type=Path, required=True)
    scoring.add_argument("--config", type=Path, required=True)
    scoring.add_argument("--decoding", choices=("greedy", "beam4"), default="greedy")
    scoring.add_argument("--selection", type=Path)
    scoring.add_argument("--parameters", type=int)
    scoring.add_argument("--train-seconds", type=float)
    scoring.add_argument("--decode-seconds", type=float)
    scoring.add_argument("--device", default="not recorded")
    scoring.add_argument("--output", type=Path, required=True)
    selection = sub.add_parser("freeze")
    selection.add_argument("--dev-results", type=Path, nargs="+", required=True)
    selection.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args) if args.command == "score" else freeze(args.dev_results, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
