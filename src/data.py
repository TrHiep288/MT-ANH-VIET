"""Reproducible IWSLT15 download, alignment checks and train-only cleaning."""
import hashlib
import io
import json
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": "train", "dev": "tst2012", "test": "tst2013"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_lines(path):
    # Only physical newlines delimit examples (not Unicode U+2028 etc.).
    with Path(path).open(encoding="utf-8") as stream:
        return [line.rstrip("\r\n") for line in stream]


def write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for line in lines:
            stream.write(line + "\n")


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_pairs(source, target):
    en, vi = read_lines(source), read_lines(target)
    if len(en) != len(vi):
        raise ValueError(f"Unaligned files: {source} ({len(en)}) vs {target} ({len(vi)})")
    return list(zip(en, vi))


def verify_raw(raw_dir, config):
    raw_dir = Path(raw_dir)
    for name, expected in config["sha256"].items():
        if sha256(raw_dir / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}; do not use modified/wrong-split data")
    for split, stem in config["splits"].items():
        pairs = read_pairs(raw_dir / f"{stem}.en", raw_dir / f"{stem}.vi")
        if len(pairs) != config["expected_pairs"][split]:
            raise ValueError(f"Unexpected pair count for {split}: {len(pairs)}")


def download(raw_dir, config, source="stanford"):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source == "mirror":
        for stem, archive in config["mirror_archives"].items():
            names = [f"{stem}.{lang}" for lang in ("en", "vi")]
            if all((raw_dir / name).exists() and sha256(raw_dir / name) == config["sha256"][name] for name in names):
                continue
            with urlopen(config["mirror_base_url"] + archive, timeout=60) as response:
                payload = response.read()
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
                for name in names:
                    # Never extract arbitrary paths, links or executable archive members.
                    matches = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == name]
                    if len(matches) != 1:
                        raise ValueError(f"Missing or ambiguous mirror member: {name}")
                    content = tar.extractfile(matches[0]).read()
                    if hashlib.sha256(content).hexdigest() != config["sha256"][name]:
                        raise ValueError(f"Mirror differs from original Stanford bytes: {name}")
                    path = raw_dir / name
                    if path.exists() and sha256(path) != config["sha256"][name]:
                        raise ValueError(f"Existing file differs: {path}; move it aside first")
                    path.write_bytes(content)
                    print(f"Verified {name} from pinned mirror", flush=True)
        verify_raw(raw_dir, config)
        write_json(raw_dir.parent / "download_receipt.json", {"source": config["mirror_base_url"], "sha256": config["sha256"]})
        return
    if source != "stanford":
        raise ValueError("source must be stanford or mirror")
    for name, expected in config["sha256"].items():
        path = raw_dir / name
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"Existing {path} has wrong checksum; move it aside before retrying")
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(config["base_url"] + name, timeout=60) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if sha256(temporary) != expected:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Verified {name}", flush=True)
    verify_raw(raw_dir, config)
    write_json(raw_dir.parent / "download_receipt.json", {"source": config["base_url"], "sha256": config["sha256"]})


def length_stats(lines):
    lengths = sorted(len(line.split()) for line in lines)
    if not lengths:
        return {"sentences": 0}
    return {"sentences": len(lengths), "tokens": sum(lengths),
            "mean": round(sum(lengths) / len(lengths), 3),
            "min": lengths[0], "max": lengths[-1],
            "p50": lengths[(len(lengths) - 1) // 2],
            "p95": lengths[int((len(lengths) - 1) * .95)]}


def overlaps(left, right):
    return {"pairs": len(set(left) & set(right)),
            "en": len({p[0] for p in left} & {p[0] for p in right}),
            "vi": len({p[1] for p in left} & {p[1] for p in right})}


def clean_splits(raw, exclusions=None):
    exclusions = exclusions or {}
    if any(int(i) < 1 or int(i) > len(raw["train"]) for i in exclusions):
        raise ValueError("Manual exclusion outside original training line range")
    normalized = {split: [(normalize(en), normalize(vi)) for en, vi in pairs]
                  for split, pairs in raw.items()}
    for split in ("dev", "test"):
        if any(not en or not vi for en, vi in normalized[split]):
            raise ValueError(f"Empty {split} pair; refusing to alter evaluation membership")
    held_out = set(normalized["dev"]) | set(normalized["test"])
    seen, train, kept_ids = set(), [], []
    removed = Counter()
    for line_id, pair in enumerate(normalized["train"], 1):
        if str(line_id) in exclusions:
            removed["manually_verified_misalignment"] += 1
        elif not all(pair):
            removed["empty"] += 1
        elif pair in held_out:
            removed["overlap_with_dev_or_test"] += 1
        elif pair in seen:
            removed["duplicate_train_pair"] += 1
        else:
            seen.add(pair)
            train.append(pair)
            kept_ids.append(line_id)
    cleaned = {**normalized, "train": train}
    report = {"raw_pairs": {s: len(p) for s, p in raw.items()},
              "clean_pairs": {s: len(p) for s, p in cleaned.items()},
              "removed_train": dict(removed),
              "manual_exclusions": exclusions,
              "normalization_changed_pairs": {s: sum(a != b for a, b in zip(raw[s], normalized[s])) for s in raw},
              "duplicates_before": {s: len(p) - len(set(p)) for s, p in normalized.items()},
              "overlaps_before": {f"{a}-{b}": overlaps(normalized[a], normalized[b])
                                  for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))},
              "overlaps_after": {f"{a}-{b}": overlaps(cleaned[a], cleaned[b])
                                 for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))},
              "lengths_whitespace_tokens": {s: {lang: length_stats([p[i] for p in pairs])
                                               for i, lang in enumerate(("en", "vi"))}
                                           for s, pairs in cleaned.items()}}
    return cleaned, kept_ids, report


def prepare(data_dir, report_path, config, exclusions=None):
    data_dir = Path(data_dir)
    verify_raw(data_dir / "raw", config)
    raw = {s: read_pairs(data_dir / "raw" / f"{stem}.en", data_dir / "raw" / f"{stem}.vi")
           for s, stem in config["splits"].items()}
    cleaned, ids, report = clean_splits(raw, exclusions)
    for split, pairs in cleaned.items():
        for i, lang in enumerate(("en", "vi")):
            write_lines(data_dir / "processed" / f"{split}.{lang}", (p[i] for p in pairs))
        write_lines(data_dir / "processed" / f"{split}.ids", map(str, ids if split == "train" else range(1, len(pairs) + 1)))
    # Preserve reference bytes exactly as distributed, independently of model inputs.
    for split in ("dev", "test"):
        path = data_dir / "references" / f"{split}.vi"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((data_dir / "raw" / f"{SPLITS[split]}.vi").read_bytes())
    files = {str(p.relative_to(data_dir)).replace("\\", "/"): sha256(p)
             for folder in ("raw", "processed", "references")
             for p in sorted((data_dir / folder).iterdir()) if p.is_file()}
    manifest = {"schema_version": 1, "splits": config["splits"], "files": files,
                "manual_exclusions": exclusions or {},
                "policy": "NFC + whitespace; remove documented misalignment, empty/duplicate/held-out exact pairs from train only; preserve eval order and raw reference bytes"}
    write_json(data_dir / "manifest.json", manifest)
    report["manifest_sha256"] = sha256(data_dir / "manifest.json")
    report["alignment_scope"] = "Official file hashes, equal counts, and original row order verified; semantic alignment is not automatically proven."
    write_json(report_path, report)
    return report


def verify_manifest(data_dir):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"Data changed since preparation: {name}")
    return sha256(data_dir / "manifest.json")
