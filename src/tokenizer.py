"""One shared BPE and vocabulary, fitted on English + Vietnamese train only."""
import io
from collections import Counter
from pathlib import Path
from importlib.metadata import version

from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe

from src.data import normalize, read_json, read_lines, sha256, verify_manifest, write_json, write_lines

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)


def unmerge(text):
    text = text.replace("@@ ", "")
    return text[:-2] if text.endswith("@@") else text


def train_bpe(data_dir, merges=8000):
    if merges < 1:
        raise ValueError("merges must be positive")
    data_dir = Path(data_dir)
    data_hash = verify_manifest(data_dir)
    out = data_dir / "bpe"
    out.mkdir(parents=True, exist_ok=True)
    train_lines = [line for lang in ("en", "vi")
                   for line in read_lines(data_dir / "processed" / f"train.{lang}")]
    if not train_lines:
        raise ValueError("Cannot learn BPE from empty training data")
    if any(token in SPECIAL_TOKENS for line in train_lines for token in line.split()):
        raise ValueError("Training text contains reserved special tokens")
    with (out / "codes.bpe").open("w", encoding="utf-8", newline="\n") as codes:
        learn_bpe(io.StringIO("\n".join(train_lines) + "\n"), codes, num_symbols=merges,
                  min_frequency=2, num_workers=1)
    with (out / "codes.bpe").open(encoding="utf-8") as codes:
        bpe = BPE(codes)
    counts = Counter(token for line in train_lines for token in bpe.process_line(line).split())
    vocab = SPECIAL_TOKENS + sorted(counts, key=lambda token: (-counts[token], token))
    write_json(out / "vocab.json", vocab)
    stats = {}
    for split in ("train", "dev", "test"):
        stats[split] = {}
        for lang in ("en", "vi"):
            encoded = [bpe.process_line(line) for line in read_lines(data_dir / "processed" / f"{split}.{lang}")]
            write_lines(out / f"{split}.{lang}", encoded)
            tokens = [t for line in encoded for t in line.split()]
            stats[split][lang] = {"tokens": len(tokens), "unknown_tokens": sum(t not in counts for t in tokens)}
    metadata = {"schema_version": 1, "algorithm": "subword-nmt BPE", "version": version("subword-nmt"),
                "requested_merges": merges, "learned_merges": len(read_lines(out / "codes.bpe")) - 1,
                "training_splits": ["train.en", "train.vi"], "data_manifest_sha256": data_hash,
                "vocab_size": len(vocab), "special_ids": dict(zip(SPECIAL_TOKENS, range(4))),
                "files": {p.name: sha256(p) for p in sorted(out.iterdir())
                          if p.name in {"codes.bpe", "vocab.json"} or p.suffix in {".en", ".vi"}},
                "token_statistics": stats}
    write_json(out / "metadata.json", metadata)
    return metadata


class SharedTokenizer:
    pad_id, unk_id, bos_id, eos_id = PAD, UNK, BOS, EOS

    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = read_json(directory / "metadata.json")
        for name in ("codes.bpe", "vocab.json"):
            if sha256(directory / name) != self.metadata["files"][name]:
                raise ValueError(f"Tokenizer checksum mismatch: {name}")
        self.tokens = read_json(directory / "vocab.json")
        if self.tokens[:4] != SPECIAL_TOKENS or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Invalid vocabulary")
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        with (directory / "codes.bpe").open(encoding="utf-8") as codes:
            self.bpe = BPE(codes)

    def __len__(self):
        return len(self.tokens)

    def encode(self, text, add_bos=False, add_eos=True):
        tokens = self.bpe.process_line(normalize(text)).split()
        # A literal special token in text is unknown, never a control ID.
        ids = [self.token_to_id.get(t, UNK) if t not in SPECIAL_TOKENS else UNK for t in tokens]
        return ([BOS] if add_bos else []) + ids + ([EOS] if add_eos else [])

    def decode(self, ids):
        tokens = []
        for index in ids:
            index = int(index)
            if index == EOS:
                break
            if index in (PAD, BOS):
                continue
            if not 0 <= index < len(self.tokens):
                raise ValueError(f"Token ID outside vocabulary: {index}")
            tokens.append(self.tokens[index])
        return unmerge(" ".join(tokens))
