"""Shared batch contract for Hoang's LSTM and Hiep's Transformer (PyTorch)."""
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data import read_pairs, read_lines, sha256
from src.tokenizer import PAD, SharedTokenizer


class TranslationDataset(Dataset):
    def __init__(self, data_dir="data", split="train"):
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev or test")
        data_dir = Path(data_dir)
        self.tokenizer = SharedTokenizer(data_dir / "bpe")
        if sha256(data_dir / "manifest.json") != self.tokenizer.metadata["data_manifest_sha256"]:
            raise ValueError("Tokenizer was trained on different prepared data")
        from src.data import read_json
        manifest = read_json(data_dir / "manifest.json")
        for suffix in ("en", "vi", "ids"):
            name = f"processed/{split}.{suffix}"
            if sha256(data_dir / name) != manifest["files"][name]:
                raise ValueError(f"Prepared data changed: {name}")
        pairs = read_pairs(data_dir / "processed" / f"{split}.en", data_dir / "processed" / f"{split}.vi")
        self.ids = [int(i) for i in read_lines(data_dir / "processed" / f"{split}.ids")]
        self.examples = [(self.tokenizer.encode(en), self.tokenizer.encode(vi, add_bos=True))
                         for en, vi in pairs]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        source, target = self.examples[index]
        return {"id": self.ids[index], "src": source, "tgt": target}


def collate_batch(examples):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    src = pad_sequence([torch.tensor(e["src"], dtype=torch.long) for e in examples], batch_first=True, padding_value=PAD)
    target = [torch.tensor(e["tgt"], dtype=torch.long) for e in examples]
    tgt_in = pad_sequence([t[:-1] for t in target], batch_first=True, padding_value=PAD)
    tgt_out = pad_sequence([t[1:] for t in target], batch_first=True, padding_value=PAD)
    return {"ids": torch.tensor([e["id"] for e in examples]), "src": src,
            "src_lengths": torch.tensor([len(e["src"]) for e in examples]),
            "tgt_in": tgt_in, "tgt_out": tgt_out,
            "src_padding_mask": src.eq(PAD), "tgt_padding_mask": tgt_in.eq(PAD),
            "tgt_causal_mask": torch.ones(tgt_in.size(1), tgt_in.size(1), dtype=torch.bool).triu(1)}
