import argparse
from pathlib import Path

import pytest

from src.data import (clean_splits, download, normalize, prepare, read_json, read_lines,
                      read_pairs, sha256, verify_raw, write_json, write_lines)
from src.evaluate import check_selection, corpus_bleu, freeze, score, write_once
from src.tokenizer import BOS, EOS, PAD, SharedTokenizer, train_bpe
from scripts.summarize_results import summarize


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    pairs = {
        "train": [("a little house is blue .", "một ngôi nhà màu xanh ."),
                  ("a little house is blue .", "một ngôi nhà màu xanh ."),
                  ("the house is very blue .", "ngôi nhà rất xanh ."),
                  ("", ""), ("held out dev sentence .", "câu phát triển .")],
        "tst2012": [("held out dev sentence .", "câu phát triển .")],
        "tst2013": [("unseen test sentence .", "câu kiểm tra .")]}
    for stem, rows in pairs.items():
        for i, lang in enumerate(("en", "vi")):
            write_lines(data / "raw" / f"{stem}.{lang}", [r[i] for r in rows])
    config = {"splits": {"train": "train", "dev": "tst2012", "test": "tst2013"},
              "expected_pairs": {"train": 5, "dev": 1, "test": 1},
              "sha256": {p.name: sha256(p) for p in (data / "raw").iterdir()}}
    prepare(data, tmp_path / "stats.json", config)
    train_bpe(data, merges=10)
    return data, config


def test_normalization_retains_case_accents_digits_punctuation():
    assert normalize("  Ca\u0301c\tNHÀ  123 !  ") == "Các NHÀ 123 !"


def test_alignment_rejects_unequal_counts(tmp_path):
    write_lines(tmp_path / "en", ["one", "two"])
    write_lines(tmp_path / "vi", ["một"])
    with pytest.raises(ValueError, match="Unaligned"):
        read_pairs(tmp_path / "en", tmp_path / "vi")


def test_cleaning_keeps_eval_membership_and_original_ids():
    raw = {"train": [("", "x"), ("a", "b"), (" a ", "b"), ("c", "d"), ("bad", "pair")],
           "dev": [("c", "d"), ("c", "d")], "test": [("e", "f")]}
    clean, ids, stats = clean_splits(raw, {"5": "manual mismatch"})
    assert clean["train"] == [("a", "b")]
    assert ids == [2]
    assert clean["dev"] == raw["dev"] and clean["test"] == raw["test"]
    assert sum(stats["removed_train"].values()) == 4
    assert stats["overlaps_after"]["train-dev"]["pairs"] == 0


def test_empty_eval_fails_instead_of_silent_deletion():
    with pytest.raises(ValueError, match="Empty dev"):
        clean_splits({"train": [], "dev": [("", "x")], "test": []})


def test_source_hashes_reject_swapped_splits(prepared):
    data, config = prepared
    (data / "raw/tst2012.en").write_bytes((data / "raw/tst2013.en").read_bytes())
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_raw(data / "raw", config)


def test_corrupt_download_never_becomes_raw_file(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr("src.data.urlopen", lambda *a, **k: io.BytesIO(b"<html>error</html>"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        download(tmp_path, {"sha256": {"train.en": "wrong"}, "base_url": "https://example.invalid/"})
    assert not (tmp_path / "train.en").exists()
    assert not (tmp_path / "train.en.part").exists()


def test_reference_bytes_preserved(prepared):
    data, _ = prepared
    assert (data / "references/dev.vi").read_bytes() == (data / "raw/tst2012.vi").read_bytes()
    assert read_lines(data / "processed/train.ids") == ["1", "3"]


def test_bpe_is_train_only_and_reproducible(prepared, tmp_path):
    data, config = prepared
    before = {name: sha256(data / "bpe" / name) for name in ("codes.bpe", "vocab.json")}
    # Alter both held-out sets, preserving train; neither codes nor vocab may change.
    for stem in ("tst2012", "tst2013"):
        for lang in ("en", "vi"):
            path = data / "raw" / f"{stem}.{lang}"
            write_lines(path, ["COMPLETELY NEW HELDOUT CHARACTERS Ω Ж"])
            config["sha256"][path.name] = sha256(path)
    # Original train contains the former dev row; exclude it to keep cleaned train fixed.
    prepare(data, tmp_path / "stats2.json", config, {"5": "keep training fixed in test"})
    train_bpe(data, merges=10)
    assert {name: sha256(data / "bpe" / name) for name in before} == before


def test_tokenizer_roundtrip_and_padding(prepared):
    data, _ = prepared
    tok = SharedTokenizer(data / "bpe")
    text = "một ngôi nhà màu xanh ."
    ids = tok.encode(text, add_bos=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert tok.decode(ids + [PAD, 1000000]) == text
    with pytest.raises(ValueError, match="outside"):
        tok.decode([-1])


def test_batch_shift_and_masks(prepared):
    pytest.importorskip("torch")
    from src.dataset import TranslationDataset, collate_batch
    data, _ = prepared
    dataset = TranslationDataset(data)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["tgt_in"][:, 0].tolist() == [BOS, BOS]
    for i, example in enumerate([dataset[0], dataset[1]]):
        n = len(example["tgt"]) - 1
        assert batch["tgt_in"][i, :n].tolist() == example["tgt"][:-1]
        assert batch["tgt_out"][i, :n].tolist() == example["tgt"][1:]
        assert batch["tgt_out"][i, n:].eq(PAD).all()
    assert batch["tgt_causal_mask"][0, 1] and not batch["tgt_causal_mask"][1, 0]
    assert batch["src_padding_mask"].equal(batch["src"].eq(PAD))


def test_bleu_identity_bpe_and_case():
    ref = ["This is a small house ."]
    result = corpus_bleu(["This is a sm@@ all house ."], ref, bpe=True)
    assert result["bleu"] == pytest.approx(100)
    assert "case:mixed" in result["signature"] and "tok:13a" in result["signature"]
    assert corpus_bleu([ref[0].lower()], ref)["bleu"] < 100
    assert corpus_bleu([""], ref)["bleu"] == 0
    with pytest.raises(ValueError, match="Expected"):
        corpus_bleu([], ref)
    with pytest.raises(ValueError, match="continuation"):
        corpus_bleu(["sm@@ all"], ref)
    with pytest.raises(ValueError, match="special"):
        corpus_bleu(["<bos> This is a small house ."], ref)


def score_args(data, tmp_path, split="dev"):
    checkpoint = tmp_path / "fake-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic test fixture, not a trained model")
    config = tmp_path / "run.json"
    write_json(config, {"layers": 2})
    hyp = tmp_path / f"{split}.hyp"
    hyp.write_bytes((data / "references" / f"{split}.vi").read_bytes())
    return argparse.Namespace(data_dir=data, model="seq2seq", seed=42, checkpoint=checkpoint,
                              config=config, decoding="greedy", split=split, selection=None,
                              parameters=100, train_seconds=5., decode_seconds=1., device="cpu",
                              hyp=hyp, hyp_format="text", output=tmp_path / f"{split}.json")


def test_score_freeze_test_and_summary(prepared, tmp_path):
    data, _ = prepared
    dev = score_args(data, tmp_path)
    score(dev)
    selection_path = tmp_path / "selection.json"
    freeze([dev.output], selection_path)
    test = score_args(data, tmp_path, "test")
    with pytest.raises(ValueError, match="Test is locked"):
        score(test)
    test.selection = selection_path
    result = score(test)
    assert result["selection_sha256"] == sha256(selection_path)
    with pytest.raises(ValueError, match="using test"):
        freeze([test.output], tmp_path / "bad-selection.json")
    summarize([test.output], tmp_path / "summary")
    assert "N/A" in (tmp_path / "summary/summary.md").read_text()
    with pytest.raises(ValueError, match="Multiple results"):
        summarize([test.output, test.output], tmp_path / "summary")
    test.checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        score(test)


@pytest.mark.parametrize("field", ["config_sha256", "seed", "model", "decoding", "data_manifest_sha256", "tokenizer_metadata_sha256"])
def test_selection_rejects_changed_identity(field):
    from src.evaluate import IDENTITY_FIELDS
    identity = dict.fromkeys(IDENTITY_FIELDS, "same")
    selection = {"selected_on": "dev", **identity, field: "changed"}
    with pytest.raises(ValueError, match=field):
        check_selection(selection, identity)


def test_run_records_cannot_be_overwritten(tmp_path):
    path = tmp_path / "record.json"
    write_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_once(path, {"original": False})


def test_changed_reference_is_rejected(prepared, tmp_path):
    data, _ = prepared
    args = score_args(data, tmp_path)
    write_lines(data / "references/dev.vi", ["tampered"])
    with pytest.raises(ValueError, match="Data changed"):
        score(args)


def test_freeze_selects_best_dev_not_input_order(tmp_path):
    from src.evaluate import IDENTITY_FIELDS
    base = {**dict.fromkeys(IDENTITY_FIELDS, "same"), "split": "dev", "signature": "metric", "reference_sha256": "ref"}
    write_json(tmp_path / "a.json", {**base, "bleu": 12., "checkpoint_sha256": "a"})
    write_json(tmp_path / "b.json", {**base, "bleu": 20., "checkpoint_sha256": "b"})
    result = freeze([tmp_path / "a.json", tmp_path / "b.json"], tmp_path / "selection.json")
    assert result["checkpoint_sha256"] == "b" and result["dev_bleu"] == 20.


@pytest.mark.parametrize("field", ["seed", "reference_sha256", "signature"])
def test_freeze_rejects_incomparable_dev_runs(tmp_path, field):
    from src.evaluate import IDENTITY_FIELDS
    base = {**dict.fromkeys(IDENTITY_FIELDS, "same"), "split": "dev", "signature": "metric", "reference_sha256": "ref", "bleu": 20.}
    write_json(tmp_path / "a.json", base)
    write_json(tmp_path / "b.json", {**base, field: "different"})
    with pytest.raises(ValueError, match=field):
        freeze([tmp_path / "a.json", tmp_path / "b.json"], tmp_path / "selection.json")
