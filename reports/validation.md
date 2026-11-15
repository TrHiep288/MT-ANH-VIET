# Validation of the data/evaluation implementation

- Python 3.11.4, Windows; PyTorch 2.14.0+cpu for batch checks.
- `python -m pytest -q`: **24 passed**.
- Downloaded all six files from the pinned mirror; all original SHA-256 values
  and split row counts verified. Stanford was unavailable from this environment.
- Ran preparation and all 8,000 BPE merges on the full corpus, not a sample.
- Verified every prepared-data and BPE artifact against its manifest.
- Loaded all 1,553 dev pairs through `TranslationDataset`; collated three actual
  examples including the first and last, producing a `[3, 104]` source batch.
- Repeated BPE learning in a separate process with `PYTHONHASHSEED=17`; the merge
  rules remained byte-identical. JSON/text artifacts use explicit LF newlines.
- `git diff --check` passed.

Final prepared-data manifest SHA-256:
`d85170344cbfe46a4e9f460a799b227665b4a90d21fdb8cb33148754c10eb6cf`.

BPE codes SHA-256:
`f04c10d97096b5e0d6ca99ec434b14baf3fe5e344084d86061457b423f16b022`.

See the statistics JSON files for remaining artifact hashes. Tests use synthetic
checkpoint bytes only inside temporary fixtures to exercise the selection guard.
No trained-model BLEU, runtime comparison, or attention analysis is claimed.
