# Upstream provenance

Vendored from **google-research/era**, commit
`b836730b5c000526af95116b1d0e2c60c8cf0a10` (HEAD of `main` when fetched,
2026-09-28), directory `implementation/`. Apache-2.0 (see `LICENSE`).

The first commit that adds `baselines/era/` contains these files **byte-for-byte**
as they are in that upstream commit (LF line endings, i.e. the git blobs):

| file here            | upstream path                    | sha256 of upstream blob |
|----------------------|----------------------------------|-------------------------|
| `futs.py`            | `implementation/futs.py`         | `5cc827f3d609105c1bd7b1103e4c33e393043c6db882bd1f44b9e4ec4301b814` |
| `futs_test.py`       | `implementation/futs_test.py`    | `720ba65fe74ed859c71d8e872cf7d23d0769fcee08c9ec52994d05c13159e39f` |
| `llm.py`             | `implementation/llm.py`          | `6a344ab43e747669d487bf051856ca6cf4fd7af41b46e652bf1f63df418d9c16` |
| `UPSTREAM_README.md` | `implementation/README.md`       | `47df8b43bfca9d4cbf7ea2dfe673b8ae90db62bca375b8e4842ae54bdf99bc57` |
| `LICENSE`            | `LICENSE`                        | `58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd` |

Not vendored (and why):

| upstream file | sha256 | reason |
|---|---|---|
| `implementation/sandbox.py` | `cd2b3c86…c7ef7844` | a stub whose `run()` raises `NotImplementedError("Must provide a sandbox…")`; ERA leaves the sandbox to the user. We supply the SimpleTES evaluation protocol instead (see `README.md`). |
| `implementation/playground_s3e1.py` | `e20beec8…5e3e04fa` | Kaggle S3E1 example. Its `PlaygroundGenerator` prompt is the template `run_era.py` generalizes (quoted there). As released it cannot run: it calls `Sandbox(timeout_seconds=60)` and `sandbox.run(prog, "wrapper", None)`, neither of which matches the stub's signature. |
| `implementation/experiment_pipeline.ipynb`, `notebooks/`, `data/` | — | examples / datasets, not ERA logic. |

To verify that the tree-search algorithm is unmodified at any later commit:

    git diff <first-era-commit> -- baselines/era/futs.py     # expect: empty
    sha256sum baselines/era/futs.py                           # expect: 5cc827f3…
