# ERA baseline with gpt-oss-120b

This branch contains only the ERA baseline: the ERA tree search ([google-research/era](https://github.com/google-research/era)) with its Gemini client replaced by an OpenAI-compatible API client for gpt-oss-120b. The search itself is upstream's, unmodified.

```
baselines/era/        the ERA port: pipeline, tests, smoke test, toy task  -> baselines/era/README.md
datasets/zapbench/    example task files: ZAPBench starter code + task description
```

History: the first commit vendors the upstream files byte-for-byte (`baselines/era/UPSTREAM.md`); the second adds the port. `git diff $(git rev-list --max-parents=0 HEAD) -- baselines/era/llm.py` shows exactly what changed in the client, and `futs.py` is unchanged.

Not included: SimpleTES, the ZAPBench evaluator and data preparation, and cluster launch scripts. The ERA README still compares the setup with the SimpleTES ZAPBench runs.

The vendored ERA files are Apache-2.0 (`baselines/era/LICENSE`).
