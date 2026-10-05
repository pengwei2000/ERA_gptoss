# ERA baseline (Flat UCB Tree Search) with gpt-oss-120b

ERA ([Aygün et al., 2025](https://arxiv.org/abs/2509.06503),
[google-research/era](https://github.com/google-research/era)) ported so it can be
compared with SimpleTES on the same footing. The LLM is gpt-oss-120b over an
OpenAI-compatible API, and candidates run in SimpleTES's sandbox. ERA's search
is untouched: `futs.py` is byte-identical to upstream (see `UPSTREAM.md`), and the
pipeline runs upstream's own `futs.search` loop, strictly one node at a time.

```
baselines/era/
  futs.py, futs_test.py    upstream, unmodified (tree search + its tests)
  llm.py                   upstream file; Gemini client -> OpenAI-compatible client
  run_era.py               task-agnostic pipeline (CLI): prompt, sandbox, logging, resume
  smoke_test.py            task-agnostic smoke test (offline; --online hits the real API)
  toy_task/                seconds-long task used by smoke_test.py
  tests/                   unit tests (stdlib unittest)

datasets/zapbench/         ZAPBench starter code + task description (example task; see below)
```

## What changed vs upstream ERA

| upstream piece | here | detail |
|---|---|---|
| `futs.py`: rank scores, PUCT, visit backprop, `search` | **unchanged**, and `futs.search` is the search loop | `run_era.run_search` calls `futs.search` as it is. Each node is selected only after the previous node has been generated and evaluated, as in upstream. |
| `futs_test.py` | **unchanged** | needs `absl-py` |
| `llm.py` `GeminiLLM` (google-genai SDK) | **replaced** by `OpenAICompatLLM` (stdlib HTTP) | The prompt preamble and markdown-fence cleanup are copied verbatim. Upstream sent no sampling parameters; we send temperature 1.0, top_p 1.0 and reasoning_effort high, plus `max_tokens` = min(49152, 49152 − prompt tokens), so prompt + completion fits the same 49152-token context SimpleTES's vLLM served. The prompt size is estimated from its length, with the tokens per character learned from the server's `usage.prompt_tokens` (the highest ratio seen). An estimate can still be too low, so a reply whose reported prompt + completion exceeds 49152 tokens is counted as truncated, as a server with that context would have cut it: `finish_reason=length`, no program, and `over_context_tokens` in `nodes.jsonl`. A well-formed reply with empty content, e.g. `finish_reason=length` after the reasoning used the whole budget, becomes a failed node that counts toward `--max-nodes`; upstream would crash on `response.text=None` and end the search. The retry policy is new (next row). |
| `llm.py` retry: HTTP 429 only, 5 tries at 5 s·2^k (~75 s); any other error raises and ends the search | **replaced** | We retry network errors, timeouts, 408/409/425/429, any 5xx and malformed 200 responses, including fields of the wrong type. The default is 10 attempts, waiting 60 s doubling to a 300 s cap (~40 min total), and we honour numeric `Retry-After`. Other 4xx codes raise immediately, and so do redirects, which are never followed so the key is never re-sent elsewhere. If retries run out, the run exits with code 3; `--resume` continues it, so a job script can restart on code 3 (our ZAPBench job script waits 30 min first). A rejected key, model or parameter exits with code 4 and stays stopped. |
| `sandbox.py` (a stub that raises `NotImplementedError`) | **not used** | `run_era.EvaluatorExecutor` is SimpleTES's evaluation protocol. Its runner script is byte-identical to `simpletes/evaluator.py::_EVAL_RUNNER`, and it keeps the hard timeout with process-group kill. Deliberate differences: it removes `*API_KEY*` variables from the candidate's environment; if the runner's last stdout line is JSON but not a dict (e.g. `evaluate()` returned `None`), the evaluation fails instead of falling back to an earlier JSON line. As in SimpleTES, after a timeout it waits for the killed runner, not for pipes a surviving descendant may hold. |
| `playground_s3e1.py` `PlaygroundGenerator` prompt (Kaggle-specific) | **generalized** in `run_era.ERA_PROMPT` | We keep the parent-score line, the parent-code block, "Please generate a NEW, IMPROVED …" and "Provide the full, runnable code including imports". The task-specific lines are replaced by the task's instruction file. The score is shown as the raw combined_score formatted `%.6g`, instead of RMSE formatted `%.5f`. |
| tree kept in memory, only the best is returned | **added** around `futs.search`, which is not modified | Every node is logged to `nodes.jsonl`, `programs/` and `best_program.py`, through wrappers around `generate_fn`/`execute_fn`, the same way upstream's `playground_s3e1.py` tracks its search. |
| no resume: `futs.search` always starts from the root | **added** by replay | `--resume` calls `futs.search` from the root again and feeds it the logged nodes instead of calling the LLM and the evaluator. `futs.search` is deterministic given those scores, so it rebuilds exactly the same tree; each replayed step checks that the recorded parent was selected. Then it continues live. |

Where the ERA paper and its code disagree, we follow the code: root visits start at 0 (Algorithm 1 says 1), and a one-node tree has rank 0.5 (paper: 1). The paper also says output logs inform the next step, but the released generator passes only the parent's code and score.

## Fairness vs the SimpleTES ZAPBench runs

| | SimpleTES | ERA baseline |
|---|---|---|
| LLM | gpt-oss-120b, self-hosted vLLM | gpt-oss-120b, AkashML API |
| sampling | temperature 1.0, top_p 1.0 (vLLM default), reasoning high | same |
| tokens per call | 49152-token context (`--max-model-len`): prompt + reasoning + answer ≤ 49152, with reasoning forced to ≤ 33552 − prompt and the answer to ≤ 15536, and prompts of ≥ 33552 tokens rejected (`vllm_token_forcing`) | prompt + completion ≤ 49152 (`--context-window`); no forced reasoning/answer split, which a hosted chat API cannot do |
| LLM HTTP timeout | 14400 s (self-hosted vLLM, queueing) | 1800 s per attempt, then the retry policy |
| budget | 1600 generations per run | 1600 nodes (`--max-nodes`, root excluded as in `futs.search`) |
| evaluator | the ZAPBench evaluator (not in this branch) through the SimpleTES runner | same evaluator, same runner |
| worker | one SLURM job per evaluation: A4000, 8 CPU, 96 GB, 4.5 h | same |
| timeouts | 14400 s in the worker + 14400 s `--eval-timeout` (includes SLURM queue time) | same |
| eval concurrency | 12–14 (SimpleTES-API run: 11) | 1: upstream ERA is serial by design |
| root | run A: `init_program_era.py`; run B: run A's best | `init_program_era.py` |
| instruction | `zapbench.txt` | `zapbench.txt`, inside ERA's prompt template |

Differences to report alongside results:
- **No reasoning/answer split.** Both runs cap each call at a 49152-token context, but only SimpleTES reserved 15536 tokens for the answer. An ERA reply can use the whole budget on reasoning; it then comes back empty (`finish_reason=length`) and becomes a failed node. Count these with `finish_reason` in `nodes.jsonl` and report them.
- **Server-side model.** The model runs on AkashML's servers rather than our vLLM, so we can't verify the server's exact build, such as quantization.
- **Wall-clock.** ERA is serial by design, as in upstream: one LLM call and one A4000 evaluation at a time. The SimpleTES-API run keeps up to 11 evaluations running at once. The node budget (1600) is the same, but ERA takes months where SimpleTES takes days: at ~1.5–4 h per node, 1600 nodes take ~100–270 days (ERA reports 192 min/node on a T4). Report the node budget and the wall-clock of both.
- **GPU.** We use an A4000 where the ERA paper used a T4 (both 16 GB). The budgets come from our SimpleTES setup, because ERA did not publish its timeouts.
- **Budget.** SimpleTES used 2 × 1600 nodes with a warm start, so a single ERA run compares with SimpleTES run A.

## Quick start (any SimpleTES-style task)

```bash
read -rs AKASHML_API_KEY && export AKASHML_API_KEY     # never stored by this code
python baselines/era/run_era.py \
  --init-program datasets/<task>/init_program.py \
  --evaluator    datasets/<task>/evaluator.py \
  --instruction  datasets/<task>/<task>.txt \
  --eval-venv    datasets/<task>/.venv \
  --output-dir   runs/era-<task> \
  --max-nodes 1600 --eval-timeout 14400
```

The pipeline needs Python ≥ 3.9 and the standard library only. The task's own dependencies live in `--eval-venv`. `python baselines/era/run_era.py -h` lists every flag with its default. The defaults are the SimpleTES ZAPBench settings, and `tests/test_run_era.py` fails if the CLI defaults and the client defaults ever drift apart.

Outputs in `--output-dir`:
- `config.json`: settings plus sha256 of the task files. It contains no secrets.
- `nodes.jsonl`: one record per finished node, root first. Each record has the parent, the score, the LLM call stats (attempts, latency, finish_reason, tokens) and the full evaluator metrics.
- `programs/NNNNN.py`: every node's code.
- `best_program.py` and `best.json`: the best node so far.
- `llm_io/`: prompts, responses and reasoning, written only with `--save-llm-io`.

`nodes.jsonl` doubles as the checkpoint: rerun with `--resume` to continue. Replaying 1600 logged nodes takes seconds and makes no API calls and no evaluations. The expansion that was running when the process stopped is redone. Resuming is refused if an experiment setting changed (model, sampling, c_puct, eval timeout, evaluation interpreter, or the task files) or if the log does not match the tree `futs.search` rebuilds.

## Tests

```bash
python baselines/era/smoke_test.py            # 45 unit tests + 17 end-to-end checks, offline, ~30 s
python baselines/era/smoke_test.py --online   # + 6 real gpt-oss calls on toy_task (needs the key)
```

The offline stage runs `run_era.py` through its CLI against a local fake API that injects 503, 429 and malformed replies, and checks:
- retries succeed, the fence cleanup works, and a crashing or timed-out program scores as a failed node;
- the best node is tracked;
- `--resume` replays the logged nodes without calling the API again, continues the run, and refuses changed settings;
- a rejected key stops the run immediately with exit code 4, and an API that stays down through every retry exits with code 3;
- the key never reaches disk or the evaluator.

## ZAPBench example

`datasets/zapbench/` holds the two task files ERA's ZAPBench run starts from:
- `init_program_era.py`: the starter program (root of the tree), a shared-weight per-neuron MLP that implements `forecast(X_train, predict_inputs, ...)`;
- `zapbench.txt`: the task description shown to the LLM inside ERA's prompt (`--instruction`).

They show what a task looks like; they are not a runnable benchmark. The ZAPBench evaluator, the data preparation and the cluster launch scripts are not part of this branch. To run ERA on ZAPBench, add an evaluator that loads the prepared data and returns `{"combined_score": ...}` (higher is better), then follow the quick start above with `--init-program datasets/zapbench/init_program_era.py --instruction datasets/zapbench/zapbench.txt`.
