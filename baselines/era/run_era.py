#!/usr/bin/env python3
"""ERA baseline: Flat UCB Tree Search + gpt-oss-120b on any SimpleTES-style task.

A task is the same triple SimpleTES's main.py takes:

  --init-program  starting program; evaluated first, becomes the tree root
  --evaluator     Python file defining evaluate(program_path) -> dict with a
                  'combined_score' (higher is better)
  --instruction   task description shown to the LLM

The search is upstream ERA's own loop, `futs.search`, called unmodified: one
node at a time, FUTS picks the node with the highest PUCT, the LLM gets the
task description plus that node's code and score and writes a new full program,
the program is executed and scored, and only then is the next node selected.
Programs are executed with SimpleTES's own evaluation protocol (fresh
subprocess running evaluate(), optional separate venv, hard timeout +
process-group kill), so a candidate sees the same sandbox under ERA as under
SimpleTES.

    export AKASHML_API_KEY=...        # read from the environment, never stored
    python baselines/era/run_era.py \
        --init-program datasets/<task>/init_program.py \
        --evaluator    datasets/<task>/evaluator.py \
        --instruction  datasets/<task>/<task>.txt \
        --output-dir   runs/era-<task> \
        --max-nodes 1600 --eval-timeout 14400

Outputs under --output-dir (nodes.jsonl is also the checkpoint):
    config.json         settings + sha256 of the task files (no secrets)
    nodes.jsonl         one JSON record per finished node, root first
    programs/NNNNN.py   code of every node
    best_program.py     best node so far (+ best.json)
    llm_io/NNNNN.json   prompt / response / reasoning (only with --save-llm-io)

Re-running the same command with --resume continues from nodes.jsonl: the
logged nodes are replayed through futs.search (see run_search), then the
search continues live. The expansion that was running when the process
stopped is redone.
"""

import argparse
import datetime
import hashlib
import json
import re
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import futs  # noqa: E402  (upstream, unmodified)
from llm import LLMRequestError, LLMUnavailableError, OpenAICompatLLM  # noqa: E402

ERA_UPSTREAM_COMMIT = "b836730b5c000526af95116b1d0e2c60c8cf0a10"

# ---------------------------------------------------------------------------
# Prompt (generate_fn)
# ---------------------------------------------------------------------------
# Upstream's only generator is the Kaggle-specific PlaygroundGenerator
# (implementation/playground_s3e1.py). Its prompt is:
#
#     {problem.description}
#     Here is a preview of the training data: ...        <- task-specific
#     The goal is to predict 'MedHouseVal'. The metric   <- task-specific
#     is RMSE ... Lower is better.
#     The previous solution had a score (RMSE) of: {rmse:.5f}
#     Previous Solution Code:
#     ```python
#     {parent_solution.program}
#     ```
#     Please generate a NEW, IMPROVED Python function named `train_and_predict`
#     that: 1..4 (signature, allowed libraries)          <- task-specific
#     ... Provide the full, runnable code including imports.
#     IMPORTANT CONSTRAINTS FOR SPEED: ...               <- task-specific
#
# The task-specific lines (data, metric, signature, constraints, budget) are
# exactly what a SimpleTES instruction file contains, so the instruction file
# takes their place; the parent-score line, the parent-code block, the "NEW,
# IMPROVED" request and the full-program request are kept. Scores are shown as
# the task's raw combined_score (higher is better) rather than upstream's
# sign-flipped RMSE, with %.6g instead of %.5f so small-magnitude scores keep
# their digits. llm.py wraps this in upstream's preamble unchanged.
ERA_PROMPT = """
{description}

The previous solution had a score of: {score} (higher is better)
Previous Solution Code:
```python
{code}
```

Please generate a NEW, IMPROVED version of this program for the task described above.
Provide the full, runnable code including imports.
"""


def build_prompt(description, parent_code, parent_score):
    return ERA_PROMPT.format(description=description.strip(),
                             score=f"{parent_score:.6g}", code=parent_code)


class EraGenerator:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, problem, parent_solution, parent_score):
        prompt = build_prompt(problem.description, parent_solution.program, parent_score)
        code, info = self.llm.draw_sample_with_info(prompt)
        info["prompt"] = prompt
        return futs.Solution(code), info


# ---------------------------------------------------------------------------
# Search -- upstream futs.search, unmodified
# ---------------------------------------------------------------------------
class ReplayError(RuntimeError):
    """nodes.jsonl does not match the tree futs.search rebuilds."""


def _split(result):
    # generate/execute return (value, info) so the run can be logged;
    # futs.search itself only ever sees the value.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    return result, {}


def run_search(problem, root, generate_fn, execute_fn, num_iterations, c_puct=1.0,
               recorded=(), on_node=None):
    """Run upstream `futs.search` as-is; log every new node through on_node.

    futs.search keeps its tree in memory and only returns the best node, so the
    only hooks are generate_fn / execute_fn (upstream's playground_s3e1.py
    tracks its search the same way, by wrapping execute_fn).

    Resume: futs.search always starts from the root and is deterministic given
    the scores it receives, so feeding it the `recorded` nodes (index 1..k of
    an earlier run, in order) instead of calling the LLM and the evaluator makes
    it rebuild exactly the same tree; the search then continues live. Every
    replayed step checks that futs.search selected the recorded parent.
    """
    index_of = {id(root.solution): 0}
    solutions = [root.solution]          # keeps objects alive, so ids stay unique
    pending = {}

    def generate(prob, parent_solution, parent_score):
        parent = index_of[id(parent_solution)]
        index = len(solutions)
        if index <= len(recorded):
            rec = recorded[index - 1]
            if rec.parent_index != parent:
                raise ReplayError(
                    f"node {index}: nodes.jsonl says parent {rec.parent_index}, futs.search "
                    f"selected {parent}; the log does not belong to this search")
            pending.update(parent=parent, replay=True, score=rec.score, gen_info={})
            return rec.solution
        solution, gen_info = _split(generate_fn(prob, parent_solution, parent_score))
        pending.update(parent=parent, replay=False, gen_info=gen_info)
        return solution

    def execute(prob, solution):
        if pending["replay"]:
            score, exec_info = pending["score"], {}
        else:
            score, exec_info = _split(execute_fn(prob, solution))
        index = len(solutions)
        index_of[id(solution)] = index
        solutions.append(solution)
        if not pending["replay"] and on_node is not None:
            on_node(futs.Node(index=index, parent_index=pending["parent"],
                              solution=solution, score=score),
                    pending["gen_info"], exec_info)
        return score

    return futs.search(problem, root.solution, root.score, generate, execute,
                       num_iterations=num_iterations, c_puct=c_puct)


# ---------------------------------------------------------------------------
# Execution (execute_fn) -- SimpleTES's evaluation protocol
# ---------------------------------------------------------------------------
# Verbatim copy of SimpleTES's runner (simpletes/evaluator.py::_EVAL_RUNNER):
# load the task evaluator in a fresh interpreter, call evaluate(program), print
# the result dict as the last JSON line on stdout.
_EVAL_RUNNER = textwrap.dedent(r'''
import importlib.util
import json
import sys
import traceback

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Invalid arguments", "combined_score": float("-inf")}))
        sys.exit(1)

    evaluator_path = sys.argv[1]
    target_file = sys.argv[2]

    try:
        # Load evaluator module
        spec = importlib.util.spec_from_file_location("user_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            print(json.dumps({"error": f"Could not load evaluator: {evaluator_path}", "combined_score": float("-inf")}))
            sys.exit(1)

        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        # Run evaluation
        result = mod.evaluate(target_file)

        # Output result as JSON
        print(json.dumps(result))

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_msg, "combined_score": float("-inf")}))
        sys.exit(1)

if __name__ == "__main__":
    main()
''').strip()

ERROR_MAX_CHARS = 4000


def _is_secret(name, api_key_env):
    return name == api_key_env or "API_KEY" in name.upper()


class EvaluatorExecutor:
    """Runs evaluate(program) in a subprocess with a hard wall-clock timeout.

    Mirrors SimpleTES's EvaluatorWorker: per-eval scratch dir as cwd and
    TMPDIR, new session so a timeout kills the whole process tree, last JSON
    line of stdout is the result, non-zero exit / timeout -> -inf. Deliberate
    differences: API-key variables are removed from the child environment so
    candidate code (and, through sbatch --export=ALL, the SLURM worker) never
    sees the LLM key; a last line that is JSON but not a dict (evaluate()
    returned None, a list, ...) is a failure rather than a cue to look further
    up for a JSON line.
    """

    def __init__(self, evaluator_path, python_executable, timeout, scratch_dir, api_key_env):
        # Absolute paths: on POSIX, Popen resolves a relative program path against
        # its cwd=, and every evaluation runs with cwd set to its own temp dir.
        self.evaluator_path = os.path.abspath(evaluator_path)
        self.python = (os.path.abspath(python_executable) if os.path.dirname(python_executable)
                       else python_executable)
        self.timeout = timeout
        self.scratch_dir = os.path.abspath(scratch_dir)
        self.api_key_env = api_key_env
        self._procs = set()
        self._procs_lock = threading.Lock()
        os.makedirs(self.scratch_dir, exist_ok=True)

    def __call__(self, problem, solution):
        eval_dir = tempfile.mkdtemp(prefix="eval_", dir=self.scratch_dir)
        program_path = os.path.join(eval_dir, "program.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write(solution.program)
        env = {k: v for k, v in os.environ.items() if not _is_secret(k, self.api_key_env)}
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = eval_dir
        t0 = time.time()
        proc = subprocess.Popen(
            [self.python, "-c", _EVAL_RUNNER, self.evaluator_path, program_path],
            cwd=eval_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with self._procs_lock:
            self._procs.add(proc)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                _reap(proc)
                metrics = {"error": "timeout", "combined_score": float("-inf")}
            except BaseException:     # e.g. KeyboardInterrupt during the root eval
                _kill_group(proc)
                raise
            else:
                out = stdout.decode("utf-8", errors="replace").strip()
                err = stderr.decode("utf-8", errors="replace").strip()
                if proc.returncode != 0:
                    metrics = {"error": err or out or f"Process exited with code {proc.returncode}",
                               "combined_score": float("-inf")}
                else:
                    metrics = _parse_last_json(out)
                    if not isinstance(metrics, dict):   # e.g. evaluate() returned None
                        metrics = {"error": f"evaluator returned {type(metrics).__name__}, "
                                            f"not a dict", "combined_score": float("-inf")}
        finally:
            with self._procs_lock:
                self._procs.discard(proc)
            shutil.rmtree(eval_dir, ignore_errors=True)
        if isinstance(metrics.get("error"), str) and len(metrics["error"]) > ERROR_MAX_CHARS:
            metrics["error"] = metrics["error"][:ERROR_MAX_CHARS] + "...[truncated]"
        score = metrics.get("combined_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            score = float("-inf")
        return float(score), {"metrics": metrics, "eval_wall_s": round(time.time() - t0, 1)}

    def kill_all(self):
        with self._procs_lock:
            procs = list(self._procs)
        for proc in procs:
            _kill_group(proc)


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        if os.name == "nt":   # no process groups: take down the runner's process tree
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        try:
            proc.kill()
        except OSError:
            pass


def _reap(proc, grace=10.0):
    """Wait for the killed runner itself, not for its pipes (as SimpleTES does
    with proc.wait()): a descendant that left its process group can hold stdout
    open long after the runner is gone. The pipes are not closed here: on
    Windows a reader thread from communicate() may still hold them, and closing
    would wait for that descendant after all; they close when proc is released."""
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def _parse_last_json(text):
    if not text:
        return {"error": "Empty evaluator output", "combined_score": float("-inf")}
    # The runner prints evaluate()'s result as the last line. If that line is
    # JSON it is the result, whatever its type: an evaluator returning None must
    # fail, not fall back to an earlier JSON-looking line the candidate printed.
    # Otherwise (output after the runner's print) scan back like SimpleTES does.
    try:
        return json.loads(text.rsplit("\n", 1)[-1].strip())
    except json.JSONDecodeError:
        pass
    for line in reversed(text.split("\n")):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON output: {e}", "combined_score": float("-inf")}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class RunLog:
    """Append-only node log. A node exists iff its line is in nodes.jsonl."""

    def __init__(self, out_dir, total_nodes, save_llm_io):
        self.out_dir = out_dir
        self.total_nodes = total_nodes
        self.save_llm_io = save_llm_io
        self.nodes_path = os.path.join(out_dir, "nodes.jsonl")
        os.makedirs(os.path.join(out_dir, "programs"), exist_ok=True)
        if save_llm_io:
            os.makedirs(os.path.join(out_dir, "llm_io"), exist_ok=True)
        self.best_index, self.best_score = None, float("-inf")
        self.t_start = time.time()
        # Set on interrupt BEFORE the running eval is killed, so a killed eval is
        # never recorded as a failed node.
        self.closed = False

    def exists(self):
        return os.path.exists(self.nodes_path) and os.path.getsize(self.nodes_path) > 0

    def load(self):
        """Rebuild the tree (visits are recomputed by the search).

        A last line torn by a crash is cut off the file (not just skipped), so the
        next append starts on a fresh line; best_program.py / best.json are
        re-derived so they always agree with nodes.jsonl.
        """
        with open(self.nodes_path, encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        if text.endswith("\n"):
            lines.pop()
        good, records = [], []
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
                good.append(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    print(f"[era] dropping truncated last line of {self.nodes_path}")
                    break
                raise SystemExit(f"[era] corrupt line {i + 1} in {self.nodes_path}")
        repaired = "".join(line + "\n" for line in good)
        if repaired != text:
            _atomic_write(self.nodes_path, repaired)
        nodes = []
        for rec in records:
            if rec["index"] != len(nodes):
                raise SystemExit(f"[era] nodes.jsonl out of order at index {rec['index']}")
            with open(os.path.join(self.out_dir, rec["program"]), encoding="utf-8") as f:
                program = f.read()
            score = rec["score"] if rec["score"] is not None else float("-inf")
            nodes.append(futs.Node(index=rec["index"], parent_index=rec["parent"],
                                   solution=futs.Solution(program), score=score))
            if score > self.best_score:
                self.best_index, self.best_score = rec["index"], score
        if self.best_index is not None:
            rec = records[self.best_index]
            self._write_best(nodes[self.best_index], rec["program"],
                             (rec.get("eval") or {}).get("metrics"))
        return nodes

    def write(self, node, gen_info, exec_info):
        if self.closed:
            return
        try:
            self._write(node, gen_info, exec_info)
        except BaseException:
            # Stop persisting: nodes finishing after a failed write would leave a
            # gap in nodes.jsonl and make it unresumable. They are redone on resume.
            self.closed = True
            raise

    def _write_best(self, node, rel, metrics):
        _atomic_write(os.path.join(self.out_dir, "best_program.py"), node.solution.program)
        _atomic_write(os.path.join(self.out_dir, "best.json"),
                      json.dumps({"index": node.index, "score": node.score,
                                  "parent": node.parent_index, "program": rel,
                                  "metrics": metrics}, indent=2))

    def _write(self, node, gen_info, exec_info):
        rel = f"programs/{node.index:05d}.py"
        with open(os.path.join(self.out_dir, rel), "w", encoding="utf-8") as f:
            f.write(node.solution.program)
        gen = None
        if gen_info:
            gen = {k: v for k, v in gen_info.items() if k not in ("prompt", "reasoning")}
            if self.save_llm_io:
                _atomic_write(os.path.join(self.out_dir, "llm_io", f"{node.index:05d}.json"),
                              json.dumps({"prompt": gen_info.get("prompt"),
                                          "response": node.solution.program,
                                          "reasoning": gen_info.get("reasoning")}, indent=1))
        rec = {
            "index": node.index,
            "parent": node.parent_index,
            "score": node.score if math.isfinite(node.score) else None,
            "program": rel,
            "finished_at": _now(),
            "gen": gen,
            "eval": exec_info or None,
        }
        with open(self.nodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if node.score > self.best_score:
            self.best_index, self.best_score = node.index, node.score
            self._write_best(node, rel, (exec_info or {}).get("metrics"))
        self._print(node, gen, exec_info or {})

    def _print(self, node, gen, exec_info):
        err = (exec_info.get("metrics") or {}).get("error")
        parts = [f"[era] node {node.index:>4}/{self.total_nodes - 1}",
                 f"parent {node.parent_index if node.parent_index is not None else '-':>4}",
                 f"score {node.score:.6g}",
                 f"best {self.best_score:.6g} (#{self.best_index})"]
        if gen:
            parts.append(f"llm {gen.get('latency_s')}s x{gen.get('attempts')} "
                         f"{gen.get('finish_reason')}")
        parts.append(f"eval {exec_info.get('eval_wall_s')}s")
        parts.append(f"elapsed {(time.time() - self.t_start) / 3600:.2f}h")
        line = " | ".join(parts)
        if err:
            line += f" | error: {str(err).splitlines()[0][:160] if str(err) else ''}"
        print(line, flush=True)


# Settings that define the experiment. Resuming with any of them changed would
# silently mix two experiments in one tree, so it is refused.
FINGERPRINT_KEYS = ("init_program_sha256", "evaluator_sha256", "instruction_sha256",
                    "model", "api_base", "temperature", "top_p", "max_tokens",
                    "context_window", "reasoning_effort", "c_puct", "eval_timeout",
                    "eval_python")


def _setting(key, value):
    # eval_python may be stored relative to the run's working directory; compare
    # it as an absolute path without resolving symlinks (venvs link one base python).
    if key == "eval_python" and isinstance(value, str) and os.path.dirname(value):
        return os.path.normcase(os.path.abspath(value))
    return value


def _venv_python(venv):
    """The venv's interpreter: bin/python (POSIX) or Scripts/python.exe (Windows)."""
    for parts in (("bin", "python"), ("Scripts", "python.exe")):
        path = os.path.join(venv, *parts)
        if os.path.exists(path):
            return path
    return os.path.join(venv, "bin", "python")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _key_on_command_line(args):
    # Checked before argparse runs: it would accept --api-key as an abbreviation
    # of --api-key-env, or echo the value in an "unrecognized argument" error.
    return any(a.split("=")[0].startswith("--api-k") and a.split("=")[0] != "--api-key-env"
               for a in args)


def parse_args(argv=None):
    raw = sys.argv[1:] if argv is None else list(argv)
    if _key_on_command_line(raw):
        print("[era] do not pass an API key on the command line; export it and name the "
              "variable with --api-key-env")
        raise SystemExit(2)
    p = argparse.ArgumentParser(
        description="ERA (Flat UCB Tree Search) baseline with an OpenAI-compatible LLM API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    t = p.add_argument_group("task (same inputs as SimpleTES main.py)")
    t.add_argument("--init-program", required=True)
    t.add_argument("--evaluator", required=True)
    t.add_argument("--instruction", required=True)
    t.add_argument("--eval-venv", default=None,
                   help="venv whose python runs the evaluator (SimpleTES --eval-venv); "
                        "default: this interpreter")
    t.add_argument("--eval-timeout", type=float, default=14400.0,
                   help="hard wall-clock limit per evaluate() call, seconds (SimpleTES --eval-timeout)")
    t.add_argument("--eval-scratch-dir", default=None,
                   help="where per-eval temp dirs go; default: <system tmp>/era_eval")
    s = p.add_argument_group("search (ERA FUTS)")
    s.add_argument("--max-nodes", type=int, default=1600,
                   help="expansions after the root (upstream num_iterations)")
    s.add_argument("--c-puct", type=float, default=1.0, help="upstream default; paper: c=1")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--resume", action="store_true",
                   help="continue from <output-dir>/nodes.jsonl if present (else start fresh)")
    s.add_argument("--allow-invalid-root", action="store_true",
                   help="start even if the initial program fails to evaluate")
    m = p.add_argument_group("LLM (OpenAI-compatible chat completions)")
    m.add_argument("--model", default="openai/gpt-oss-120b")
    m.add_argument("--api-base", default="https://api.akashml.com/v1")
    m.add_argument("--api-key-env", default="AKASHML_API_KEY",
                   help="name of the environment variable holding the key")
    m.add_argument("--temperature", type=float, default=1.0)
    m.add_argument("--top-p", type=float, default=1.0)
    m.add_argument("--max-tokens", type=int, default=49152,
                   help="cap on generated (reasoning + answer) tokens per call")
    m.add_argument("--context-window", type=int, default=49152,
                   help="also cap prompt + generated tokens, like SimpleTES's vLLM "
                        "--max-model-len 49152; 0 = no cap beyond --max-tokens")
    m.add_argument("--reasoning-effort", default="high", help="'none' to omit the field")
    m.add_argument("--request-timeout", type=float, default=1800.0,
                   help="seconds to wait for one HTTP response before retrying")
    m.add_argument("--retry-attempts", type=int, default=10)
    m.add_argument("--retry-base-delay", type=float, default=60.0,
                   help="first retry wait, seconds; doubles per attempt")
    m.add_argument("--retry-max-delay", type=float, default=300.0)
    m.add_argument("--save-llm-io", action="store_true",
                   help="also save prompt/response/reasoning per node")
    args = p.parse_args(raw)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        p.error("--api-key-env must name an environment variable")
    return args


def build_llm(args, api_key):
    """The API client for these CLI settings (the SimpleTES-API parity test, not in this branch, uses it too)."""
    return OpenAICompatLLM(
        api_key=api_key, model_name=args.model, api_base=args.api_base,
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        context_window=args.context_window or None,
        reasoning_effort=None if args.reasoning_effort.lower() == "none" else args.reasoning_effort,
        request_timeout=args.request_timeout, max_attempts=args.retry_attempts,
        retry_base_delay=args.retry_base_delay, retry_max_delay=args.retry_max_delay)


def main(argv=None):
    args = parse_args(argv)
    for flag in ("init_program", "evaluator", "instruction"):
        if not os.path.isfile(getattr(args, flag)):
            print(f"[era] --{flag.replace('_', '-')} not found: {getattr(args, flag)}")
            return 2
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"[era] environment variable {args.api_key_env} is not set "
              f"(e.g.  read -rs {args.api_key_env} && export {args.api_key_env})")
        return 2
    eval_python = _venv_python(args.eval_venv) if args.eval_venv else sys.executable
    if not os.path.exists(eval_python):
        print(f"[era] evaluation python not found: {eval_python}")
        return 2
    reasoning_effort = None if args.reasoning_effort.lower() == "none" else args.reasoning_effort

    # backpropagate_visit (upstream) recurses once per tree level; a 1600-node
    # search can in principle build a chain deeper than Python's default 1000.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    log = RunLog(out_dir, total_nodes=args.max_nodes + 1, save_llm_io=args.save_llm_io)

    config = {
        "era_upstream_commit": ERA_UPSTREAM_COMMIT,
        "init_program": os.path.abspath(args.init_program),
        "init_program_sha256": _sha256(args.init_program),
        "evaluator": os.path.abspath(args.evaluator),
        "evaluator_sha256": _sha256(args.evaluator),
        "instruction": os.path.abspath(args.instruction),
        "instruction_sha256": _sha256(args.instruction),
        "eval_python": eval_python,
        "eval_timeout": args.eval_timeout,
        "max_nodes": args.max_nodes,
        "c_puct": args.c_puct,
        "model": args.model,
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "context_window": args.context_window or None,
        "reasoning_effort": reasoning_effort,
        "request_timeout": args.request_timeout,
        "retry_attempts": args.retry_attempts,
        "retry_base_delay": args.retry_base_delay,
        "retry_max_delay": args.retry_max_delay,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "started_at": _now(),
    }
    config_path = os.path.join(out_dir, "config.json")
    if log.exists():
        if not args.resume:
            print(f"[era] {log.nodes_path} already exists; pass --resume to continue it "
                  f"or choose a new --output-dir")
            return 2
        with open(config_path, encoding="utf-8") as f:
            first = json.load(f)
        changed = {k: (first.get(k), config[k]) for k in FINGERPRINT_KEYS
                   if _setting(k, first.get(k)) != _setting(k, config[k])}
        if changed:
            print(f"[era] refusing to resume: experiment settings differ from the "
                  f"original run: {changed}")
            return 2
    else:
        _atomic_write(config_path, json.dumps(config, indent=2))

    print(f"[era] ERA FUTS (google-research/era@{ERA_UPSTREAM_COMMIT[:7]}) -> {out_dir}")
    print(f"[era] {args.max_nodes} nodes (serial, as upstream), c_puct={args.c_puct}, "
          f"eval timeout {args.eval_timeout:.0f}s, eval python {eval_python}")

    try:
        llm = build_llm(args, api_key)
    except LLMRequestError as e:
        print(f"[era] {args.api_key_env}: {e}")
        return 2
    print(f"[era] {llm!r}")
    executor = EvaluatorExecutor(
        args.evaluator, eval_python, args.eval_timeout,
        args.eval_scratch_dir or os.path.join(tempfile.gettempdir(), "era_eval"),
        args.api_key_env)
    with open(args.instruction, encoding="utf-8") as f:
        problem = futs.Problem(f.read())

    def _on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        nodes = log.load() if log.exists() else []
        if nodes:
            print(f"[era] resuming: replaying {len(nodes) - 1} logged nodes through "
                  f"futs.search (best {log.best_score:.6g} at #{log.best_index})")
        else:
            with open(args.init_program, encoding="utf-8") as f:
                root_solution = futs.Solution(f.read())
            print("[era] evaluating the initial program ...", flush=True)
            root_score, root_info = executor(problem, root_solution)
            # SimpleTES evaluators report failure as a finite sentinel (ZAPBench:
            # -10.0) with an 'error' key / validity 0, not only as -inf.
            metrics = root_info["metrics"]
            root_failed = (not math.isfinite(root_score) or bool(metrics.get("error"))
                           or metrics.get("validity") == 0.0)
            if root_failed and not args.allow_invalid_root:
                print(f"[era] initial program failed to evaluate: "
                      f"{root_info['metrics'].get('error')}\n"
                      f"[era] fix the task setup, or pass --allow-invalid-root")
                return 2
            root = futs.Node(index=0, parent_index=None, solution=root_solution, score=root_score)
            nodes = [root]
            log.write(root, {}, root_info)

        if len(nodes) - 1 >= args.max_nodes:
            print(f"[era] already have {len(nodes) - 1} expansions >= --max-nodes; nothing to do")
        else:
            run_search(problem, nodes[0], EraGenerator(llm), executor,
                       num_iterations=args.max_nodes, c_puct=args.c_puct,
                       recorded=nodes[1:], on_node=log.write)
    except ReplayError as e:
        print(f"[era] cannot resume: {e}")
        return 2
    except KeyboardInterrupt as e:
        log.closed = True
        executor.kill_all()
        print(f"[era] interrupted ({e}); finished nodes are in {log.nodes_path}. "
              f"The running expansion is dropped; re-run with --resume to continue.", flush=True)
        return 130
    except (LLMUnavailableError, LLMRequestError) as e:
        executor.kill_all()
        print(f"[era] STOPPED: LLM API error: {e}\n"
              f"[era] finished nodes are saved; re-run the same command with --resume "
              f"once the API is healthy (exit 3) or the key/model/parameters are fixed "
              f"(exit 4).", flush=True)
        return 3 if isinstance(e, LLMUnavailableError) else 4   # 3 = retry later, 4 = fix config

    print(f"[era] done: best score {log.best_score:.6g} at node "
          f"#{log.best_index} -> {os.path.join(out_dir, 'best_program.py')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
