"""run_era.py pieces that the fairness of the comparison depends on.

Why: the smoke tests and the ZAPBench launcher build their LLM client from
OpenAICompatLLM's defaults, run_era.py from its CLI defaults. If the two ever
drift, the smoke test would validate different sampling settings from the ones
the run uses. The executor must also keep the LLM key out of candidate code and
turn every kind of evaluation failure into a failed node, never a crash.
"""

import inspect
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import futs  # noqa: E402
import run_era  # noqa: E402
from llm import OpenAICompatLLM  # noqa: E402

TASK = ["--init-program", "a", "--evaluator", "b", "--instruction", "c", "--output-dir", "d"]


class Defaults(unittest.TestCase):
    def test_cli_defaults_equal_client_defaults(self):
        cli = run_era.parse_args(TASK)
        ctor = {k: p.default for k, p in inspect.signature(OpenAICompatLLM).parameters.items()}
        pairs = {"model": "model_name", "api_base": "api_base", "temperature": "temperature",
                 "top_p": "top_p", "max_tokens": "max_tokens", "context_window": "context_window",
                 "reasoning_effort": "reasoning_effort", "request_timeout": "request_timeout",
                 "retry_attempts": "max_attempts", "retry_base_delay": "retry_base_delay",
                 "retry_max_delay": "retry_max_delay"}
        for cli_name, ctor_name in pairs.items():
            self.assertEqual(getattr(cli, cli_name), ctor[ctor_name], cli_name)

    def test_defaults_are_the_simpletes_zapbench_settings(self):
        cli = run_era.parse_args(TASK)
        self.assertEqual((cli.model, cli.temperature, cli.top_p, cli.max_tokens,
                          cli.context_window, cli.reasoning_effort, cli.max_nodes, cli.c_puct,
                          cli.eval_timeout),
                         ("openai/gpt-oss-120b", 1.0, 1.0, 49152, 49152, "high", 1600, 1.0,
                          14400.0))


class KeyOnCommandLine(unittest.TestCase):
    """A key typed on the command line must stop the run without reaching any log.

    argparse would take --api-key as an abbreviation of --api-key-env, and the
    'variable not set' message (and config.json) would then carry the key.
    """

    def test_refused_without_echo(self):
        for flags in (["--api-key", "sk-SECRET-7731"], ["--api-key=sk-SECRET-7731"], ["--api-ke", "sk-SECRET-7731"]):
            out = io.StringIO()
            with redirect_stdout(out), mock.patch("sys.stderr", io.StringIO()) as err:
                with self.assertRaises(SystemExit):
                    run_era.parse_args(TASK + flags)
            self.assertNotIn("sk-SECRET-7731", out.getvalue() + err.getvalue(), flags)

    def test_key_env_must_be_a_variable_name(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                run_era.parse_args(TASK + ["--api-key-env", "sk-SECRET-7731"])
        self.assertNotIn("sk-SECRET-7731", err.getvalue())
        self.assertEqual(run_era.parse_args(TASK + ["--api-key-env", "MY_KEY"]).api_key_env,
                         "MY_KEY")


class Prompt(unittest.TestCase):
    def test_prompt_carries_task_parent_code_and_score(self):
        p = run_era.build_prompt("  TASK BODY \n", "def forecast(): pass\n", -0.0193810)
        self.assertIn("TASK BODY", p)
        self.assertIn("The previous solution had a score of: -0.019381 (higher is better)", p)
        self.assertIn("```python\ndef forecast(): pass\n\n```", p)
        self.assertIn("Please generate a NEW, IMPROVED version", p)
        self.assertIn("Provide the full, runnable code including imports.", p)
        self.assertIn("score of: -inf", run_era.build_prompt("t", "c", float("-inf")))


class Executor(unittest.TestCase):
    def _run(self, evaluator_src, timeout=30):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "evaluator.py")
        with open(path, "w") as f:
            f.write(evaluator_src)
        ex = run_era.EvaluatorExecutor(path, sys.executable, timeout, os.path.join(d, "s"),
                                       "AKASHML_API_KEY")
        with mock.patch.dict(os.environ, {"AKASHML_API_KEY": "k", "OTHER_API_KEY": "k2",
                                          "KEEP_ME": "1"}):
            return ex(futs.Problem("p"), futs.Solution("x = 1\n"))

    def test_api_keys_are_hidden_from_the_evaluation(self):
        score, info = self._run(
            "import os\n"
            "def evaluate(p):\n"
            "    return {'combined_score': 1.5,\n"
            "            'keys': sorted(k for k in os.environ if 'API_KEY' in k),\n"
            "            'keep': os.environ.get('KEEP_ME')}\n")
        self.assertEqual(score, 1.5)
        self.assertEqual(info["metrics"]["keys"], [])
        self.assertEqual(info["metrics"]["keep"], "1")      # the rest of the env is passed

    def test_failures_become_minus_inf_nodes(self):
        cases = {
            "nan score": "def evaluate(p):\n    return {'combined_score': float('nan')}\n",
            "missing score": "def evaluate(p):\n    return {'mae': 1.0}\n",
            "evaluator raises": "def evaluate(p):\n    raise ValueError('bad')\n",
            "hard exit": "import os\ndef evaluate(p):\n    os._exit(3)\n",
            "no json": "def evaluate(p):\n    print('hello')\n    return None\n",
            "returns None": "def evaluate(p):\n    return None\n",
            "returns list": "def evaluate(p):\n    return [1, 2]\n",
        }
        for name, src in cases.items():
            score, info = self._run(src)
            self.assertEqual(score, float("-inf"), name)

    def test_timeout_kills_and_scores_minus_inf(self):
        score, info = self._run("import time\ndef evaluate(p):\n    time.sleep(60)\n", timeout=2)
        self.assertEqual(score, float("-inf"))
        self.assertEqual(info["metrics"]["error"], "timeout")
        self.assertLess(info["eval_wall_s"], 30)

    def test_relative_paths_are_made_absolute(self):
        # On POSIX a relative interpreter path would be resolved against each
        # evaluation's temp cwd and not be found (a job script may pass one).
        ex = run_era.EvaluatorExecutor("ev.py", os.path.join("venv", "bin", "python"), 5,
                                       os.path.join(tempfile.mkdtemp(), "s"), "K")
        self.assertTrue(os.path.isabs(ex.python) and os.path.isabs(ex.scratch_dir))
        self.assertEqual(run_era.EvaluatorExecutor("ev.py", "python3", 5, tempfile.mkdtemp(),
                                                   "K").python, "python3")   # PATH lookup kept

    def test_real_score_passes_through(self):
        score, _ = self._run("def evaluate(p):\n    return {'combined_score': -0.0271}\n")
        self.assertTrue(math.isclose(score, -0.0271))


class RootGuard(unittest.TestCase):
    """A failed initial program must stop the run before any LLM call."""

    def _main(self, evaluator_src, *extra):
        d = tempfile.mkdtemp()
        for name, text in (("evaluator.py", evaluator_src), ("init.py", "x = 1\n"),
                           ("task.txt", "task\n")):
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        argv = ["--init-program", os.path.join(d, "init.py"),
                "--evaluator", os.path.join(d, "evaluator.py"),
                "--instruction", os.path.join(d, "task.txt"),
                "--output-dir", os.path.join(d, "out"), "--max-nodes", "0",
                "--api-base", "http://127.0.0.1:9/v1", *extra]
        with mock.patch.dict(os.environ, {"AKASHML_API_KEY": "dummy"}), \
                redirect_stdout(io.StringIO()):
            return run_era.main(argv)

    def test_sentinel_failure_is_refused(self):
        # exactly what the ZAPBench evaluator returns on any failure
        src = ("def evaluate(p):\n    return {'combined_score': -10.0, 'validity': 0.0,\n"
               "                                 'error': 'sbatch rc=1; no result file.'}\n")
        self.assertEqual(self._main(src), 2)
        self.assertEqual(self._main(src, "--allow-invalid-root"), 0)

    def test_valid_root_starts(self):
        self.assertEqual(self._main("def evaluate(p):\n    return {'combined_score': -0.03}\n"),
                         0)


class Persistence(unittest.TestCase):
    def _log(self, d):
        return run_era.RunLog(d, total_nodes=10, save_llm_io=False)

    def _node(self, i, score):
        return futs.Node(index=i, parent_index=None if i == 0 else 0,
                         solution=futs.Solution(f"p{i}"), score=score)

    def test_torn_last_line_is_cut_so_later_nodes_survive(self):
        d = tempfile.mkdtemp()
        log = self._log(d)
        with redirect_stdout(io.StringIO()):
            for i, s in enumerate([0.1, 0.5, 0.2]):
                log.write(self._node(i, s), {}, {"metrics": {}})
            with open(log.nodes_path, "a") as f:
                f.write('{"index": 3, "parent": 0, "sco')        # crash mid-append
            log2 = self._log(d)
            self.assertEqual(len(log2.load()), 3)
            log2.write(self._node(3, 0.9), {}, {"metrics": {}})  # next append
            log3 = self._log(d)
            nodes = log3.load()
        self.assertEqual([n.index for n in nodes], [0, 1, 2, 3])
        with open(os.path.join(d, "best.json")) as f:
            self.assertEqual(json.load(f)["index"], 3)

    def test_best_files_follow_the_log_on_resume(self):
        d = tempfile.mkdtemp()
        log = self._log(d)
        with redirect_stdout(io.StringIO()):
            log.write(self._node(0, 0.1), {}, {"metrics": {}})
            log.write(self._node(1, 0.7), {}, {"metrics": {}})
        with open(os.path.join(d, "best_program.py"), "w") as f:
            f.write("stale")                                     # crash before best was updated
        self._log(d).load()
        with open(os.path.join(d, "best_program.py")) as f:
            self.assertEqual(f.read(), "p1")

    def test_a_failed_write_stops_all_later_writes(self):
        d = tempfile.mkdtemp()
        log = self._log(d)
        with redirect_stdout(io.StringIO()):
            log.write(self._node(0, 0.1), {}, {"metrics": {}})
            with mock.patch.object(log, "_write_best", side_effect=OSError("EIO")):
                with self.assertRaises(OSError):
                    log.write(self._node(1, 0.9), {}, {"metrics": {}})
            log.write(self._node(2, 0.3), {}, {"metrics": {}})  # any later write is dropped
            nodes = self._log(d).load()
        self.assertEqual([n.index for n in nodes], [0, 1])       # no gap: resumable


if __name__ == "__main__":
    unittest.main()
