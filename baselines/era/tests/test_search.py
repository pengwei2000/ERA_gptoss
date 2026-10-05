"""The ERA baseline must run upstream's search exactly, including across restarts.

Why: the baseline is only ERA if its tree grows the way upstream `futs.search`
grows it -- one node at a time, each selection seeing every earlier result. The
run is resumed many times (7-day SLURM walls, API outages), so a resumed run
must end up with the very tree an uninterrupted run would have built, without
re-querying the LLM or re-running evaluations for nodes it already has.
"""

import hashlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import futs  # noqa: E402
import run_era  # noqa: E402
from run_era import ReplayError, run_search  # noqa: E402

PROBLEM = futs.Problem("toy")


def _score(program):
    # Deterministic pseudo-random score, so PUCT has real choices to make.
    return int(hashlib.sha256(program.encode()).hexdigest()[:8], 16) / 16 ** 8


def _root():
    return futs.Node(index=0, parent_index=None, solution=futs.Solution("r"), score=_score("r"))


class Task:
    """Mock LLM + evaluator whose output depends only on the tree so far."""

    def __init__(self, recorded=(), fail_on_call=None):
        self.tree = [_root(), *recorded]
        self.gen_calls = self.exec_calls = 0
        self.fail_on_call = fail_on_call

    def generate(self, _p, parent, _s):
        self.gen_calls += 1
        if self.gen_calls == self.fail_on_call:
            raise RuntimeError("API down")
        k = sum(1 for n in self.tree
                if n.parent_index is not None
                and self.tree[n.parent_index].solution.program == parent.program)
        return futs.Solution(f"{parent.program}.{k}"), {"llm": "info"}

    def execute(self, _p, sol):
        self.exec_calls += 1
        return _score(sol.program), {"metrics": {}}

    def on_node(self, node, gen_info, exec_info):
        assert node.index == len(self.tree), "nodes must be logged in index order"
        self.tree.append(node)

    def run(self, n, recorded=()):
        return run_search(PROBLEM, _root(), self.generate, self.execute, num_iterations=n,
                          recorded=list(recorded), on_node=self.on_node)


def _shape(tree):
    return [(n.parent_index, n.solution.program, n.score) for n in tree]


class LiveRun(unittest.TestCase):
    def test_same_tree_as_plain_futs_search(self):
        calls = []
        children = {}

        def generate(_p, parent, _s):
            k = children.get(parent.program, 0)
            children[parent.program] = k + 1
            calls.append((parent.program, f"{parent.program}.{k}"))
            return futs.Solution(f"{parent.program}.{k}")

        def execute(_p, sol):
            return _score(sol.program)

        best_up = futs.search(PROBLEM, futs.Solution("r"), _score("r"), generate, execute,
                              num_iterations=80, c_puct=1.0)
        up_calls, calls[:] = list(calls), []
        children.clear()
        logged = []
        best_ours = run_search(PROBLEM, _root(), generate, execute, num_iterations=80,
                               on_node=lambda n, g, e: logged.append(n))

        self.assertEqual(up_calls, calls)
        self.assertEqual((best_up[0].program, best_up[1]), (best_ours[0].program, best_ours[1]))
        self.assertGreater(len({p for p, _ in up_calls}), 10)       # the search really branched
        # the log names the parent futs.search actually expanded
        programs = ["r"] + [n.solution.program for n in logged]
        self.assertEqual([(programs[n.parent_index], n.solution.program) for n in logged],
                         up_calls)


class Resume(unittest.TestCase):
    def test_replay_rebuilds_the_tree_without_calling_llm_or_evaluator(self):
        full = Task()
        full.run(40)

        part = Task()
        part.run(15)
        resumed = Task(recorded=part.tree[1:])
        resumed.run(40, recorded=part.tree[1:])

        self.assertEqual(_shape(full.tree), _shape(resumed.tree))
        self.assertEqual((resumed.gen_calls, resumed.exec_calls), (25, 25))

    def test_failure_then_resume_equals_uninterrupted_run(self):
        full = Task()
        full.run(40)

        crashed = Task(fail_on_call=10)                   # e.g. the API stays down
        with self.assertRaises(RuntimeError):
            crashed.run(40)
        self.assertEqual(len(crashed.tree), 10)            # root + the 9 finished nodes
        resumed = Task(recorded=crashed.tree[1:])
        resumed.run(40, recorded=crashed.tree[1:])
        self.assertEqual(_shape(full.tree), _shape(resumed.tree))

    def test_a_log_from_another_search_is_rejected(self):
        part = Task()
        part.run(10)
        bad = list(part.tree[1:])
        n = bad[5]
        bad[5] = futs.Node(index=n.index, parent_index=(n.parent_index + 1) % n.index,
                           solution=n.solution, score=n.score)
        with self.assertRaises(ReplayError):
            Task(recorded=bad).run(20, recorded=bad)

    def test_scores_survive_the_log_exactly(self):
        # Replay is exact only if nodes.jsonl gives back bit-identical scores.
        d = tempfile.mkdtemp()
        log = run_era.RunLog(d, total_nodes=5, save_llm_io=False)
        scores = [0.1 + 0.2, float("-inf"), -0.019381006026712, 1e-17, -10.0]
        with redirect_stdout(io.StringIO()):
            for i, s in enumerate(scores):
                log.write(futs.Node(index=i, parent_index=None if i == 0 else 0,
                                    solution=futs.Solution(f"p{i}"), score=s), {}, {})
            loaded = run_era.RunLog(d, total_nodes=5, save_llm_io=False).load()
        self.assertEqual([n.score for n in loaded], scores)


if __name__ == "__main__":
    unittest.main()
