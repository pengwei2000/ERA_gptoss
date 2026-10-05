#!/usr/bin/env python3
"""Task-agnostic smoke test for the ERA baseline pipeline (no cluster needed).

    python baselines/era/smoke_test.py            # offline, fake API, no key (~1 min)
    python baselines/era/smoke_test.py --online   # + real API stage (needs the key)

Offline, everything goes through run_era.py's real CLI, against a local
OpenAI-compatible server that injects failures, on toy_task/ (1-D regression,
evaluates in milliseconds):
  1. unit tests: live and resumed runs build exactly futs.search's tree; retry policy
  2. an 8-node run: retries after 503 / 429 / malformed replies, fence cleanup,
     a crashing program (-10 sentinel), a program killed by --eval-timeout
     (-inf), best-node tracking, the key never written to disk and never
     visible to the evaluator
  3. --resume replays the logged nodes without calling the API again, then
     continues the same tree; earlier records stay untouched
  4. --resume with a changed experiment setting is refused
  5. a rejected key (HTTP 401) stops the run at once, exit code 4; an API that
     stays down through every retry exits 3 (a job script can auto-resume on 3 only)
  6. a missing key variable exits 2 before doing anything
Online (--online): a 6-node run on toy_task against the real
endpoint with the production sampling settings; every LLM call must succeed.
Exit status 0 iff every check passed.
"""

import argparse
import http.server
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUN_ERA = os.path.join(HERE, "run_era.py")
TOY = os.path.join(HERE, "toy_task")
TASK_ARGS = ["--init-program", os.path.join(TOY, "init_program.py"),
             "--evaluator", os.path.join(TOY, "evaluator.py"),
             "--instruction", os.path.join(TOY, "task.txt")]
KEY_ENV = "AKASHML_API_KEY"

RESULTS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    return ok


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
POLY = '''def fit_predict(x_train, y_train, x_test):
    n = {deg} + 1
    A = [[sum(x ** (i + j) for x in x_train) for j in range(n)] for i in range(n)]
    b = [sum(y * x ** i for x, y in zip(x_train, y_train)) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(n):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * ac for a, ac in zip(A[r], A[c])]
                b[r] -= f * b[c]
    w = [b[i] / A[i][i] for i in range(n)]
    return [sum(w[i] * x ** i for i in range(n)) for x in x_test]
'''
CRASH = "def fit_predict(x_train, y_train, x_test):\n    raise RuntimeError('boom')\n"
SLOW = "import time\n\ndef fit_predict(x_train, y_train, x_test):\n    time.sleep(120)\n"


def _fenced(code):
    return f"```python\n{code}```"


class FakeAPI:
    """Replays a script of replies in request order."""

    def __init__(self, key, mode="script"):
        self.key, self.mode = key, mode
        self.n_requests = 0
        self.bad_auth = 0
        self.lock = threading.Lock()
        self.script = [("status", 503), ("ok", _fenced(POLY.format(deg=3))),
                       ("retry_after", 429), ("ok", CRASH),
                       ("garbage", 200), ("ok", _fenced(SLOW))]
        self.cycle = [POLY.format(deg=1), _fenced(POLY.format(deg=5)), _fenced(POLY.format(deg=7))]
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                with outer.lock:
                    outer.n_requests += 1
                    i = outer.n_requests
                    if self.headers.get("Authorization") != f"Bearer {outer.key}":
                        outer.bad_auth += 1
                if outer.mode == "unauthorized":
                    return self._send(401, b'{"error":"Unauthorized"}')
                if outer.mode == "down":
                    return self._send(503, b"service unavailable")
                kind, val = (outer.script[i - 1] if i <= len(outer.script)
                             else ("ok", outer.cycle[(i - len(outer.script) - 1) % 3]))
                if kind == "status":
                    return self._send(val, b"temporarily unavailable")
                if kind == "retry_after":
                    return self._send(val, b"slow down", {"Retry-After": "0"})
                if kind == "garbage":
                    return self._send(200, b"<html>gateway hiccup</html>")
                body = {"choices": [{"message": {"content": val, "reasoning_content": "..."},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
                return self._send(200, json.dumps(body).encode())

            def _send(self, status, payload, headers=None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_era(args, key, timeout=300):
    env = dict(os.environ)
    env.pop(KEY_ENV, None)
    if key is not None:
        env[KEY_ENV] = key
    t0 = time.time()
    p = subprocess.run([sys.executable, RUN_ERA, *args], env=env, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def records(out_dir):
    with open(os.path.join(out_dir, "nodes.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key_on_disk(out_dir, key):
    for root, _, files in os.walk(out_dir):
        for name in files:
            with open(os.path.join(root, name), "rb") as f:
                if key.encode() in f.read():
                    return os.path.join(root, name)
    return None


def tree_ok(recs):
    return (all(r["index"] == i for i, r in enumerate(recs)) and recs[0]["parent"] is None
            and all(r["parent"] is not None and 0 <= r["parent"] < r["index"] for r in recs[1:]))


def show_tail(output, n=25):
    print("      | " + "\n      | ".join(output.strip().splitlines()[-n:]))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_unit_tests():
    print("\n[1] unit tests")
    suite = unittest.defaultTestLoader.discover(os.path.join(HERE, "tests"))
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
    check("search, sandbox and LLM client unit tests", result.wasSuccessful(),
          f"{result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")


def stage_offline(work):
    key = "smoke-" + secrets.token_hex(12)
    api = FakeAPI(key)
    out = os.path.join(work, "offline")
    fast = ["--api-base", api.base, "--retry-base-delay", "0.2", "--retry-max-delay", "0.5",
            "--eval-timeout", "5", "--eval-scratch-dir", os.path.join(work, "scratch")]
    try:
        print("\n[2] 8-node run against a fake API that injects failures")
        rc, log, dt = run_era([*TASK_ARGS, *fast, "--output-dir", out, "--max-nodes", "8",
                               "--save-llm-io"], key)
        if not check("run exits 0", rc == 0, f"rc={rc}, {dt:.0f}s"):
            show_tail(log)
            return
        recs = records(out)
        check("exactly 1 root + 8 nodes, valid parent links", len(recs) == 9 and tree_ok(recs))
        check("every request carried the key as a Bearer token", api.bad_auth == 0)
        retries = sum(r["gen"]["attempts"] - 1 for r in recs[1:])
        check("503 / 429 / malformed-200 were each retried, then succeeded", retries == 3,
              f"{retries} retries recorded")
        errs = [(r["eval"]["metrics"] or {}).get("error") for r in recs]
        check("crashing program scored as the -10 sentinel",
              any(r["score"] == -10.0 for r in recs))
        slow = [r for r, e in zip(recs, errs) if e == "timeout"]
        check("slow program killed at --eval-timeout and scored -inf",
              len(slow) == 1 and slow[0]["score"] is None and slow[0]["eval"]["eval_wall_s"] < 20,
              f"wall {slow[0]['eval']['eval_wall_s'] if slow else '?'}s")
        valid = [r for r in recs if r["score"] is not None and r["score"] > -10]
        best = max(valid, key=lambda r: r["score"])
        with open(os.path.join(out, "best.json"), encoding="utf-8") as f:
            bj = json.load(f)
        with open(os.path.join(out, "best_program.py"), encoding="utf-8") as f, \
                open(os.path.join(out, best["program"]), encoding="utf-8") as g:
            same = f.read() == g.read()
        check("best.json / best_program.py track the best node",
              bj["index"] == best["index"] and same and best["score"] > recs[0]["score"],
              f"root {recs[0]['score']:.4f} -> best {best['score']:.4f} (#{best['index']})")
        fenced = [r for r in recs[1:] if "```" in open(os.path.join(out, r["program"]),
                                                       encoding="utf-8").read()]
        check("markdown fences stripped (upstream cleanup)", not fenced)
        leaked = key_on_disk(out, key)
        check("API key never written to the run directory", leaked is None, leaked or "")
        seen = [r["eval"]["metrics"].get("saw_api_key") for r in recs
                if "saw_api_key" in (r["eval"]["metrics"] or {})]
        check("evaluator never saw any *API_KEY* variable", seen and not any(seen),
              f"{len(seen)} evaluations checked")

        print("\n[3] --resume to 11 nodes")
        with open(os.path.join(out, "nodes.jsonl"), encoding="utf-8") as f:
            before = f.read()
        requests_before = api.n_requests
        rc, log, _ = run_era([*TASK_ARGS, *fast, "--output-dir", out, "--max-nodes", "11",
                              "--resume"], key)
        with open(os.path.join(out, "nodes.jsonl"), encoding="utf-8") as f:
            after = f.read()
        recs2 = records(out)
        if not check("resume run exits 0 with 12 records, earlier records untouched",
                     rc == 0 and len(recs2) == 12 and after.startswith(before) and tree_ok(recs2),
                     f"rc={rc}, {len(recs2)} records"):
            show_tail(log)
        check("the 8 logged nodes were replayed, not re-generated (3 new API calls)",
              api.n_requests - requests_before == 3 and "replaying 8 logged nodes" in log,
              f"{api.n_requests - requests_before} API calls during resume")

        print("\n[4] --resume with a changed experiment setting")
        rc, log, _ = run_era([*TASK_ARGS, *fast, "--output-dir", out, "--max-nodes", "20",
                              "--temperature", "0.7", "--resume"], key)
        check("refused (exit 2), log unchanged",
              rc == 2 and "refusing to resume" in log and len(records(out)) == 12, f"rc={rc}")
        rc, log, _ = run_era([*TASK_ARGS, *fast, "--output-dir", out, "--max-nodes", "20"], key)
        check("existing run without --resume is refused (exit 2)", rc == 2, f"rc={rc}")
    finally:
        api.close()

    print("\n[5] rejected key (HTTP 401)")
    api = FakeAPI(key, mode="unauthorized")
    fast[1] = api.base
    try:
        rc, log, dt = run_era([*TASK_ARGS, *fast, "--output-dir", os.path.join(work, "bad_key"),
                               "--max-nodes", "8"], key)
        check("stops at once with exit 4 (not auto-resumed), no retries",
              rc == 4 and "not retryable" in log and api.n_requests == 1 and dt < 60,
              f"rc={rc}, {api.n_requests} requests, {dt:.0f}s")
    finally:
        api.close()

    print("\n[5b] API down through every retry (HTTP 503)")
    api = FakeAPI(key, mode="down")
    fast[1] = api.base
    try:
        rc, log, dt = run_era([*TASK_ARGS, *fast, "--output-dir", os.path.join(work, "api_down"),
                               "--max-nodes", "8"], key)
        check("retries, then exits 3 (the code a job script can auto-resume on)",
              rc == 3 and api.n_requests >= 10, f"rc={rc}, {api.n_requests} requests, {dt:.0f}s")
    finally:
        api.close()

    print("\n[6] missing key variable")
    rc, log, _ = run_era([*TASK_ARGS, "--output-dir", os.path.join(work, "no_key")], None)
    check("exits 2 with a message naming the variable", rc == 2 and KEY_ENV in log, f"rc={rc}")


def stage_online(work):
    print("\n[7] real API: 6-node run with the production sampling settings")
    key = os.environ.get(KEY_ENV)
    if not check(f"{KEY_ENV} is set", bool(key)):
        return
    from llm import OpenAICompatLLM
    try:
        ping = OpenAICompatLLM(api_key=key, max_tokens=16, reasoning_effort="low",
                               max_attempts=2, retry_base_delay=5, request_timeout=60,
                               log=lambda m: print("   ", m))
        _, info = ping.draw_sample_with_info("Reply with the single word OK.")
        check("endpoint reachable, key accepted", True, f"{info['latency_s']}s")
    except Exception as e:  # noqa: BLE001
        check("endpoint reachable, key accepted", False, f"{type(e).__name__}: {e}")
        return
    out = os.path.join(work, "online")
    rc, log, dt = run_era([*TASK_ARGS, "--output-dir", out, "--max-nodes", "6",
                           "--eval-timeout", "60",
                           "--eval-scratch-dir", os.path.join(work, "scratch")], key,
                          timeout=3600)
    if not check("run exits 0", rc == 0, f"rc={rc}, {dt / 60:.1f} min"):
        show_tail(log)
        return
    recs = records(out)
    gens = [r["gen"] for r in recs[1:]]
    check("6 nodes, every LLM call answered", len(recs) == 7 and all(g and g["attempts"] for g in gens))
    finish = {}
    for g in gens:
        finish[g["finish_reason"]] = finish.get(g["finish_reason"], 0) + 1
    valid = [r for r in recs[1:] if r["score"] is not None and r["score"] > -10]
    print(f"      finish_reason: {finish}; programs that ran: {len(valid)}/6; "
          f"mean LLM latency {sum(g['latency_s'] for g in gens) / len(gens):.0f}s; "
          f"retries {sum(g['attempts'] - 1 for g in gens)}")
    if valid:
        best = max(valid, key=lambda r: r["score"])
        print(f"      root score {recs[0]['score']:.4f} -> best {best['score']:.4f}")
    else:
        print("      WARNING: no generated program ran -- inspect", out)
    leaked = key_on_disk(out, key)
    check("API key never written to the run directory", leaked is None, leaked or "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--online", action="store_true", help=f"also test the real API ({KEY_ENV})")
    ap.add_argument("--keep", action="store_true", help="keep the temporary run directories")
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="era_smoke_")
    print(f"ERA pipeline smoke test (work dir {work})")
    stage_unit_tests()
    stage_offline(work)
    if args.online:
        stage_online(work)
    ok = all(RESULTS)
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(RESULTS)}/{len(RESULTS)} checks passed")
    if ok and not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"run directories kept in {work}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
