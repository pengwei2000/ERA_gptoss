"""The API client must wait out server trouble and fail loud on caller mistakes.

Why: a 1600-node search runs for weeks. A transient 503 must cost minutes, not
the run; a wrong key or model name must stop the run immediately instead of
burning 40 minutes of retries per node. And the request must carry exactly the
sampling settings the SimpleTES runs used, or the comparison is not fair.
"""

import http.server
import json
import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402
from llm import LLMRequestError, LLMUnavailableError, OpenAICompatLLM  # noqa: E402

KEY = "test-key-5f1c0e"


def _ok(content, finish="stop"):
    return 200, {}, json.dumps({
        "choices": [{"message": {"content": content, "reasoning_content": "thinking"},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22}}).encode()


class _Server:
    """Local OpenAI-compatible endpoint that replays a scripted list of replies."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "auth": self.headers.get("Authorization")})
                status, headers, payload = outer.script.pop(0) if outer.script else _ok("x")
                self.send_response(status)
                for k, v in headers.items():
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


def _client(base, **kw):
    kw.setdefault("retry_base_delay", 0.01)
    kw.setdefault("retry_max_delay", 0.02)
    kw.setdefault("max_attempts", 4)
    return OpenAICompatLLM(api_key=KEY, api_base=base, log=lambda _m: None, **kw)


class Request(unittest.TestCase):
    def test_body_matches_simpletes_sampling_and_upstream_prompt(self):
        srv = _Server([_ok("```python\nprint(1)\n```")])
        try:
            code, info = _client(srv.base).draw_sample_with_info("TASK TEXT")
        finally:
            srv.close()
        req = srv.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], f"Bearer {KEY}")
        b = req["body"]
        self.assertEqual((b["model"], b["temperature"], b["top_p"], b["reasoning_effort"]),
                         ("openai/gpt-oss-120b", 1.0, 1.0, "high"))
        prompt = b["messages"][0]["content"]
        # prompt + completion must fit SimpleTES's 49152-token context
        self.assertLess(b["max_tokens"], 49152)
        self.assertGreater(b["max_tokens"], 49152 - 1000)
        self.assertIn("Return ONLY the python code.", prompt)      # upstream preamble
        self.assertIn("--- BEGIN PROMPT ---\nTASK TEXT\n--- END PROMPT ---", prompt)
        self.assertEqual(code, "print(1)")                           # upstream fence cleanup
        self.assertEqual((info["attempts"], info["finish_reason"], info["completion_tokens"]),
                         (1, "stop", 22))

    def test_key_not_in_repr(self):
        self.assertNotIn(KEY, repr(_client("http://127.0.0.1:1/v1")))

    def test_trailing_newline_in_key_is_stripped(self):
        # `export KEY=$(cat key.txt)` on a CRLF file keeps the \r
        srv = _Server([_ok("x = 1")])
        try:
            OpenAICompatLLM(api_key=KEY + "\r\n", api_base=srv.base).draw_sample("p")
        finally:
            srv.close()
        self.assertEqual(srv.requests[0]["auth"], f"Bearer {KEY}")

    def test_malformed_key_fails_without_echoing_it(self):
        with self.assertRaises(LLMRequestError) as cm:
            OpenAICompatLLM(api_key="abc\ndef-secret")
        self.assertNotIn("def-secret", str(cm.exception))


class ContextWindow(unittest.TestCase):
    """SimpleTES's vLLM served a 49152-token context: prompt + completion <= 49152."""

    def test_completion_shrinks_with_the_prompt_and_learns_real_size(self):
        long_prompt = "word " * 8000                      # ~40k chars
        srv = _Server([
            (200, {}, json.dumps({"choices": [{"message": {"content": "a"},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 8200}}).encode()),
            _ok("b")])
        try:
            c = _client(srv.base)
            c.draw_sample(long_prompt)
            c.draw_sample(long_prompt)
        finally:
            srv.close()
        first, second = (r["body"]["max_tokens"] for r in srv.requests)
        self.assertLessEqual(first + 8200, 49152)          # conservative before any usage
        self.assertLessEqual(second + 8200, 49152)         # uses the reported prompt size
        self.assertGreater(second, first)                  # ...and is no longer over-cautious
        self.assertGreater(second + 8200, 49152 - 300)

    def test_cold_start_neither_starves_nor_overgrants_long_prompts(self):
        # The first calls after every (re)start have no usage.prompt_tokens yet.
        # SimpleTES prompts reach ~128k chars (o200k measured 0.237-0.268
        # tokens/char on them): the first call must leave a usable budget and
        # still keep prompt + completion inside the 49152-token window.
        c = _client("http://127.0.0.1:9/v1")
        for chars in (20_000, 88_000, 128_511):
            prompt = "x" * chars
            cap = c.completion_cap(prompt)
            self.assertLessEqual(cap + int(0.268 * chars), 49152)
            if chars == 128_511:
                self.assertGreater(cap, 12_000)

    def test_no_window_sends_max_tokens_unchanged(self):
        srv = _Server([_ok("x")])
        try:
            _client(srv.base, context_window=None).draw_sample("word " * 8000)
        finally:
            srv.close()
        self.assertEqual(srv.requests[0]["body"]["max_tokens"], 49152)


class Retries(unittest.TestCase):
    def test_server_errors_are_retried_until_success(self):
        srv = _Server([(503, {}, b"busy"), (429, {"Retry-After": "0"}, b"slow down"),
                       (200, {}, b"<html>not json</html>"), _ok("y = 2")])
        try:
            code, info = _client(srv.base).draw_sample_with_info("p")
        finally:
            srv.close()
        self.assertEqual(code, "y = 2")
        self.assertEqual(info["attempts"], 4)
        self.assertEqual(len(info["errors"]), 3)

    def test_gives_up_after_max_attempts(self):
        srv = _Server([(502, {}, b"bad gateway")] * 10)
        try:
            with self.assertRaises(LLMUnavailableError):
                _client(srv.base, max_attempts=3).draw_sample("p")
        finally:
            srv.close()
        self.assertEqual(len(srv.requests), 3)

    def test_network_failure_is_retried(self):
        c = _client("http://127.0.0.1:9/v1", max_attempts=2, request_timeout=2)
        with self.assertRaises(LLMUnavailableError):
            c.draw_sample("p")

    def test_every_5xx_is_retried(self):
        for status in (501, 507, 530, 599):
            srv = _Server([(status, {}, b"origin unreachable"), _ok("ok = 1")])
            try:
                code = _client(srv.base).draw_sample("p")
            finally:
                srv.close()
            self.assertEqual((code, len(srv.requests)), ("ok = 1", 2), status)

    def test_redirects_are_not_followed_so_the_key_stays_put(self):
        other = _Server([_ok("never")])
        try:
            srv = _Server([(302, {"Location": other.base + "/chat/completions"}, b"")])
            try:
                with self.assertRaises(LLMRequestError):
                    _client(srv.base).draw_sample("p")
            finally:
                srv.close()
        finally:
            other.close()
        self.assertEqual(other.requests, [])

    def test_client_errors_fail_immediately(self):
        for status in (400, 401, 403, 404, 422):
            srv = _Server([(status, {}, b'{"error":"nope"}'), _ok("never")])
            try:
                with self.assertRaises(LLMRequestError):
                    _client(srv.base).draw_sample("p")
            finally:
                srv.close()
            self.assertEqual(len(srv.requests), 1, status)

    def test_empty_answer_is_returned_not_retried(self):
        # e.g. the reasoning used the whole budget: that is the model's answer,
        # and upstream ERA would score the empty program as a failed node.
        srv = _Server([_ok("", finish="length")])
        try:
            code, info = _client(srv.base).draw_sample_with_info("p")
        finally:
            srv.close()
        self.assertEqual((code, info["finish_reason"], info["attempts"]), ("", "length", 1))

    def test_backoff_is_minutes_scale_by_default(self):
        c = OpenAICompatLLM(api_key=KEY, log=lambda _m: None, max_attempts=6)
        sleeps = []
        with mock.patch.object(llm.time, "sleep", sleeps.append), \
                mock.patch.object(c, "_post", return_value=(503, {}, b"down")):
            with self.assertRaises(LLMUnavailableError):
                c.draw_sample("p")
        base = [60, 120, 240, 300, 300]
        self.assertEqual(len(sleeps), len(base))
        for got, want in zip(sleeps, base):
            self.assertGreaterEqual(got, want)
            self.assertLessEqual(got, want * 1.1 + 1e-9)

    def test_retry_after_is_honoured(self):
        c = OpenAICompatLLM(api_key=KEY, log=lambda _m: None, max_attempts=2)
        sleeps = []
        with mock.patch.object(llm.time, "sleep", sleeps.append), \
                mock.patch.object(c, "_post", return_value=(429, {"Retry-After": "500"}, b"")):
            with self.assertRaises(LLMUnavailableError):
                c.draw_sample("p")
        self.assertGreaterEqual(sleeps[0], 500)


if __name__ == "__main__":
    unittest.main()
