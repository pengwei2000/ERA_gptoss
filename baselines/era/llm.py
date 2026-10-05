"""LLM client for the ERA baseline: gpt-oss-120b over an OpenAI-compatible API.

CHANGED vs upstream (google-research/era implementation/llm.py @ b836730):
  * `GeminiLLM` (google-genai SDK) is replaced by `OpenAICompatLLM`, a stdlib-only
    client for any OpenAI-compatible /chat/completions endpoint (AkashML by
    default). No third-party dependency.
  * Sampling parameters are sent explicitly and match the SimpleTES ZAPBench runs:
    temperature 1.0, top_p 1.0 (vLLM's default), reasoning_effort "high".
    Upstream sent none (Gemini defaults).
  * Token budget. SimpleTES served gpt-oss with a 49152-token context
    (vLLM --max-model-len), so prompt + reasoning + answer <= 49152. An API's
    max_tokens counts generated tokens only, so each call here sends
    max_tokens = min(49152, context_window - prompt tokens), estimating the
    prompt from the server-reported usage.prompt_tokens of earlier calls. What
    cannot be reproduced over a chat API is SimpleTES's forced split of that
    budget into reasoning (<= 33616 - prompt) and answer (15536).
  * Retry policy. Upstream retried only HTTP 429, 5 attempts, 5 s * 2^k backoff
    (~75 s in total), and re-raised everything else -- which ends the whole
    search. Here every server-side / transient failure (network errors,
    timeouts, 408/409/425/429, any 5xx, malformed 200 bodies) is retried with
    minutes-scale backoff (default 60 s doubling to a 300 s cap, 10 attempts,
    ~40 min), honouring Retry-After. Client errors (400/401/403/404/422...) and
    redirects are raised immediately: they are configuration mistakes and must
    fail loud (redirects are never followed, so the key is never re-sent).
  * A well-formed reply with no content (e.g. finish_reason "length" after the
    reasoning used the whole budget) is returned as "" and becomes a failed
    node. Upstream would crash here (re.sub on response.text = None) and end
    the search.
UNCHANGED: the `LLM` protocol, the prompt preamble wrapped around every prompt,
and the markdown-fence cleanup regexes are copied verbatim from upstream.

The API key is passed in by the caller and is never logged or written anywhere.
"""

import http.client
import json
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Protocol, Tuple


class LLM(Protocol):
    def draw_sample(self, prompt: str) -> str:
        ...


class LLMRequestError(RuntimeError):
    """Non-retryable API error (bad key, bad model name, bad parameters)."""


class LLMUnavailableError(RuntimeError):
    """The API kept failing server-side after every retry."""


# Transient 4xx worth waiting out; every 5xx is retried too. Any other status
# (other 4xx, and 3xx because redirects are not followed) is a caller mistake
# and is raised immediately.
RETRYABLE_4XX = frozenset({408, 409, 425, 429})

# Before the first reply reports usage.prompt_tokens, assume this many tokens
# per prompt character plus a chat-template allowance. o200k (gpt-oss) measured
# 0.237-0.268 on the ZAPBench prompts of the SimpleTES runs; 0.28 stays just
# above that, so the first calls after a (re)start neither over-grant nor
# starve long prompts.
_INITIAL_TOKENS_PER_CHAR = 0.28
_TEMPLATE_TOKENS = 128


def _retryable(status: int) -> bool:
    return status >= 500 or status in RETRYABLE_4XX


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None   # surface the 3xx as an HTTPError instead of re-sending the key


def _preamble(prompt: str) -> str:
    # Verbatim from upstream GeminiLLM.draw_sample.
    return f"""
You are an expert Data Scientist and Python programmer.
Your task is to write Python code to solve a machine learning problem.
Return ONLY the python code.

--- BEGIN PROMPT ---
{prompt}
--- END PROMPT ---
"""


def _clean(content: str) -> str:
    # Verbatim from upstream GeminiLLM.draw_sample.
    # Clean up markdown code blocks if present
    content = re.sub(r'^```python\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
    return content


class OpenAICompatLLM:
    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b",
        api_base: str = "https://api.akashml.com/v1",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 49152,
        context_window: Optional[int] = 49152,
        reasoning_effort: Optional[str] = "high",
        request_timeout: float = 1800.0,
        max_attempts: int = 10,
        retry_base_delay: float = 60.0,
        retry_max_delay: float = 300.0,
        log: Callable[[str], None] = print,
    ):
        api_key = (api_key or "").strip()
        if not api_key:
            raise LLMRequestError("empty API key")
        if not (api_key.isascii() and api_key.isprintable()) or " " in api_key:
            # Never echo the key: http.client would put it in its error message.
            raise LLMRequestError(f"API key contains whitespace, control or non-ASCII "
                                  f"characters (length {len(api_key)}); re-export it")
        self._api_key = api_key
        self.model_name = model_name
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.context_window = context_window
        self._tokens_per_char = None   # learned from usage.prompt_tokens
        self._ratio_lock = threading.Lock()   # the client may be shared by threads
        self.reasoning_effort = reasoning_effort
        self.request_timeout = request_timeout
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.log = log
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:  # never show the key
        return (f"OpenAICompatLLM(model={self.model_name!r}, url={self.url!r}, "
                f"temperature={self.temperature}, top_p={self.top_p}, "
                f"max_tokens={self.max_tokens}, context_window={self.context_window}, "
                f"reasoning_effort={self.reasoning_effort!r})")

    def draw_sample(self, prompt: str) -> str:
        return self.draw_sample_with_info(prompt)[0]

    def draw_sample_with_info(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """ERA's call: upstream preamble around the prompt, upstream fence cleanup
        on the answer. Returns (cleaned code, call metadata)."""
        content, meta = self.complete(_preamble(prompt))
        return _clean(content), meta

    def complete(self, full_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """One chat completion of `full_prompt` sent as-is (no ERA preamble or
        cleanup), with this client's sampling settings, token budget and retry
        policy. Also used by the SimpleTES-API launcher (not in this branch) so SimpleTES
        calls the API exactly the way ERA does. Returns (content, call metadata). Raises
        LLMRequestError / LLMUnavailableError; never returns on a failed call."""
        body = self.request_body(full_prompt)
        errors = []
        t_start = time.time()
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            retry_after = None
            try:
                status, headers, raw = self._post(body)
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    ConnectionError, http.client.HTTPException, OSError) as e:
                errors.append(f"attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
            else:
                if status == 200:
                    parsed = _parse_completion(raw)
                    if parsed is not None:
                        content, meta = parsed
                        self._learn_prompt_size(full_prompt, meta.get("prompt_tokens"))
                        meta.update(attempts=attempt, latency_s=round(time.time() - t0, 2),
                                    total_s=round(time.time() - t_start, 2), errors=errors,
                                    max_tokens_sent=body["max_tokens"])
                        return content, meta
                    errors.append(f"attempt {attempt}: HTTP 200 with malformed body: "
                                  f"{raw[:200]!r}")
                elif _retryable(status):
                    errors.append(f"attempt {attempt}: HTTP {status}: {raw[:200]!r}")
                    retry_after = _retry_after_seconds(headers)
                else:
                    raise LLMRequestError(
                        f"HTTP {status} from {self.url} (not retryable; check the API "
                        f"key, model name, parameters and --api-base): {raw[:500]!r}")
            if attempt == self.max_attempts:
                break
            delay = min(self.retry_base_delay * 2 ** (attempt - 1), self.retry_max_delay)
            if retry_after is not None:
                delay = max(delay, min(retry_after, 900.0))
            delay += random.uniform(0, 0.1 * delay)
            self.log(f"  [llm] {errors[-1]} -- retrying in {delay:.0f}s "
                     f"(attempt {attempt + 1}/{self.max_attempts})")
            time.sleep(delay)
        raise LLMUnavailableError(
            f"{self.max_attempts} attempts failed over {time.time() - t_start:.0f}s; "
            f"last errors: {errors[-3:]}")

    def completion_cap(self, full_prompt: str) -> int:
        """max_tokens for this prompt: the configured cap, shrunk so that prompt +
        completion fits the context window (as on SimpleTES's vLLM server)."""
        if not self.context_window:
            return self.max_tokens
        per_char = self._tokens_per_char or _INITIAL_TOKENS_PER_CHAR
        prompt_tokens = int(len(full_prompt) * per_char) + _TEMPLATE_TOKENS
        return max(1, min(self.max_tokens, self.context_window - prompt_tokens))

    def _learn_prompt_size(self, full_prompt: str, prompt_tokens: Optional[int]) -> None:
        if isinstance(prompt_tokens, int) and prompt_tokens > 0 and full_prompt:
            observed = max(0, prompt_tokens - _TEMPLATE_TOKENS) / len(full_prompt)
            # keep the largest ratio seen: never over-grant tokens on a later call
            with self._ratio_lock:
                self._tokens_per_char = max(self._tokens_per_char or 0.0, observed)

    def request_body(self, full_prompt: str) -> Dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.completion_cap(full_prompt),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _post(self, body: Dict[str, Any]) -> Tuple[int, Any, bytes]:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "era-baseline/1.0",
            },
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.request_timeout) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
            except (OSError, http.client.HTTPException):
                body = b""
            finally:
                e.close()
            return e.code, e.headers, body
        except (ValueError, UnicodeError):
            # e.g. a malformed header value; the message would contain the key
            raise LLMRequestError("could not build the HTTP request (malformed API key "
                                  "or --api-base?)") from None


def _parse_completion(raw: bytes) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(content, metadata) from an OpenAI-style body, or None if malformed.

    A well-formed reply with empty content (e.g. finish_reason="length" after the
    reasoning used the whole budget) is NOT malformed: it is the model's answer,
    and the resulting empty program scores as a failed node. (Upstream would
    crash on it instead -- see the module docstring.)
    """
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if not isinstance(message, dict):
        return None
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = data.get("usage") or {}
    return content, {
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "reasoning": reasoning,
    }


def _retry_after_seconds(headers) -> Optional[float]:
    try:
        value = headers.get("Retry-After") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
