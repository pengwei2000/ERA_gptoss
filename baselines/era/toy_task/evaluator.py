"""Toy task evaluator: 1-D regression, score = -RMSE (higher is better).

Same contract as every SimpleTES evaluator: evaluate(program_path) -> dict with
'combined_score'. Runs in seconds with the standard library only, so the ERA
pipeline can be exercised anywhere. It also reports whether any *API_KEY*
variable reached the evaluation environment ('saw_api_key'), which the smoke
test requires to be False.
"""

import importlib.util
import math
import os
import random
import sys

FAILURE_SCORE = -10.0


def _data():
    rng = random.Random(0)

    def f(x):
        return math.sin(1.5 * x) + 0.3 * x * x

    xs = [rng.uniform(-3, 3) for _ in range(400)]
    ys = [f(x) + rng.gauss(0, 0.1) for x in xs]
    return xs[:200], ys[:200], xs[200:], [f(x) for x in xs[200:]]


def evaluate(program_path):
    saw_api_key = any("API_KEY" in k.upper() for k in os.environ)
    x_tr, y_tr, x_te, y_te = _data()
    try:
        spec = importlib.util.spec_from_file_location("candidate", program_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod   # as a normal import would; @dataclass looks it up
        spec.loader.exec_module(mod)
        pred = [float(v) for v in mod.fit_predict(list(x_tr), list(y_tr), list(x_te))]
        if len(pred) != len(y_te) or not all(math.isfinite(v) for v in pred):
            raise ValueError(f"need {len(y_te)} finite predictions, got {len(pred)}")
    except Exception as e:  # noqa: BLE001 -- any candidate failure is a failed node
        return {"combined_score": FAILURE_SCORE, "validity": 0.0,
                "error": f"{type(e).__name__}: {e}", "saw_api_key": saw_api_key}
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, y_te)) / len(y_te))
    return {"combined_score": -rmse, "rmse": rmse, "validity": 1.0,
            "saw_api_key": saw_api_key}
