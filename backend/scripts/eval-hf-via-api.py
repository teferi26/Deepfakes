"""Evaluate the running API against a Hugging Face image dataset.

This script is intentionally lightweight and uses the same online pipeline as end-users:
/auth/login -> /v1/upload (auto-enqueues analysis) -> /v1/jobs/{job_id} polling.

Defaults are chosen for CIFAKE:
- label "fake" => expected ai_generated
- label "real" => expected not_ai_generated

Run inside the api container, e.g.
  python -u scripts/eval-hf-via-api.py --email ... --password ... --dataset-id yanbax/CIFAKE_autotrain_compatible
"""

from __future__ import annotations

import argparse
import io
import os
import time
from dataclasses import dataclass

import requests


@dataclass
class EvalCounts:
    attempted: int = 0
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    inconclusive: int = 0
    failures: int = 0


def _sleep_with_jitter(seconds: float) -> None:
    # Tiny jitter to avoid synchronizing with the server window.
    try:
        import random

        time.sleep(max(0.0, seconds) + random.uniform(0.0, 0.25))
    except Exception:
        time.sleep(max(0.0, seconds))


def _request_with_429_backoff(
    method: str,
    url: str,
    *,
    max_retries_429: int,
    timeout_s: int,
    **kwargs,
):
    delay_s = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries_429 + 1):
        try:
            r = requests.request(method, url, timeout=timeout_s, **kwargs)
            if r.status_code != 429:
                r.raise_for_status()
                return r

            # 429: Too Many Requests
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    delay_s = max(delay_s, float(retry_after))
                except Exception:
                    pass
            if attempt >= max_retries_429:
                r.raise_for_status()
            _sleep_with_jitter(delay_s)
            delay_s = min(delay_s * 1.6, 20.0)
        except Exception as e:
            last_exc = e
            if attempt >= max_retries_429:
                raise
            _sleep_with_jitter(delay_s)
            delay_s = min(delay_s * 1.6, 20.0)
    if last_exc:
        raise last_exc
    raise RuntimeError("request failed")


def _as_jpeg_bytes(image_obj) -> bytes:
    """Convert HF dataset image into JPEG bytes."""
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow is required inside the container") from e

    if image_obj is None:
        raise ValueError("image is None")

    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)

    if isinstance(image_obj, dict):
        # Some datasets return dicts like {'bytes': ..., 'path': ...}
        if image_obj.get("bytes"):
            return image_obj["bytes"]
        if image_obj.get("path"):
            with open(image_obj["path"], "rb") as f:
                return f.read()

    if hasattr(image_obj, "read"):
        return image_obj.read()

    if isinstance(image_obj, Image.Image):
        img = image_obj
    else:
        # HF often returns PIL images already; try to coerce
        img = Image.fromarray(image_obj)

    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _login(api_base: str, email: str, password: str, timeout_s: int) -> str:
    r = requests.post(
        f"{api_base}/auth/login",
        json={"email": email, "password": password},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _upload(api_base: str, token: str, jpeg_bytes: bytes, timeout_s: int, max_retries_429: int) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("eval.jpg", jpeg_bytes, "image/jpeg")}
    r = _request_with_429_backoff(
        "POST",
        f"{api_base}/v1/upload",
        headers=headers,
        files=files,
        max_retries_429=max_retries_429,
        timeout_s=timeout_s,
    )
    return r.json()["job_id"]


def _poll_result(
    api_base: str,
    token: str,
    job_id: str,
    timeout_s: int,
    poll_s: float,
    max_retries_429: int,
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = _request_with_429_backoff(
            "GET",
            f"{api_base}/v1/jobs/{job_id}",
            headers=headers,
            max_retries_429=max_retries_429,
            timeout_s=30,
        )
        j = r.json()
        if j.get("status") == "success":
            return j.get("result") or {}
        if j.get("status") == "failure":
            raise RuntimeError("job failed")
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def _submit_feedback(api_base: str, token: str, analysis_id: str, label: str, timeout_s: int) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(
        f"{api_base}/v1/history/{analysis_id}/feedback",
        headers=headers,
        json={"label": label, "comment": "auto-eval"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json() if isinstance(r.json(), dict) else {}


def _extract_decision(result: dict) -> tuple[str | None, float | None, dict]:
    # Best-effort parsing across minor schema shifts.
    ai_decision = result.get("ai_decision")
    ai_probability = result.get("ai_probability")

    def _find_first(d, key: str):
        if isinstance(d, dict):
            if key in d and d[key] is not None:
                return d[key]
            for v in d.values():
                found = _find_first(v, key)
                if found is not None:
                    return found
        elif isinstance(d, list):
            for v in d:
                found = _find_first(v, key)
                if found is not None:
                    return found
        return None

    clip_meta: dict = {}
    clip_obj = result.get("clip")
    if isinstance(clip_obj, dict):
        clip_meta = clip_obj

    # Fallbacks
    if ai_probability is None and isinstance(result.get("probability"), (int, float)):
        ai_probability = float(result["probability"])

    return ai_decision, ai_probability, {
        "head_version": (
            clip_meta.get("head_version")
            or result.get("head_version")
            or result.get("clip_head_version")
            or _find_first(result, "head_version")
            or _find_first(result, "clip_head_version")
        ),
        "head_source": (
            clip_meta.get("head_source")
            or result.get("head_source")
            or result.get("clip_head_source")
            or _find_first(result, "head_source")
            or _find_first(result, "clip_head_source")
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--split", default="train")

    p.add_argument("--fake-label", default="fake", help="Class name treated as ai_generated")
    p.add_argument("--real-label", default="real", help="Class name treated as not_ai_generated")

    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--fake-start", type=int, default=2000)
    p.add_argument("--real-start", type=int, default=70000)

    p.add_argument("--request-timeout-s", type=int, default=60)
    p.add_argument("--job-timeout-s", type=int, default=120)
    p.add_argument("--poll-s", type=float, default=1.0)
    p.add_argument("--throttle-s", type=float, default=0.25, help="Sleep between samples to avoid 429")
    p.add_argument("--max-retries-429", type=int, default=20, help="How many times to retry when receiving HTTP 429")
    p.add_argument(
        "--submit-feedback",
        action="store_true",
        default=False,
        help="If set, posts correct labels to /v1/history/{analysis_id}/feedback for each evaluated sample.",
    )
    args = p.parse_args()

    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset_id} ({args.split})")
    ds = load_dataset(args.dataset_id, split=args.split)

    # Infer label names if present
    label_names = None
    try:
        feat = ds.features.get("label")
        if feat is not None and hasattr(feat, "names"):
            label_names = list(feat.names)
    except Exception:
        label_names = None

    if not label_names:
        raise SystemExit("Dataset must have a ClassLabel 'label' feature with names")

    try:
        fake_idx = label_names.index(args.fake_label)
        real_idx = label_names.index(args.real_label)
    except ValueError:
        raise SystemExit(f"Label names {label_names} do not include fake={args.fake_label} real={args.real_label}")

    token = _login(args.api_base, args.email, args.password, timeout_s=args.request_timeout_s)

    counts = EvalCounts()
    cm = {"ai_generated": {"ai_generated": 0, "not_ai_generated": 0, "inconclusive": 0, "other": 0},
          "not_ai_generated": {"ai_generated": 0, "not_ai_generated": 0, "inconclusive": 0, "other": 0}}

    head_seen = {"head_version": None, "head_source": None}

    def run_one(idx: int, expected: str):
        nonlocal counts, head_seen
        counts.attempted += 1
        try:
            row = ds[idx]
            jpeg = _as_jpeg_bytes(row["image"])
            job_id = _upload(
                args.api_base,
                token,
                jpeg,
                timeout_s=args.request_timeout_s,
                max_retries_429=args.max_retries_429,
            )
            result = _poll_result(
                args.api_base,
                token,
                job_id,
                timeout_s=args.job_timeout_s,
                poll_s=args.poll_s,
                max_retries_429=args.max_retries_429,
            )
            decision, prob, head_meta = _extract_decision(result)

            if args.submit_feedback:
                analysis_id = result.get("analysis_id")
                if isinstance(analysis_id, str) and analysis_id:
                    try:
                        _submit_feedback(
                            args.api_base,
                            token,
                            analysis_id=analysis_id,
                            label=expected,
                            timeout_s=args.request_timeout_s,
                        )
                    except Exception as e:
                        # Feedback is best-effort; evaluation should continue.
                        print(f"idx={idx} feedback ERROR: {e}")

            if head_seen["head_version"] is None and head_meta.get("head_version"):
                head_seen = head_meta

            counts.total += 1

            if decision in ("ai_generated", "not_ai_generated"):
                if decision == expected:
                    counts.correct += 1
                else:
                    counts.incorrect += 1
                cm[expected][decision] += 1
            elif decision == "inconclusive":
                counts.inconclusive += 1
                cm[expected]["inconclusive"] += 1
            else:
                counts.failures += 1
                cm[expected]["other"] += 1

            print(f"idx={idx} expected={expected} decision={decision} prob={prob}")
        except Exception as e:
            counts.failures += 1
            cm[expected]["other"] += 1
            print(f"idx={idx} expected={expected} ERROR: {e}")
        finally:
            if args.throttle_s and args.throttle_s > 0:
                time.sleep(args.throttle_s)

    # Pick a slice that should be outside the earlier seed ranges (which were head+tail).
    # CIFAKE is block-ordered: fake first, real later.
    fake_indices = []
    i = args.fake_start
    while len(fake_indices) < args.n_per_class and i < len(ds):
        row = ds[i]
        if int(row["label"]) == fake_idx:
            fake_indices.append(i)
        i += 1

    real_indices = []
    i = args.real_start
    while len(real_indices) < args.n_per_class and i < len(ds):
        row = ds[i]
        if int(row["label"]) == real_idx:
            real_indices.append(i)
        i += 1

    if len(fake_indices) < args.n_per_class or len(real_indices) < args.n_per_class:
        raise SystemExit(f"Could not collect enough samples per class. got fake={len(fake_indices)} real={len(real_indices)}")

    print(f"Evaluating {len(fake_indices)} fake + {len(real_indices)} real via API...")

    for idx in fake_indices:
        run_one(idx, expected="ai_generated")

    for idx in real_indices:
        run_one(idx, expected="not_ai_generated")

    decided = counts.correct + counts.incorrect
    decided_acc = (counts.correct / decided) if decided else 0.0
    overall_acc = (counts.correct / counts.total) if counts.total else 0.0

    print("\n=== Summary ===")
    print(f"Head seen: version={head_seen.get('head_version')} source={head_seen.get('head_source')}")
    print(f"Attempted: {counts.attempted}")
    print(f"Total: {counts.total}")
    print(f"Correct: {counts.correct}")
    print(f"Incorrect: {counts.incorrect}")
    print(f"Inconclusive: {counts.inconclusive}")
    print(f"Failures/Other: {counts.failures}")
    print(f"Accuracy (overall): {overall_acc:.3f}")
    print(f"Accuracy (decided only): {decided_acc:.3f}")

    print("\nConfusion matrix (rows=expected, cols=decision):")
    for expected in ("ai_generated", "not_ai_generated"):
        row = cm[expected]
        print(
            f"{expected}: ai_generated={row['ai_generated']} not_ai_generated={row['not_ai_generated']} "
            f"inconclusive={row['inconclusive']} other={row['other']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
