"""Seed a *licensed* Hugging Face dataset into the app's dataset bank.

This script:
- Searches Hugging Face for datasets tagged with an explicit license.
- Tries to auto-detect an image column + a label column.
- Builds an auto-labeled ZIP with folders:
    ai_generated/
    not_ai_generated/
- Uploads it to the API endpoint: POST /v1/dataset/upload-zip-auto

Run inside the running API container (recommended):
  docker compose exec api python scripts/hf-seed-dataset.py \
    --api-base http://localhost:8000 --email you@x.com --password '...' \
    --max-per-class 2000

Or run from host if you have deps installed.

Safety:
- By default, only allowlisted licenses are accepted.
- You can override with --accept-any-license (YOU decide compliance).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import secrets
import sys
import time
import traceback
import zipfile
import os

from dataclasses import dataclass
from typing import Any, Iterable

import requests


ALLOWLIST_LICENSES = {
    # permissive / common
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc0-1.0",
    "bsd-3-clause",
    "bsd-2-clause",
}


def _normalize_license(lic: str) -> set[str]:
    parts = re.split(r"[,;/|]\s*", lic.strip().lower())
    return {p for p in parts if p}


def _hf_api_search_datasets(query: str, limit: int = 50) -> list[dict[str, Any]]:
    # Public HF API; returns tags including license:*.
    url = "https://huggingface.co/api/datasets"
    params = {"search": query, "limit": str(limit)}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    return []


def _hf_api_dataset_license_tag(dataset_id: str) -> str | None:
    # Returns the first license tag (e.g., "mit") if present.
    # HF API response contains a 'tags' list with entries like 'license:mit'.
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json() if isinstance(r.json(), dict) else {}
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        return None
    for t in tags:
        if isinstance(t, str) and t.startswith("license:"):
            lic = t.split(":", 1)[1].strip()
            return lic or None
    return None


def _hf_dataset_license_via_datasets(dataset_id: str) -> str | None:
    # Prefer datasets' builder metadata (often matches dataset card tags).
    from datasets import load_dataset_builder

    builder = load_dataset_builder(dataset_id)
    lic = getattr(builder.info, "license", None)
    if isinstance(lic, str) and lic.strip():
        return lic.strip()
    if isinstance(lic, (list, tuple)) and lic:
        parts = [str(x).strip() for x in lic if str(x).strip()]
        return ",".join(parts) if parts else None
    return None


def _license_ok(license_str: str | None, accept_any_license: bool) -> bool:
    if accept_any_license:
        return bool(license_str and license_str.strip())
    if not license_str:
        return False
    lic_set = _normalize_license(license_str)
    return bool(lic_set & ALLOWLIST_LICENSES)


def _infer_image_and_label_columns(features: Any) -> tuple[str, str] | None:
    # Features is usually a datasets.Features mapping.
    # We look for an Image feature and a label-ish feature.
    try:
        from datasets import Image, ClassLabel

        image_col = None
        label_col = None

        for k, v in features.items():
            if isinstance(v, Image):
                image_col = k
                break

        # First try ClassLabel / common label names
        for k, v in features.items():
            if k in {"label", "labels", "class", "category", "target"}:
                label_col = k
                break
            if isinstance(v, ClassLabel):
                label_col = k
                break

        if image_col and label_col and image_col != label_col:
            return image_col, label_col
    except Exception:
        return None

    return None


def _map_label_to_folder(raw_label: Any, class_names: list[str] | None) -> str | None:
    # Return ai_generated / not_ai_generated / None
    if class_names is not None:
        try:
            idx = int(raw_label)
            if 0 <= idx < len(class_names):
                raw_label = class_names[idx]
        except Exception:
            pass

    if isinstance(raw_label, str):
        low = raw_label.strip().lower()
        if any(tok in low for tok in ["ai", "fake", "synthetic", "generated", "gan", "diffusion"]):
            if "real" not in low:
                return "ai_generated"
        if "real" in low or low in {"not_ai", "not-ai", "authentic"}:
            return "not_ai_generated"
        if low in {"1", "true", "yes"}:
            return "ai_generated"
        if low in {"0", "false", "no"}:
            return "not_ai_generated"

    if isinstance(raw_label, bool):
        return "ai_generated" if raw_label else "not_ai_generated"

    if isinstance(raw_label, (int, float)):
        return "ai_generated" if int(raw_label) == 1 else "not_ai_generated"

    return None


def _as_jpeg_bytes(image: Any) -> bytes:
    from PIL import Image

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    if isinstance(image, (bytes, bytearray)):
        return bytes(image)

    if isinstance(image, dict):
        b = image.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)

        # HF datasets Image feature often yields {"path": "...", "bytes": None}
        p = image.get("path")
        if isinstance(p, str) and p and os.path.exists(p):
            with Image.open(p) as im:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=92)
                return buf.getvalue()

    raise ValueError("Unsupported image type")


def _build_zip(samples: Iterable[tuple[str, bytes, str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder, img_bytes, fname in samples:
            zf.writestr(f"{folder}/{fname}", img_bytes)
    return buf.getvalue()


def _api_register(api_base: str, email: str, password: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/auth/register"
    r = requests.post(url, json={"email": email, "password": password}, timeout=60)
    if r.status_code == 400:
        # already exists
        return {}
    r.raise_for_status()
    return r.json()


def _api_login(api_base: str, email: str, password: str) -> str:
    url = f"{api_base.rstrip('/')}/auth/login"
    r = requests.post(url, json={"email": email, "password": password}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["access_token"]


def _api_upload_zip_auto(api_base: str, token: str, zip_bytes: bytes) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/dataset/upload-zip-auto"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("hf_seed.zip", zip_bytes, "application/zip")},
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def _api_train(api_base: str, token: str) -> str:
    url = f"{api_base.rstrip('/')}/v1/dataset/train"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return r.json()["job_id"]


def _api_get_job(api_base: str, token: str, job_id: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/jobs/{job_id}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return r.json()


@dataclass
class Candidate:
    dataset_id: str
    license: str


def _pick_candidates(query: str, accept_any_license: bool, limit: int = 50) -> list[Candidate]:
    raw = _hf_api_search_datasets(query=query, limit=limit)
    candidates: list[Candidate] = []
    for item in raw:
        dataset_id = item.get("id")
        tags = item.get("tags") or []
        if not dataset_id or not isinstance(tags, list):
            continue
        lic_tags = [t for t in tags if isinstance(t, str) and t.startswith("license:")]
        if not lic_tags:
            continue
        lic = lic_tags[0].split(":", 1)[1].strip()
        if not lic:
            continue

        if not _license_ok(lic, accept_any_license):
            continue

        candidates.append(Candidate(dataset_id=dataset_id, license=lic))

    # De-dup
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        if c.dataset_id in seen:
            continue
        seen.add(c.dataset_id)
        out.append(c)
    return out


def _try_load_samples(
    dataset_id: str,
    max_per_class: int,
    *,
    streaming_first: bool = True,
    debug: bool = False,
) -> tuple[list[tuple[str, bytes, str]], str] | None:
    from datasets import load_dataset, load_dataset_builder

    builder = load_dataset_builder(dataset_id)
    features = builder.info.features
    inferred = _infer_image_and_label_columns(features)
    if inferred is None:
        return None
    image_col, label_col = inferred

    class_names = None
    try:
        from datasets import ClassLabel

        feat = features[label_col]
        if isinstance(feat, ClassLabel):
            class_names = list(feat.names)
    except Exception:
        pass

    # pick a split
    split = "train"
    try:
        splits = list(builder.info.splits.keys())
        if "train" not in splits and splits:
            split = splits[0]
    except Exception:
        pass

    num_examples = None
    try:
        sp = builder.info.splits.get(split)
        num_examples = getattr(sp, "num_examples", None)
    except Exception:
        num_examples = None

    def _iter_rows():
        # 1) streaming (cheap), 2) fallback to a sliced non-streaming subset
        if streaming_first:
            try:
                yield from load_dataset(dataset_id, split=split, streaming=True)
                return
            except Exception as e:
                if debug:
                    print("[debug] streaming load failed")
                    traceback.print_exc()
                else:
                    print(f"[warn] streaming load failed: {e}")

        # Fallback: download only subsets (non-streaming).
        # Some datasets are block-ordered by class (e.g., all label 0 then all label 1),
        # so scanning only the head slice can miss an entire class.
        max_scan = max(5000, max_per_class * 40)

        head = f"{split}[:{max_scan}]"
        if debug:
            print(f"[debug] trying non-streaming head slice: {head}")
        yield from load_dataset(dataset_id, split=head, streaming=False)

        # Also try a middle slice to better cover block-ordered datasets.
        # Example: CIFAKE has all 0s then all 1s; head+tail alone can bias
        # towards the extreme ends. The midpoint often straddles the boundary.
        if isinstance(num_examples, int) and num_examples > (max_scan * 3):
            mid_start = max(0, (num_examples // 2) - (max_scan // 2))
            mid_end = min(num_examples, mid_start + max_scan)
            mid = f"{split}[{mid_start}:{mid_end}]"
            if debug:
                print(f"[debug] trying non-streaming mid slice: {mid}")
            yield from load_dataset(dataset_id, split=mid, streaming=False)

        if isinstance(num_examples, int) and num_examples > max_scan:
            start = max(0, num_examples - max_scan)
            tail = f"{split}[{start}:{num_examples}]"
            if debug:
                print(f"[debug] trying non-streaming tail slice: {tail}")
            yield from load_dataset(dataset_id, split=tail, streaming=False)

    ds = _iter_rows()

    ai_count = 0
    real_count = 0
    out: list[tuple[str, bytes, str]] = []

    scanned = 0
    for row in ds:
        scanned += 1
        folder = _map_label_to_folder(row.get(label_col), class_names)
        if folder is None:
            continue

        if folder == "ai_generated" and ai_count >= max_per_class:
            continue
        if folder == "not_ai_generated" and real_count >= max_per_class:
            continue

        try:
            img_bytes = _as_jpeg_bytes(row.get(image_col))
        except Exception:
            continue

        if folder == "ai_generated":
            ai_count += 1
            fname = f"ai_{ai_count:06d}.jpg"
        else:
            real_count += 1
            fname = f"real_{real_count:06d}.jpg"

        out.append((folder, img_bytes, fname))

        if ai_count >= max_per_class and real_count >= max_per_class:
            break

    if ai_count == 0 or real_count == 0:
        return None

    meta = {
        "dataset_id": dataset_id,
        "split": split,
        "image_col": image_col,
        "label_col": label_col,
        "rows_scanned": scanned,
        "ai": ai_count,
        "real": real_count,
    }
    return out, json.dumps(meta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--email", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--auto-user", action="store_true", default=False)
    ap.add_argument("--query", default="ai generated real images")
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--accept-any-license", action="store_true", default=False)
    ap.add_argument("--dataset-id", default="")
    ap.add_argument("--no-streaming", action="store_true", default=False)
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--train-after", action="store_true", default=False)
    ap.add_argument("--train-poll-seconds", type=int, default=5)
    ap.add_argument("--train-timeout-seconds", type=int, default=30 * 60)
    args = ap.parse_args()

    if args.auto_user or (not args.email and not args.password):
        # Avoid passing real credentials; create a throwaway user.
        # Email must be unique.
        nonce = secrets.token_hex(4)
        args.email = f"hfseed+{nonce}@example.com"
        args.password = secrets.token_urlsafe(18)
        print(f"Using auto-user email={args.email}")

    if not args.email or not args.password:
        print("Provide --email and --password, or use --auto-user")
        return 2

    # Ensure user exists; then login to read api_key.
    try:
        _api_register(args.api_base, args.email, args.password)
    except Exception as e:
        # likely already exists
        print(f"Register warning: {e}")

    token = _api_login(args.api_base, args.email, args.password)

    if args.dataset_id:
        dataset_ids = [args.dataset_id]
    else:
        cands = _pick_candidates(args.query, accept_any_license=args.accept_any_license, limit=args.limit)
        dataset_ids = [c.dataset_id for c in cands]

    if not dataset_ids:
        print("No candidates found with explicit license metadata.")
        return 3

    for dataset_id in dataset_ids:
        lic_builder = None
        lic_tag = None
        try:
            lic_builder = _hf_dataset_license_via_datasets(dataset_id)
        except Exception:
            lic_builder = None
        try:
            lic_tag = _hf_api_dataset_license_tag(dataset_id)
        except Exception:
            lic_tag = None

        # Accept if either source provides an allowlisted license.
        lic_effective = lic_builder or lic_tag
        if not _license_ok(lic_effective, args.accept_any_license):
            if not lic_effective:
                print(f"[skip] {dataset_id}: no license metadata (builder+tag)")
            else:
                print(f"[skip] {dataset_id}: license '{lic_effective}' not allowed")
            continue

        print(f"[try] {dataset_id} (license_tag={lic_tag or ''} license_builder={lic_builder or ''})")

        try:
            loaded = _try_load_samples(
                dataset_id,
                args.max_per_class,
                streaming_first=(not args.no_streaming),
                debug=bool(args.debug),
            )
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            print(f"[skip] {dataset_id}: load failed: {e}")
            continue

        if loaded is None:
            print(f"[skip] {dataset_id}: could not infer/parse 2-class image dataset")
            continue

        samples, meta = loaded
        zbytes = _build_zip(samples)

        # Include a small metadata file inside the ZIP for traceability.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            meta_obj = json.loads(meta)
            meta_obj["license_tag"] = lic_tag
            meta_obj["license_builder"] = lic_builder
            zf.writestr("_meta.json", json.dumps(meta_obj))
            for folder, img_bytes, fname in samples:
                zf.writestr(f"{folder}/{fname}", img_bytes)
        zbytes = buf.getvalue()

        print(f"Uploading ZIP ({len(zbytes)/1024/1024:.1f}MB) to /v1/dataset/upload-zip-auto")
        resp = _api_upload_zip_auto(args.api_base, token, zbytes)
        print(json.dumps(resp, indent=2))

        if args.train_after:
            job_id = _api_train(args.api_base, token)
            print(f"Train enqueued job_id={job_id}")
            deadline = time.time() + float(args.train_timeout_seconds)
            while time.time() < deadline:
                job = _api_get_job(args.api_base, token, job_id)
                status = (job.get("status") or "").lower()
                if status == "success":
                    result = job.get("result")
                    print(json.dumps(result, indent=2))
                    # If the train task reports below-target, return non-zero.
                    if isinstance(result, dict) and result.get("reason") == "below_target_accuracy":
                        return 10
                    return 0
                if status == "failure":
                    print(json.dumps(job, indent=2))
                    return 11
                time.sleep(max(1, int(args.train_poll_seconds)))
            print("Training poll timeout")
            return 12

        return 0

    print("No dataset succeeded (license+parsing). Try --dataset-id <id> or --accept-any-license if appropriate.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
