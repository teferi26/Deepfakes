#!/usr/bin/env python3
"""End-to-end smoke test (simula la interfaz):

1) /auth/register
2) /auth/login
3) /v1/upload (auto encola análisis)
4) /v1/jobs/{job_id} hasta success

Sube 1 imagen AI (DiffusionDB) y 1 imagen real (Hemg) para verificar:
- que el pipeline funciona
- que CLIPDetector usa el head nuevo en DB (model_artifacts)

Uso:
  python scripts/e2e_interface_smoketest.py
"""

import io
import time
import requests
from datasets import load_dataset

BASE = "http://api:8000"
PASSWORD = "testpassword123"


def _auth_headers():
    email = f"autotest_{int(time.time())}@example.com"

    r = requests.post(f"{BASE}/auth/register", json={"email": email, "password": PASSWORD}, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"register failed {r.status_code}: {r.text}")

    r = requests.post(f"{BASE}/auth/login", json={"email": email, "password": PASSWORD}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"login failed {r.status_code}: {r.text}")

    token = r.json()["access_token"]
    return email, {"Authorization": f"Bearer {token}"}


def _upload_png(pil_img, filename: str, headers: dict):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    data = buf.getvalue()

    files = {"file": (filename, data, "image/png")}
    r = requests.post(f"{BASE}/v1/upload", headers=headers, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"upload failed {r.status_code}: {r.text}")
    return r.json()


def _wait_job(job_id: str, headers: dict, timeout_s: int = 600):
    t0 = time.time()
    while True:
        r = requests.get(f"{BASE}/v1/jobs/{job_id}", headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"job get failed {r.status_code}: {r.text}")

        payload = r.json()
        status = payload.get("status")
        if status in ("success", "failure"):
            return payload

        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"timeout waiting job {job_id}. last={payload}")

        time.sleep(2)


def _extract_clip_head_version(job_result: dict):
    result = (job_result or {}).get("result") or {}
    ensemble = result.get("ensemble_details") or {}
    for r in ensemble.get("individual_results") or []:
        # Formato del API: {name, probability, confidence, weight, extra{head_version,head_source}}
        if r.get("name") == "clip_universal":
            extra = r.get("extra") or {}
            return extra.get("head_version")
    return None


def _summarize_individual(job_result: dict) -> list[tuple[str, float | None]]:
    result = (job_result or {}).get("result") or {}
    ensemble = result.get("ensemble_details") or {}
    out: list[tuple[str, float | None]] = []
    for r in ensemble.get("individual_results") or []:
        out.append((r.get("name") or "?", r.get("probability")))
    return out


def main():
    email, headers = _auth_headers()
    print("Authenticated as", email)

    # AI sample
    print("\nLoading DiffusionDB sample...")
    ds_ai = load_dataset("poloclub/diffusiondb", "2m_random_1k", split="train", trust_remote_code=True)
    ai_img = ds_ai[0]["image"]
    up = _upload_png(ai_img, "ai_sample.png", headers)
    print("Uploaded AI sample:", {k: up[k] for k in ("job_id", "object_key", "status")})

    res = _wait_job(up["job_id"], headers, timeout_s=900)
    print("AI job status:", res["status"])
    if res.get("result"):
        print("AI probability:", res["result"].get("probability"))
        print("AI decision:", res["result"].get("ai_decision"))
        print("AI clip head version:", _extract_clip_head_version(res))
        print("AI individual:", _summarize_individual(res))

    # REAL sample
    print("\nLoading Hemg real sample...")
    ds_h = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", trust_remote_code=True)
    labels = ds_h["label"]
    real_idxs = [i for i, lbl in enumerate(labels) if lbl == 1]
    # Elegimos aleatorio para evitar algún outlier raro/mal etiquetado.
    import random
    real_idx = random.choice(real_idxs)
    real_img = ds_h[real_idx]["image"]

    up = _upload_png(real_img, "real_sample.png", headers)
    print("Uploaded REAL sample:", {k: up[k] for k in ("job_id", "object_key", "status")})

    res = _wait_job(up["job_id"], headers, timeout_s=900)
    print("REAL job status:", res["status"])
    if res.get("result"):
        print("REAL probability:", res["result"].get("probability"))
        print("REAL decision:", res["result"].get("ai_decision"))
        print("REAL clip head version:", _extract_clip_head_version(res))
        print("REAL individual:", _summarize_individual(res))


if __name__ == "__main__":
    main()
