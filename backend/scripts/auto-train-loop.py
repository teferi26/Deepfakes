"""Automated training loop to reach production accuracy targets.

This script orchestrates the full training pipeline:
1. Seeds data from CIFAKE (up to target per class)
2. Triggers training
3. Evaluates model performance
4. Repeats until targets are met

TARGETS:
- ≤3% inconclusive rate
- ≥97% accuracy on decided samples

Run inside the API container:
  docker compose exec api python scripts/auto-train-loop.py \
    --api-base http://localhost:8000 \
    --email admin@example.com --password 'yourpassword' \
    --max-rounds 10

Or set environment variables and run without args.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Any

import requests


# Target metrics
TARGET_INCONCLUSIVE_RATE = 0.03  # ≤3%
TARGET_ACCURACY = 0.97  # ≥97% on decided samples

# Data seeding configuration
SEED_BATCH_SIZE = 2000  # Samples to add per round
MAX_SAMPLES_PER_CLASS = 10000  # Maximum total samples per class


@dataclass
class Metrics:
    """Evaluation metrics from a single round."""
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    inconclusive: int = 0
    failures: int = 0
    
    @property
    def decided(self) -> int:
        return self.correct + self.incorrect
    
    @property
    def accuracy(self) -> float:
        if self.decided == 0:
            return 0.0
        return self.correct / self.decided
    
    @property
    def inconclusive_rate(self) -> float:
        evaluated = self.decided + self.inconclusive
        if evaluated == 0:
            return 1.0
        return self.inconclusive / evaluated
    
    @property
    def meets_targets(self) -> bool:
        return (
            self.inconclusive_rate <= TARGET_INCONCLUSIVE_RATE and
            self.accuracy >= TARGET_ACCURACY
        )
    
    def __str__(self) -> str:
        return (
            f"Metrics(total={self.total}, correct={self.correct}, incorrect={self.incorrect}, "
            f"inconclusive={self.inconclusive}, failures={self.failures}, "
            f"accuracy={self.accuracy:.2%}, inconclusive_rate={self.inconclusive_rate:.2%})"
        )


@dataclass
class TrainingState:
    """Persistent state across training rounds."""
    round_number: int = 0
    total_seeded: dict = field(default_factory=lambda: {"ai": 0, "real": 0})
    metrics_history: list = field(default_factory=list)
    best_accuracy: float = 0.0
    best_inconclusive_rate: float = 1.0


def _sleep_with_jitter(seconds: float) -> None:
    import random
    time.sleep(max(0.0, seconds) + random.uniform(0.0, 0.25))


def _request_with_retry(
    method: str,
    url: str,
    max_retries: int = 5,
    timeout_s: int = 300,
    **kwargs,
):
    """Make HTTP request with retry logic for 429 and transient errors."""
    delay_s = 1.0
    last_exc = None
    
    for attempt in range(max_retries + 1):
        try:
            r = requests.request(method, url, timeout=timeout_s, **kwargs)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "5")
                try:
                    delay_s = max(delay_s, float(retry_after))
                except:
                    pass
                if attempt >= max_retries:
                    r.raise_for_status()
                print(f"  Rate limited, waiting {delay_s:.1f}s...")
                _sleep_with_jitter(delay_s)
                delay_s = min(delay_s * 1.5, 30.0)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            print(f"  Request failed ({e}), retrying in {delay_s:.1f}s...")
            _sleep_with_jitter(delay_s)
            delay_s = min(delay_s * 1.5, 30.0)
    
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed")


class TrainingOrchestrator:
    """Orchestrates the automated training loop."""
    
    def __init__(
        self,
        api_base: str,
        email: str,
        password: str,
        dataset_id: str = "yanbax/CIFAKE_autotrain_compatible",
        eval_samples: int = 200,
    ):
        self.api_base = api_base.rstrip("/")
        self.email = email
        self.password = password
        self.dataset_id = dataset_id
        self.eval_samples = eval_samples
        self.token = None
        self.state = TrainingState()
    
    def login(self) -> None:
        """Authenticate with the API."""
        print(f"Logging in as {self.email}...")
        
        # Try to register first (in case user doesn't exist)
        try:
            _request_with_retry(
                "POST",
                f"{self.api_base}/auth/register",
                json={"email": self.email, "password": self.password},
                max_retries=1,
                timeout_s=30,
            )
            print("  Registered new user")
        except:
            pass  # User already exists
        
        # Login
        r = _request_with_retry(
            "POST",
            f"{self.api_base}/auth/login",
            json={"email": self.email, "password": self.password},
            timeout_s=30,
        )
        self.token = r.json()["access_token"]
        print("  Logged in successfully")
    
    def get_dataset_stats(self) -> dict:
        """Get current dataset statistics."""
        r = _request_with_retry(
            "GET",
            f"{self.api_base}/v1/dataset/stats",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout_s=30,
        )
        return r.json()
    
    def seed_data(self, samples_per_class: int) -> dict:
        """Seed training data from HuggingFace dataset."""
        print(f"\n📦 Seeding {samples_per_class} samples per class from {self.dataset_id}...")
        
        # This uses the internal seeding mechanism
        # We'll shell out to hf-seed-dataset.py
        import subprocess
        
        cmd = [
            sys.executable,
            "scripts/hf-seed-dataset.py",
            "--api-base", self.api_base,
            "--email", self.email,
            "--password", self.password,
            "--dataset-id", self.dataset_id,
            "--max-per-class", str(samples_per_class),
            # Don't pass --train-after, we'll train ourselves after seeding
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
        )
        
        if result.returncode != 0:
            print(f"  Warning: Seeding returned non-zero: {result.stderr}")
        else:
            print(f"  Seeding output: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
        
        return {"status": "completed", "output": result.stdout}
    
    def trigger_training(self) -> str:
        """Trigger model training and return job ID."""
        print("\n🏋️ Triggering training...")
        
        r = _request_with_retry(
            "POST",
            f"{self.api_base}/v1/dataset/train",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout_s=60,
        )
        job_id = r.json()["job_id"]
        print(f"  Training job started: {job_id}")
        return job_id
    
    def wait_for_job(self, job_id: str, timeout_minutes: int = 30) -> dict:
        """Wait for a job to complete."""
        print(f"  Waiting for job {job_id}...")
        
        start = time.time()
        timeout_s = timeout_minutes * 60
        
        while time.time() - start < timeout_s:
            r = _request_with_retry(
                "GET",
                f"{self.api_base}/v1/jobs/{job_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout_s=30,
            )
            data = r.json()
            status = data.get("status", "unknown")
            
            if status in ("success", "completed", "trained"):
                print(f"  Job completed: {status}")
                return data
            elif status in ("failed", "error"):
                print(f"  Job failed: {data}")
                return data
            
            # Still running
            time.sleep(5)
        
        raise TimeoutError(f"Job {job_id} timed out after {timeout_minutes} minutes")
    
    def evaluate_model(self) -> Metrics:
        """Evaluate current model on a held-out sample from CIFAKE."""
        print(f"\n📊 Evaluating model on {self.eval_samples} samples...")
        
        from datasets import load_dataset
        from PIL import Image
        import io
        
        # Load a fresh split for evaluation
        ds = load_dataset(self.dataset_id, split="test", streaming=True)
        
        metrics = Metrics()
        
        for i, sample in enumerate(ds):
            if i >= self.eval_samples:
                break
            
            metrics.total += 1
            
            try:
                # Get image and label
                img = sample.get("image")
                label = sample.get("label")
                
                # Convert label
                if isinstance(label, int):
                    expected = "ai_generated" if label == 1 else "not_ai_generated"
                elif isinstance(label, str):
                    low = label.lower()
                    if "fake" in low or "ai" in low:
                        expected = "ai_generated"
                    else:
                        expected = "not_ai_generated"
                else:
                    expected = "ai_generated" if label else "not_ai_generated"
                
                # Convert image to bytes
                if isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=92)
                    img_bytes = buf.getvalue()
                elif isinstance(img, dict) and img.get("bytes"):
                    img_bytes = img["bytes"]
                else:
                    metrics.failures += 1
                    continue
                
                # Upload and analyze
                r = _request_with_retry(
                    "POST",
                    f"{self.api_base}/v1/upload",
                    headers={"Authorization": f"Bearer {self.token}"},
                    files={"file": ("eval.jpg", img_bytes, "image/jpeg")},
                    timeout_s=60,
                )
                job_id = r.json()["job_id"]
                
                # Poll for result
                result = self._poll_analysis(job_id)
                
                if result is None:
                    metrics.failures += 1
                    continue
                
                ai_decision = result.get("ai_decision", "inconclusive")
                
                if ai_decision == "inconclusive":
                    metrics.inconclusive += 1
                elif ai_decision == expected:
                    metrics.correct += 1
                else:
                    metrics.incorrect += 1
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{self.eval_samples} - {metrics}")
                    
            except Exception as e:
                print(f"  Sample {i} failed: {e}")
                metrics.failures += 1
        
        return metrics
    
    def _poll_analysis(self, job_id: str, timeout_s: int = 120) -> dict | None:
        """Poll for analysis result."""
        start = time.time()
        
        while time.time() - start < timeout_s:
            try:
                r = _request_with_retry(
                    "GET",
                    f"{self.api_base}/v1/jobs/{job_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout_s=30,
                )
                data = r.json()
                status = data.get("status", "unknown")
                
                if status in ("success", "completed"):
                    return data.get("result", data)
                elif status in ("failed", "error"):
                    return None
                
                time.sleep(2)
            except:
                time.sleep(2)
        
        return None
    
    def run_training_loop(self, max_rounds: int = 10) -> bool:
        """Run the full training loop until targets are met."""
        print("\n" + "=" * 60)
        print("🚀 AUTOMATED TRAINING LOOP")
        print(f"   Target accuracy: ≥{TARGET_ACCURACY:.0%}")
        print(f"   Target inconclusive rate: ≤{TARGET_INCONCLUSIVE_RATE:.0%}")
        print("=" * 60)
        
        self.login()
        
        # Get initial stats
        try:
            stats = self.get_dataset_stats()
            print(f"\n📈 Initial dataset stats: {stats}")
        except Exception as e:
            print(f"  Could not get stats: {e}")
        
        for round_num in range(1, max_rounds + 1):
            self.state.round_number = round_num
            print(f"\n{'='*60}")
            print(f"📍 ROUND {round_num}/{max_rounds}")
            print("=" * 60)
            
            # Step 1: Seed more data if needed
            current_samples = self.state.total_seeded.get("ai", 0) + self.state.total_seeded.get("real", 0)
            target_samples = min(SEED_BATCH_SIZE * round_num, MAX_SAMPLES_PER_CLASS)
            
            if current_samples < target_samples * 2:  # *2 because both classes
                samples_to_add = min(SEED_BATCH_SIZE, MAX_SAMPLES_PER_CLASS - current_samples // 2)
                if samples_to_add > 0:
                    self.seed_data(samples_to_add)
                    self.state.total_seeded["ai"] += samples_to_add
                    self.state.total_seeded["real"] += samples_to_add
            
            # Step 2: Trigger training
            try:
                job_id = self.trigger_training()
                result = self.wait_for_job(job_id)
                print(f"  Training result: {json.dumps(result, indent=2)[:500]}")
            except Exception as e:
                print(f"  Training failed: {e}")
                continue
            
            # Step 3: Evaluate
            metrics = self.evaluate_model()
            self.state.metrics_history.append({
                "round": round_num,
                "metrics": str(metrics),
                "accuracy": metrics.accuracy,
                "inconclusive_rate": metrics.inconclusive_rate,
            })
            
            # Track best
            if metrics.accuracy > self.state.best_accuracy:
                self.state.best_accuracy = metrics.accuracy
            if metrics.inconclusive_rate < self.state.best_inconclusive_rate:
                self.state.best_inconclusive_rate = metrics.inconclusive_rate
            
            # Report
            print(f"\n📊 Round {round_num} Results:")
            print(f"   {metrics}")
            print(f"   Best accuracy so far: {self.state.best_accuracy:.2%}")
            print(f"   Best inconclusive rate so far: {self.state.best_inconclusive_rate:.2%}")
            
            # Check if targets are met
            if metrics.meets_targets:
                print("\n" + "🎉" * 20)
                print("✅ TARGETS MET!")
                print(f"   Accuracy: {metrics.accuracy:.2%} (target: ≥{TARGET_ACCURACY:.0%})")
                print(f"   Inconclusive: {metrics.inconclusive_rate:.2%} (target: ≤{TARGET_INCONCLUSIVE_RATE:.0%})")
                print("🎉" * 20)
                return True
            
            # Gap to target
            acc_gap = TARGET_ACCURACY - metrics.accuracy
            inc_gap = metrics.inconclusive_rate - TARGET_INCONCLUSIVE_RATE
            print(f"\n   Gap to target:")
            print(f"   - Accuracy: {acc_gap:+.2%}")
            print(f"   - Inconclusive: {inc_gap:+.2%}")
        
        print("\n" + "=" * 60)
        print("⚠️  MAX ROUNDS REACHED - TARGETS NOT MET")
        print(f"   Best accuracy: {self.state.best_accuracy:.2%}")
        print(f"   Best inconclusive rate: {self.state.best_inconclusive_rate:.2%}")
        print("=" * 60)
        
        # Print history
        print("\n📈 Training History:")
        for h in self.state.metrics_history:
            print(f"   Round {h['round']}: acc={h['accuracy']:.2%}, inconclusive={h['inconclusive_rate']:.2%}")
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Automated training loop for AI image detection")
    parser.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"),
                        help="Base URL of the API")
    parser.add_argument("--email", default=os.getenv("ADMIN_EMAIL", "admin@example.com"),
                        help="Admin email for authentication")
    parser.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", ""),
                        help="Admin password")
    parser.add_argument("--dataset-id", default="yanbax/CIFAKE_autotrain_compatible",
                        help="HuggingFace dataset ID for training data")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum training rounds")
    parser.add_argument("--eval-samples", type=int, default=200,
                        help="Number of samples for evaluation")
    
    args = parser.parse_args()
    
    if not args.password:
        print("Error: --password is required (or set ADMIN_PASSWORD env var)")
        sys.exit(1)
    
    orchestrator = TrainingOrchestrator(
        api_base=args.api_base,
        email=args.email,
        password=args.password,
        dataset_id=args.dataset_id,
        eval_samples=args.eval_samples,
    )
    
    success = orchestrator.run_training_loop(max_rounds=args.max_rounds)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
