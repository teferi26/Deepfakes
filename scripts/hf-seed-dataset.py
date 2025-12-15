"""Host-side wrapper.

The fully supported version of this script lives in the backend image at:
  backend/scripts/hf-seed-dataset.py

Reason: the backend Docker image has the required deps (datasets, torch, etc).

Recommended usage:
  docker compose up --build -d
  docker compose exec api python scripts/hf-seed-dataset.py --help
"""

if __name__ == "__main__":
    print(
        "Run this inside the API container:\n"
        "  docker compose exec api python scripts/hf-seed-dataset.py --help\n"
    )
    raise SystemExit(0)
