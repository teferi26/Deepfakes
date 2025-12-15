"""Reset per-user decision thresholds (UserCalibration) to safe defaults.

Run inside api container:
  python -u scripts/reset-user-calibration.py --email admin@iadicare.com

This does NOT touch model artifacts; it only resets threshold_low/high used to
turn probabilities into ai_decision buckets.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


# Ensure the app package (./app) is importable when running from ./scripts
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.database import SessionLocal
from app.models import User, UserCalibration


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    p.add_argument("--threshold-low", type=float, default=0.35)
    p.add_argument("--threshold-high", type=float, default=0.65)
    args = p.parse_args()

    if args.threshold_low >= args.threshold_high:
        raise SystemExit("threshold-low must be < threshold-high")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == args.email).first()
        if not user:
            raise SystemExit(f"User not found: {args.email}")

        cal = db.query(UserCalibration).filter(UserCalibration.user_id == user.id).first()
        if not cal:
            print("No calibration row found; nothing to reset.")
            return 0

        cal.threshold_low = float(args.threshold_low)
        cal.threshold_high = float(args.threshold_high)
        cal.updated_at = datetime.utcnow()
        db.commit()

        print(
            f"Reset calibration for {args.email}: "
            f"threshold_low={cal.threshold_low} threshold_high={cal.threshold_high}"
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
