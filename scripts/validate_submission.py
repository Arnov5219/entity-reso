#!/usr/bin/env python3
"""Thin CLI wrapper for submission validation."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    validator_path = root / "utils" / "validate_submission.py"
    cmd = [sys.executable, str(validator_path)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
