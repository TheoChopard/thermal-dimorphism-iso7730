"""
run_pipeline.py — CLI Orchestrator
=====================================
Usage:
  python run_pipeline.py --simulate           # full simulation (~5s)
  python run_pipeline.py --validate           # CBE validation only (~3min)
  python run_pipeline.py --all                # simulation then validation
  python run_pipeline.py --simulate --quick   # N_GEN=20k for quick tests (~1s)
  python run_pipeline.py --all --quick        # full pipeline in quick mode
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Module files ───────────────────────────────────────────────────────────
SIM_FILE = HERE / "thermal_dimorphism.py"
VAL_FILE = HERE / "Validation.py"

def _check_file(path: Path, name: str) -> bool:
    if not path.exists():
        print(f"  [ERROR] {name} not found: {path}")
        return False
    return True

def _run(script: Path, env: dict = None, label: str = "") -> int:
    """Run a Python script in a subprocess."""
    merged_env = {**os.environ, **(env or {})}
    t0 = time.time()
    print(f"\n{'='*78}")
    print(f"  ▶ {label or script.name}")
    print(f"{'='*78}")
    result = subprocess.run(
        [sys.executable, str(script)],
        env=merged_env
    )
    elapsed = time.time() - t0
    status  = "✓ OK" if result.returncode == 0 else f"✗ ERROR (code {result.returncode})"
    print(f"\n  {status}  ({elapsed:.1f}s)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Thermal Dimorphism Pipeline — CLI Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --simulate          full simulation
  python run_pipeline.py --validate          CBE validation (loads existing JSON)
  python run_pipeline.py --all               simulation + validation in sequence
  python run_pipeline.py --simulate --quick  N_GEN=20,000 for quick tests (~1s)
  python run_pipeline.py --all --quick       full pipeline in quick mode
        """,
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Run thermal_dimorphism.py")
    parser.add_argument("--validate", action="store_true",
                        help="Run Validation.py (requires decompressed_data.csv)")
    parser.add_argument("--all",      action="store_true",
                        help="--simulate then --validate")
    parser.add_argument("--quick",    action="store_true",
                        help="N_GEN=20,000 for quick tests (simulation only)")
    parser.add_argument("--n-boot",   type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    args = parser.parse_args()

    if args.all:
        args.simulate = True
        args.validate = True

    if not args.simulate and not args.validate:
        parser.print_help()
        return 0

    # Environment variables passed to subprocesses
    env_overrides = {}
    if args.quick:
        env_overrides["TD_QUICK_MODE"] = "1"
        env_overrides["TD_N_GEN"]      = "20000"
        print("  [QUICK MODE] N_GEN = 20,000")
    if args.n_boot != 1000:
        env_overrides["TD_N_BOOT"] = str(args.n_boot)

    pipeline_t0 = time.time()
    rc_total    = 0

    if args.simulate:
        if not _check_file(SIM_FILE, "Simulation"):
            return 1
        rc = _run(SIM_FILE, env=env_overrides,
                  label="SIMULATION — thermal_dimorphism.py")
        rc_total += rc

    if args.validate:
        if not _check_file(VAL_FILE, "Validation"):
            return 1
        rc = _run(VAL_FILE, label="VALIDATION — Validation.py")
        rc_total += rc

    elapsed_total = time.time() - pipeline_t0
    print(f"\n{'='*78}")
    status = "✓ Pipeline complete" if rc_total == 0 else "⚠ Pipeline finished with errors"
    print(f"  {status}  ({elapsed_total:.1f}s total)")
    print(f"  Outputs → {HERE / 'output'}")
    print(f"{'='*78}")
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
