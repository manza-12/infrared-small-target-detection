"""
ACM Infrared Small Target Detection — Multi-person Training Runner

Usage (one person just runs ONE of these):
  python run.py --person 1    # FPN+BiLocal, FPN+AsymBi
  python run.py --person 2    # FPN+BiGlobal, UNet+BiLocal
  python run.py --person 3    # UNet+AsymBi   (best model, slower)
  python run.py --person 4    # UNet+BiGlobal

Before running, do ONE-TIME setup:
  python setup_data.py
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

# ---- Config ---------------------------------------------------------------
PYTHON = sys.executable  # use whichever Python runs this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'acm', 'train.py')
SETUP_SCRIPT = os.path.join(BASE_DIR, 'setup_data.py')
SIRST_ROOT = os.path.join(BASE_DIR, 'data', 'sirst')

# Six model configs (backbone, fuse_mode)
ALL_CONFIGS = [
    ('FPN',  'BiLocal'),
    ('FPN',  'AsymBi'),
    ('FPN',  'BiGlobal'),
    ('UNet', 'BiLocal'),
    ('UNet', 'AsymBi'),
    ('UNet', 'BiGlobal'),
]

# Person assignments (1-indexed)
PERSON_ASSIGN = {
    1: [0, 1],           # FPN+BiLocal, FPN+AsymBi
    2: [2, 3],           # FPN+BiGlobal, UNet+BiLocal
    3: [4],              # UNet+AsymBi (best performer)
    4: [5],              # UNet+BiGlobal
}

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def run_cmd(cmd, desc):
    """Run a command, stream output, raise on failure."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] {desc} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"  DONE: {desc}")


def run_cmd_env(cmd, desc, env):
    """Run a command with custom environment variables."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] {desc} failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"  DONE: {desc}")


def main():
    parser = argparse.ArgumentParser(description='ACM Multi-person Training Runner')
    parser.add_argument('--person', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Person ID (1-4)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just print what would be run, don\'t execute')
    args = parser.parse_args()

    configs = PERSON_ASSIGN[args.person]
    names = [f"{ALL_CONFIGS[i][0]}+{ALL_CONFIGS[i][1]}" for i in configs]

    print("=" * 60)
    print(f"  Person {args.person} — {len(configs)} config(s):")
    for n in names:
        print(f"    - {n}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    # --- one-time data check ---
    if not os.path.isdir(SIRST_ROOT):
        print("\n[STEP 0] Running setup_data.py (first-time data preparation)...")
        if not args.dry_run:
            run_cmd(f'"{PYTHON}" "{SETUP_SCRIPT}"', 'setup_data.py')
    else:
        print(f"\n[SKIP] Data already prepared at {SIRST_ROOT}")

    # --- train each config ---
    total = len(configs)
    for idx, cfg_i in enumerate(configs):
        backbone, fuse = ALL_CONFIGS[cfg_i]
        name = f"{backbone}+{fuse}"

        cmd = (
            f'"{PYTHON}" "{TRAIN_SCRIPT}" '
            f'--backbone-mode {backbone} '
            f'--fuse-mode {fuse} '
            f'--batch-size {args.batch_size} '
            f'--epochs {args.epochs} '
            f'--learning_rate {args.lr}'
        )
        env = os.environ.copy()
        env['SIRST_DATA_ROOT'] = SIRST_ROOT

        print(f"\n[{idx+1}/{total}] {name}")
        if args.dry_run:
            print(f"  [DRY-RUN] SIRST_DATA_ROOT={SIRST_ROOT} {cmd}")
        else:
            run_cmd_env(cmd, name, env)

    print("\n" + "=" * 60)
    print(f"  Person {args.person} — ALL DONE at {datetime.now()}")
    print("=" * 60)

    # Print results
    result_dir = os.path.join(BASE_DIR, 'acm', 'result')
    if os.path.isdir(result_dir):
        print("\n--- Results ---")
        for d in sorted(os.listdir(result_dir)):
            ckpt_dir = os.path.join(result_dir, d, 'checkpoint')
            if os.path.isdir(ckpt_dir):
                files = sorted(os.listdir(ckpt_dir))
                if files:
                    print(f"  {d}  →  {files[-1]}")


if __name__ == '__main__':
    main()
