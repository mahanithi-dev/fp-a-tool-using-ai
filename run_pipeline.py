from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpa_variance_tool.pipeline import run_end_to_end_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FP&A variance analysis pipeline.")
    parser.add_argument("--budget", required=True, help="Path to budget CSV/Excel file.")
    parser.add_argument("--actuals", required=True, help="Path to actuals CSV/Excel file.")
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "latest_run"),
        help="Directory for generated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_end_to_end_pipeline(
        budget_path=args.budget,
        actuals_path=args.actuals,
        output_dir=args.output,
    )

    print("Pipeline completed successfully.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
