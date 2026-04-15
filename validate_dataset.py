from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpa_variance_tool.data_validation import (
    load_dataset_for_validation,
    validate_required_financial_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether a dataset contains required FP&A columns."
    )
    parser.add_argument("--file", required=True, help="Path to CSV or Excel file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset_for_validation(args.file)
    result = validate_required_financial_columns(df)

    print(f"Validation status: {result.status}")
    if result.missing_or_incorrect_columns:
        print("Missing or incorrect columns:")
        for item in result.missing_or_incorrect_columns:
            print(f"- {item}")
    else:
        print("Missing or incorrect columns: None")

    if result.suggested_fixes:
        print("Suggested fixes:")
        for item in result.suggested_fixes:
            print(f"- {item}")
    else:
        print("Suggested fixes: None")


if __name__ == "__main__":
    main()
