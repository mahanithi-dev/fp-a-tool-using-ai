from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpa_variance_tool.data_validation import (
    load_raw_dataset_headers,
    map_uploaded_columns_to_fpa_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map uploaded dataset columns to standard FP&A fields."
    )
    parser.add_argument("--file", required=True, help="Path to CSV or Excel file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    column_names = load_raw_dataset_headers(args.file)
    mapping = map_uploaded_columns_to_fpa_fields(column_names)
    print(json.dumps(mapping, indent=2))


if __name__ == "__main__":
    main()
