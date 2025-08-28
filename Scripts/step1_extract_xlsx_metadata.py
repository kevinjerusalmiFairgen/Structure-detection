#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Tuple

import pandas as pd


def try_parse_numeric(value: Any) -> Tuple[bool, float]:
    if value is None:
        return False, 0.0
    if isinstance(value, (int, float)) and not (isinstance(value, float) and (pd.isna(value))):
        return True, float(value)
    try:
        s = str(value).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return False, 0.0
        return True, float(s)
    except Exception:
        return False, 0.0


def build_question_objects(df: pd.DataFrame) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        # Drop NA for value analysis
        non_null = series.dropna()
        possible_answers: Dict[str, Any] = {}
        if non_null.empty:
            possible_answers = {}
        else:
            # Determine if numeric-like and number of distinct values
            distinct = non_null.unique().tolist()
            # If all numeric and too many distinct, emit min/max
            nums: List[float] = []
            all_numeric = True
            for v in distinct:
                ok, num = try_parse_numeric(v)
                if not ok:
                    all_numeric = False
                    break
                nums.append(num)
            if all_numeric and len(distinct) > 25:
                if nums:
                    vmin = min(nums)
                    vmax = max(nums)
                    # ints if all floats are whole
                    if all(float(x).is_integer() for x in nums):
                        vmin = int(vmin)
                        vmax = int(vmax)
                    possible_answers = {"min": vmin, "max": vmax}
                else:
                    possible_answers = {}
            else:
                # Build mapping of distinct values (stringified) to label (string)
                # Limit to at most 200 keys to avoid huge payloads
                mapping: Dict[str, Any] = {}
                for v in distinct[:200]:
                    key = str(v)
                    mapping[key] = key
                possible_answers = mapping

        questions.append(
            {
                "question_code": str(col),
                "question_text": str(col),
                "possible_answers": possible_answers,
            }
        )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract question metadata from an Excel .xlsx file into JSON (question codes from column names)."
    )
    parser.add_argument("--input", required=True, help="Path to the input .xlsx file")
    parser.add_argument("--output", help="Optional output JSON path. If omitted, prints to stdout.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    parser.add_argument("--sheet", default=0, help="Sheet index or name (default: 0)")
    parser.add_argument("--include-empty", action="store_true", help="Keep columns with no possible answers")

    args = parser.parse_args()

    # Read Excel
    df = pd.read_excel(args.input, sheet_name=args.sheet, engine="openpyxl")
    questions = build_question_objects(df)
    total_before = len(questions)
    if not args.include_empty:
        questions = [q for q in questions if q.get("possible_answers")]
    payload = json.dumps(questions, ensure_ascii=False, indent=args.indent)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()


