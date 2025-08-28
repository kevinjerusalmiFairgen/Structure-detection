#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Tuple

try:
    import pyreadstat
except ImportError as exc:
    raise SystemExit(
        "pyreadstat is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


def coerce_label_key_to_string(key: Any) -> str:
    if key is None:
        return ""
    if isinstance(key, bytes):
        try:
            return key.decode("utf-8", errors="replace")
        except Exception:
            return str(key)
    if isinstance(key, float):
        if key.is_integer():
            return str(int(key))
    return str(key)


def try_parse_numeric(value: Any) -> Tuple[bool, float]:
    if isinstance(value, (int, float)):
        return True, float(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return False, 0.0
    if isinstance(value, str):
        try:
            return True, float(value.strip())
        except Exception:
            return False, 0.0
    return False, 0.0


def build_question_objects(meta: "pyreadstat.metadata_container") -> List[Dict[str, Any]]:
    column_names: List[str] = list(getattr(meta, "column_names", []) or [])
    column_labels: List[str] = list(getattr(meta, "column_labels", []) or [])

    # Map variable name -> label (question text)
    name_to_label: Dict[str, str] = {}
    if column_labels and len(column_labels) == len(column_names):
        for name, label in zip(column_names, column_labels):
            name_to_label[name] = label if (label is not None and label != "") else name
    else:
        for name in column_names:
            name_to_label[name] = name

    # Value labels can be provided either per-variable or via labelsets
    variable_value_labels: Dict[str, Dict[Any, str]] = getattr(meta, "variable_value_labels", {}) or {}
    value_labels_catalog: Dict[str, Dict[Any, str]] = getattr(meta, "value_labels", {}) or {}
    variable_to_labelset: Dict[str, str] = getattr(meta, "variable_to_labelset", {}) or {}

    questions: List[Dict[str, Any]] = []
    for var_name in column_names:
        # Resolve possible answers
        possible_answers_raw: Dict[Any, str] = {}

        if var_name in variable_value_labels and variable_value_labels[var_name]:
            possible_answers_raw = variable_value_labels[var_name]
        elif var_name in variable_to_labelset:
            labelset_name = variable_to_labelset.get(var_name)
            if labelset_name and labelset_name in value_labels_catalog:
                possible_answers_raw = value_labels_catalog[labelset_name]

        # If all keys are numeric and there are many (>25), compress to min/max
        possible_answers: Dict[str, Any] = {}
        if possible_answers_raw:
            numeric_vals: List[float] = []
            all_numeric = True
            for k in possible_answers_raw.keys():
                ok, num = try_parse_numeric(k)
                if not ok:
                    all_numeric = False
                    break
                numeric_vals.append(num)

            if all_numeric and len(numeric_vals) > 25:
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
                # Use ints if all are whole numbers
                if all(float(v).is_integer() for v in numeric_vals):
                    min_val = int(min_val)
                    max_val = int(max_val)
                possible_answers = {"min": min_val, "max": max_val}
            else:
                # Fallback: full mapping, stringify keys
                for k, v in possible_answers_raw.items():
                    possible_answers[coerce_label_key_to_string(k)] = v if v is not None else ""

        questions.append(
            {
                "question_code": var_name,
                "question_text": name_to_label.get(var_name, var_name),
                "possible_answers": possible_answers,
            }
        )

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract question metadata from an SPSS .sav file into JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input .sav file",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON output. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )
    parser.add_argument(
        "--print",
        dest="print_json",
        action="store_true",
        help="Also print JSON to stdout even if --output is provided.",
    )
    parser.add_argument(
        "--include-empty",
        dest="include_empty",
        action="store_true",
        help="Include questions with no possible answers. By default, such questions are removed.",
    )

    args = parser.parse_args()

    print(f"[step1] Reading SPSS metadata from: {args.input}")
    # Read only metadata to keep it fast and memory efficient
    _, meta = pyreadstat.read_sav(args.input, metadataonly=True)
    print("[step1] Building question objects from metadata...")
    questions = build_question_objects(meta)
    total_before = len(questions)
    if not args.include_empty:
        questions = [q for q in questions if q.get("possible_answers")]  # drop empties
        dropped = total_before - len(questions)
        print(f"[step1] Filtered empty questions: {dropped} dropped, {len(questions)} kept")
    else:
        print(f"[step1] Keeping all questions: {len(questions)}")

    payload = json.dumps(questions, ensure_ascii=False, indent=args.indent)

    if args.output:
        print(f"[step1] Writing JSON to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
        print("[step1] Done")

    if (not args.output) or args.print_json:
        print(payload)


if __name__ == "__main__":
    main()


