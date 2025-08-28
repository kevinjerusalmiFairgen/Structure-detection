#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def emit_groups(grouped_items: List[Dict[str, Any]], *, min_columns: int = 2) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    gid = 0
    for item in grouped_items:
        if not isinstance(item, dict):
            continue
        subs = item.get("sub_questions")
        if isinstance(subs, list) and len(subs) >= min_columns:
            columns: List[str] = []
            for sq in subs:
                if isinstance(sq, dict) and isinstance(sq.get("question_code"), str):
                    columns.append(sq["question_code"])
            if len(columns) >= min_columns:
                groups.append({
                    "id": f"group_{gid}",
                    "name": str(item.get("question_text", item.get("question_code", f"group_{gid}"))),
                    "columns": columns,
                })
                gid += 1
    return {"groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit compact groups JSON from step2 grouped questions JSON.")
    parser.add_argument("--input", required=True, help="Path to step2_grouped_questions.json")
    parser.add_argument("--output", required=True, help="Path to write groups JSON (with key 'groups')")
    parser.add_argument("--min-columns", type=int, default=2)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON array")
    out = emit_groups(data, min_columns=args.min_columns)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=args.indent)
    print(f"[groups] Written: {args.output}")


if __name__ == "__main__":
    main()


