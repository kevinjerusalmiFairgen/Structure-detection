#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List

try:
    from google import genai
    from google.genai.types import Content, Part
except Exception as exc:
    raise SystemExit(
        "google-genai is required. Install with: pip install google-genai"
    ) from exc

def build_prompt() -> str:
    return (
        "You are given a COMPACT JSON array of survey metadata variables (appended after this instruction).\n"
        "Each element has: question_code, question_text, pa_type where pa_type ∈ {range, labels:N}.\n\n"
        "Goal: Identify ALL question groups (multi-select and grids).\n"
        "Return ONLY a JSON array of objects: {\"group_code\": string, \"group_text\": string, \"columns\": [codes...], \"group_type\": \"multi-select\"|\"grid\"}. The group_type field is REQUIRED.\n\n"
        "Guidelines:\n"
        "- A group must contain at least 2 members.\n"
        "- Do NOT include variables with pa_type == range.\n"
        "- Prefer grouping variables sharing a base code (letter+digits, e.g., D01, C14, A27) and similar question text.\n"
        "- Multi-select groups: options for a select-all-that-apply question; use group_type=\"multi-select\".\n"
        "- Grid groups: SAME PREFIX (same letter+digits base) with near-identical question_texts (minor wording/punctuation differences only), representing rows/items of the SAME question; use group_type=\"grid\".\n"
        "  Examples of grids: C08r1..C08r6 (spend by category), D31r1..D31r4; multi-select: A27r1..A27r6, D36r*.\n"
        "- Be conservative: only group when semantics align (options/items of the same question). If uncertain, leave as single-select.\n"
        "- Keep codes exactly as provided.\n"
        "- The response must be a pure JSON array (no markdown, no comments).\n"
    )


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    s = text.find("[")
    e = text.rfind("]")
    if s != -1 and e != -1 and e > s:
        snippet = text[s : e + 1]
        return json.loads(snippet)
    # fallback: try to parse as-is
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise ValueError("Model did not return a JSON array")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-prompt LLM grouping: upload metadata JSON and ask Gemini to group multi-selects and keep others."
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata questions JSON")
    parser.add_argument("--output", required=True, help="Path to write the final combined JSON")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--indent", type=int, default=2)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or "***REMOVED***"
    if not api_key:
        raise SystemExit("Set --api-key or GOOGLE_API_KEY.")

    client = genai.Client(api_key=api_key)

    # Load metadata and build a compact representation to reduce payload size
    print(f"[step3] Loading metadata: {args.metadata}")
    with open(args.metadata, "r", encoding="utf-8") as f:
        full_meta: List[Dict[str, Any]] = json.load(f)

    print("[step3] Building compact items for LLM prompt...")
    compact_items: List[Dict[str, Any]] = []
    for q in full_meta:
        code = q.get("question_code")
        text = q.get("question_text")
        pa = q.get("possible_answers")
        if isinstance(pa, dict) and set(pa.keys()) == {"min", "max"}:
            pa_type = "range"
        elif isinstance(pa, dict):
            pa_type = f"labels:{len(pa)}"
        else:
            pa_type = "labels:0"
        compact_items.append({"question_code": code, "question_text": text, "pa_type": pa_type})

    compact_json = json.dumps(compact_items, ensure_ascii=False)
    prompt = build_prompt()

    # Try up to 3 times on transient server errors
    last_err: Any = None
    print("[step3] Calling Gemini for grouping (single prompt)...")
    for _ in range(3):
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=[Content(role="user", parts=[Part.from_text(text=prompt), Part.from_text(text=compact_json)])],
                config={"temperature": 0.0},
            )
            text = getattr(response, "text", "") or "[]"
            groups = extract_json_array(text)
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    else:
        raise SystemExit(f"Gemini call failed after retries: {last_err}")

    # Assemble final items from groups + original metadata
    print("[step3] Assembling grouped metadata output...")
    member_to_group: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        cols = g.get("columns")
        if not isinstance(cols, list) or len(cols) < 2:
            continue
        gcode = g.get("group_code") or "MULTI_GROUP"
        gtext = g.get("group_text") or gcode
        gtype = g.get("group_type") or "multi-select"
        for c in cols:
            member_to_group[c] = {"code": gcode, "text": gtext}

    final_items: List[Dict[str, Any]] = []
    emitted_groups: Dict[str, Dict[str, Any]] = {}
    for q in full_meta:
        code = q.get("question_code")
        grp = member_to_group.get(code)
        if grp:
            gcode = grp["code"]
            if gcode not in emitted_groups:
                emitted_groups[gcode] = {
                    "question_code": gcode,
                    "question_text": grp["text"],
                    "question_type": ("grid" if any((isinstance(g, dict) and g.get("group_code") == gcode and g.get("group_type") == "grid") for g in groups) else "multi-select"),
                    "sub_questions": [],
                }
                final_items.append(emitted_groups[gcode])
            emitted_groups[gcode]["sub_questions"].append({
                "question_code": code,
                "possible_answers": q.get("possible_answers", {}),
            })
        else:
            pa = q.get("possible_answers")
            qtype = "integer" if isinstance(pa, dict) and set(pa.keys()) == {"min", "max"} else "single-select"
            final_items.append({
                "question_code": code,
                "question_text": q.get("question_text"),
                "question_type": qtype,
                "possible_answers": pa,
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=args.indent)

    print(json.dumps({
        "total_items": len(final_items),
        "groups": len(emitted_groups),
    }))
    print(f"[step3] Written combined questions to: {args.output}")


if __name__ == "__main__":
    main()


