#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple


def normalize_possible_answers(pa: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(pa, dict) and set(pa.keys()) == {"min", "max"}:
        return ("range", {"min": pa.get("min"), "max": pa.get("max")})
    if isinstance(pa, dict):
        try:
            # Stable stringify with sorted keys to compare equality across variables
            norm = {str(k): pa[k] for k in sorted(pa.keys(), key=lambda x: str(x))}
        except Exception:
            norm = {str(k): pa.get(k) for k in pa.keys()}
        return (f"labels:{len(norm)}", norm)
    return ("labels:0", {})


def guess_base_code(code: str) -> str:
    # Heuristic: cut trailing segments like _a, _b, _1, r1, r2, .a, -a
    if not isinstance(code, str) or not code:
        return code or ""
    c = code
    for sep in ["_", ".", "-"]:
        if sep in c:
            parts = c.split(sep)
            if len(parts[-1]) <= 3:
                return sep.join(parts[:-1])
    # Remove trailing 1-3 alnum chars if preceding is alnum
    import re
    m = re.match(r"^(.*?)([a-zA-Z]?[0-9]{1,2}|[a-zA-Z]{1,2})$", c)
    if m and len(c) - len(m.group(1)) <= 3:
        return m.group(1)
    return c


def fallback_grouping(meta: List[Dict[str, Any]], pdf: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Build lookups
    code_to_pdf: Dict[str, Dict[str, Any]] = {str(q.get("question_code")): q for q in pdf if q.get("question_code")}

    # Cluster by (base_code, normalized_possible_answers)
    clusters: Dict[Tuple[str, str, int], List[str]] = {}
    for q in meta:
        code = q.get("question_code")
        if not code:
            continue
        pa_type, norm_pa = normalize_possible_answers(q.get("possible_answers"))
        if pa_type == "range":
            continue
        base = guess_base_code(code)
        key = (base, pa_type, len(norm_pa))
        clusters.setdefault(key, []).append(code)

    groups: List[Dict[str, Any]] = []
    for (base, pa_type, n), members in clusters.items():
        if len(members) < 2:
            continue
        # Check PDF types suggest multi/grid for at least some members
        pdf_types = {str(code_to_pdf.get(m, {}).get("question_type") or "") for m in members}
        if {"matrix_multi", "multi_select"} & pdf_types or len(members) >= 3:
            gtype = "multi-select" if "multi_select" in pdf_types else ("grid" if "matrix_multi" in pdf_types else "multi-select")
            groups.append({
                "group_code": base or (members[0] + "_GROUP"),
                "group_text": base or (members[0] + " group"),
                "columns": sorted(members),
                "group_type": gtype,
            })

    return groups


def assemble_output(groups: List[Dict[str, Any]], meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    member_to_group: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        cols = g.get("columns")
        if not isinstance(cols, list) or len(cols) < 2:
            continue
        for c in cols:
            member_to_group[str(c)] = {"code": g.get("group_code") or "GROUP", "text": g.get("group_text") or "GROUP", "group_type": g.get("group_type") or "multi-select"}

    final_items: List[Dict[str, Any]] = []
    emitted_groups: Dict[str, Dict[str, Any]] = {}
    for q in meta:
        code = str(q.get("question_code"))
        grp = member_to_group.get(code)
        if grp:
            gcode = grp["code"]
            if gcode not in emitted_groups:
                emitted_groups[gcode] = {
                    "question_code": gcode,
                    "question_text": grp["text"],
                    "question_type": ("grid" if grp.get("group_type") == "grid" else "multi-select"),
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
    return final_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect multi-select/grid groups using BOTH SPSS metadata (step1) and questionnaire (step2).")
    parser.add_argument("--metadata", required=True, help="Path to metadata questions JSON from step1")
    parser.add_argument("--pdf-questions", required=True, help="Path to questionnaire JSON from step2")
    parser.add_argument("--output", required=True, help="Path to write combined grouped JSON")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--api-key", dest="api_key", help="Optional Gemini API key for enhanced grouping (falls back to heuristic)")
    parser.add_argument("--model", default="gemini-2.5-pro")

    args = parser.parse_args()

    print(f"[step3b] Loading metadata: {args.metadata}")
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"[step3b] Loading questionnaire: {args.pdf_questions}")
    with open(args.pdf_questions, "r", encoding="utf-8") as f:
        pdf = json.load(f)

    # Build compact items for LLM prompt
    compact: List[Dict[str, Any]] = []
    code_to_meta = {str(q.get("question_code")): q for q in meta if q.get("question_code")}
    code_to_pdf = {str(q.get("question_code")): q for q in pdf if q.get("question_code")}
    all_codes = sorted(set(code_to_meta.keys()) | set(code_to_pdf.keys()))
    for code in all_codes:
        m = code_to_meta.get(code, {})
        p = code_to_pdf.get(code, {})
        pa_type, norm_pa = normalize_possible_answers(m.get("possible_answers"))
        compact.append({
            "question_code": code,
            "question_text": p.get("question_text") or m.get("question_text") or code,
            "pa_type": pa_type,
            "pdf_question_type": p.get("question_type") or "",
            "labels_count": (len(norm_pa) if isinstance(norm_pa, dict) else 0),
        })

    groups: List[Dict[str, Any]] = []

    # Try Gemini-enhanced grouping if available
    used_llm = False
    try:
        from google import genai  # type: ignore
        from google.genai.types import Content, Part  # type: ignore
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or ""
        if api_key:
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are given a COMPACT JSON array of survey variables derived from both SPSS metadata and a PDF questionnaire.\n"
                "Each object has: question_code, question_text, pa_type where pa_type ∈ {range, labels:N}, pdf_question_type (may be empty), labels_count.\n\n"
                "Task: Identify ALL question groups (multi-select and grids).\n"
                "Return ONLY a JSON array of objects: {\"group_code\": string, \"group_text\": string, \"columns\": [codes...], \"group_type\": \"multi-select\"|\"grid\"}.\n\n"
                "Guidelines:\n"
                "- A group must contain at least 2 members.\n"
                "- Do NOT include variables with pa_type == range.\n"
                "- Prefer grouping variables that share a base code (e.g., A27, C14) and similar text.\n"
                "- If pdf_question_type suggests multi_select or matrix_multi, that strongly indicates grouping.\n"
                "- Be conservative: only group when semantics align.\n"
                "- Keep codes exactly as provided and avoid hallucinations.\n"
            )
            print("[step3b] Calling Gemini for enhanced grouping…")
            resp = client.models.generate_content(
                model=args.model,
                contents=[Content(role="user", parts=[Part.from_text(text=prompt), Part.from_text(text=json.dumps(compact, ensure_ascii=False))])],
                config={"temperature": 0.0},
            )
            txt = getattr(resp, "text", "") or "[]"
            s = txt.find("[")
            e = txt.rfind("]")
            if s != -1 and e != -1 and e > s:
                groups = json.loads(txt[s : e + 1])
                used_llm = True
    except Exception as exc:
        print(f"[step3b] Gemini not used ({exc}); falling back to deterministic heuristic.")

    if not groups:
        print("[step3b] Using deterministic fallback grouping…")
        groups = fallback_grouping(meta, pdf)

    print(f"[step3b] Groups detected: {len(groups)} (llm={used_llm})")
    final_items = assemble_output(groups, meta)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=args.indent)

    print(json.dumps({
        "total_items": len(final_items),
        "groups": len([x for x in final_items if isinstance(x, dict) and x.get("sub_questions")]),
        "used_llm": used_llm,
        "output": args.output,
    }))
    print(f"[step3b] Written grouped questions to: {args.output}")


if __name__ == "__main__":
    main()


