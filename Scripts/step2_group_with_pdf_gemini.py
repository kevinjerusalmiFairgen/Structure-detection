#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

try:
    from google import genai
    from google.genai import types
    from google.genai.types import Content, Part, UploadFileConfig
except Exception as exc:
    raise SystemExit(
        "google-genai is required. Install with: pip install google-genai"
    ) from exc


def build_prompt() -> str:
    return (
        "You are given: (1) a PDF questionnaire (vision file part) and (2) a COMPACT JSON array of SPSS metadata variables.\n"
        "Goal: Using BOTH sources, reorganize ONLY the provided SPSS metadata into groups and produce a COMBINED questions JSON.\n\n"
        "Rules:\n"
        "- SOURCE OF TRUTH: SPSS metadata. Do NOT add variables or options not present in metadata.\n"
        "- Manipulate metadata only: preserve codes and possible_answers exactly as provided; reorder into groups.\n"
        "- Use the PDF ONLY to decide grouping (multi-select or grid) and recode relationships.\n"
        "- Group types: multi-select or grid. A group must have >=2 members.\n"
        "- Do NOT include range (min/max) variables inside groups. Keep them as standalone items.\n"
        "- Output MUST be JSON array. For grouped items, emit an object: {\"question_code\": group_code, \"question_text\": text, \"question_type\": \"multi-select\"|\"grid\", \"sub_questions\": [{\"question_code\": code, \"possible_answers\": {...}}]}.\n"
        "- For standalone variables, emit: {\"question_code\": code, \"question_text\": text, \"question_type\": \"integer\"|\"single-select\", \"possible_answers\": {...}}.\n"
        "- Optional recode metadata: If and only if a variable (or group) is a true recode (a hidden/computed variable derived by a formula from other variables), add \"recode_from\": [source_codes...]. Otherwise omit this field entirely. Do not guess.\n"
        "- IMPORTANT: Do NOT confound recodes with logics (skip/display dependencies).\n"
        "  * Recodes: hidden/computed variables produced from existing data (e.g., age group derived from AGE).\n"
        "  * Logics: routing/enablement dependencies (e.g., show Q10 only if Q5=\"Yes\"). Logics are NOT recodes and must NOT appear as recode_from.\n"
        "  * Only populate recode_from for genuine derived/computed variables; do not include gating/skip conditions.\n"
        "- STRICT: All emitted question_code values must be a subset of the provided metadata codes. No invented codes.\n"
        "- Return ONLY the final JSON array; no markdown.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group metadata using the questionnaire PDF + SPSS metadata in a single Gemini call."
    )
    parser.add_argument("--pdf", required=True, help="Path to the questionnaire PDF")
    parser.add_argument("--metadata", required=True, help="Path to metadata questions JSON (from step1)")
    parser.add_argument("--output", required=True, help="Path to write the combined grouped JSON")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--fallback", action="store_true", help="If no groups from API, emit heuristic prefix-based groups instead of failing")
    parser.add_argument("--flash", action="store_true", help="Use Gemini 2.5 Flash with higher thinking budget (4096)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        raise SystemExit("Set --api-key or GOOGLE_API_KEY.")

    client = genai.Client(api_key=api_key)

    # Choose model and thinking budget
    model_name = args.model
    thinking_budget = 1024
    temperature = 0.0
    if args.flash:
        model_name = "gemini-2.5-flash"
        thinking_budget = 256
        temperature = 0.1

    # Load metadata and compact
    with open(args.metadata, "r", encoding="utf-8") as f:
        full_meta: List[Dict[str, Any]] = json.load(f)

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
    # Light size cap for faster calls while staying smart
    if len(compact_items) > 1200:
        compact_items = compact_items[:1200]
    compact_json = json.dumps(compact_items, ensure_ascii=False)

    # Upload PDF
    with open(args.pdf, "rb") as f:
        file_obj = client.files.upload(file=f, config=UploadFileConfig(mime_type="application/pdf"))

    # Wait until ACTIVE
    start = time.time()
    name = getattr(file_obj, "name", None)
    while True:
        refreshed = client.files.get(name=name)
        state = getattr(refreshed, "state", None)
        if state in ("ACTIVE", "SUCCEEDED", "READY"):
            file_obj = refreshed
            break
        if time.time() - start > 90:
            raise SystemExit("Timed out waiting for PDF to be ready.")
        time.sleep(1.0)

    # Prepare request
    prompt = build_prompt()
    contents = [
        Content(
            role="user",
            parts=[
                Part.from_uri(file_uri=file_obj.uri, mime_type="application/pdf"),
                Part.from_text(text=prompt),
                Part.from_text(text="SPSS_METADATA_COMPACT_JSON:\n" + compact_json),
            ],
        )
    ]

    generate_cfg = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=False,
        ),
    )
    def call_model(curr_model: str, cfg: "types.GenerateContentConfig") -> str:
        resp = client.models.generate_content(
            model=curr_model,
            contents=contents,
            config=cfg,
        )
        return getattr(resp, "text", "") or "[]"

    text = call_model(model_name, generate_cfg)
    if not isinstance(text, str) or not text.strip():
        # Write raw empty output marker and exit
        try:
            with open(args.output + ".raw.txt", "w", encoding="utf-8") as rf:
                rf.write("(empty response)\n")
        except Exception:
            pass
        print("[error] Model returned empty response.")
        raise SystemExit(2)

    # Extract JSON array safely
    def extract_json_array_str(s: str) -> str:
        start = s.find("[")
        if start == -1:
            raise ValueError("no '[' found")
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        raise ValueError("no matching ']' found")

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("not a list")
    except Exception:
        try:
            snippet = extract_json_array_str(text)
            data = json.loads(snippet)
            if not isinstance(data, list):
                raise ValueError
        except Exception as exc:
            try:
                with open(args.output + ".raw.txt", "w", encoding="utf-8") as rf:
                    rf.write(text)
            except Exception:
                pass
            print(f"[error] Failed to parse JSON array from model response: {exc}")
            raise SystemExit(2)

    # Warn if model returned no groups (or only standalones)
    def has_groups(items: List[Dict[str, Any]]) -> bool:
        for it in items:
            if isinstance(it, dict):
                subs = it.get("sub_questions")
                if isinstance(subs, list) and len(subs) >= 2:
                    return True
        return False

    if not has_groups(data):
        # Retry once with alternate model/settings
        try:
            alt_model = "gemini-2.5-pro" if model_name == "gemini-2.5-flash" else "gemini-2.5-flash"
            alt_cfg = types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024,
                    include_thoughts=False,
                ),
            )
            print(f"[warn] No groups found; retrying with {alt_model}…")
            text2 = call_model(alt_model, alt_cfg)
            if not isinstance(text2, str) or not text2.strip():
                data2 = []
            else:
                try:
                    data2 = json.loads(text2)
                    if not isinstance(data2, list):
                        raise ValueError
                except Exception:
                    try:
                        snippet2 = extract_json_array_str(text2)
                        data2 = json.loads(snippet2)
                        if not isinstance(data2, list):
                            data2 = []
                    except Exception:
                        # persist raw retry output
                        try:
                            with open(args.output + ".retry.raw.txt", "w", encoding="utf-8") as rf:
                                rf.write(text2)
                        except Exception:
                            pass
                        data2 = []
            if has_groups(data2):
                data = data2
            else:
                if getattr(args, "fallback", False):
                    print("[warn] No groups after retry; using heuristic fallback grouping.")
                    # Heuristic fallback grouping by base prefix
                    def compute_base(code: str) -> str:
                        if "_" in code:
                            parts = code.split("_")
                            if len(parts) > 1:
                                return "_".join(parts[:-1])
                        import re
                        m = re.match(r"^(.*?)([A-Za-z]?\d{1,3})$", code)
                        return m.group(1) if m else code
                    base_to_members: Dict[str, List[Dict[str, Any]]] = {}
                    for q in full_meta:
                        c = str(q.get("question_code", ""))
                        if not c:
                            continue
                        b = compute_base(c)
                        base_to_members.setdefault(b, []).append(q)
                    grouped_items: List[Dict[str, Any]] = []
                    for base, members in base_to_members.items():
                        if len(members) >= 2:
                            grouped_items.append({
                                "question_code": f"{base}_GROUP",
                                "question_text": base,
                                "question_type": "multi-select",
                                "sub_questions": [
                                    {"question_code": m.get("question_code"), "possible_answers": m.get("possible_answers", {})}
                                    for m in members
                                ],
                            })
                    member_codes = {m.get("question_code") for members in base_to_members.values() if len(members) >= 2 for m in members}
                    for q in full_meta:
                        c = q.get("question_code")
                        if c in member_codes:
                            continue
                        pa = q.get("possible_answers")
                        qtype = "integer" if isinstance(pa, dict) and set(pa.keys()) == {"min", "max"} else "single-select"
                        grouped_items.append({
                            "question_code": c,
                            "question_text": q.get("question_text"),
                            "question_type": qtype,
                            "possible_answers": pa,
                        })
                    data = grouped_items
                else:
                    print("[error] Grouping produced no groups after retry. Aborting.")
                    raise SystemExit(2)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[error] Retry failed: {exc}")
            raise SystemExit(2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=args.indent)
    print(f"[group] Written grouped questions to: {args.output}")


if __name__ == "__main__":
    main()


