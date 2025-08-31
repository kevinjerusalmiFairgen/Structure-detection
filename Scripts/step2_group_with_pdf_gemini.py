#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import shutil
import subprocess
from typing import Any, Dict, List
import re

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
        "INPUTS\n"
        "- PDF questionnaire (vision)\n"
        "- SPSS metadata COMPACT JSON + ALLOWED_CODES (exact allowlist)\n\n"
        "OBJECTIVE\n"
        "- Find group columns that are part of the same question (multi-select) and find true derived variables (recodes). Treat any grid as multi-select. Do NOT emit standalone non-recode variables. When a variable is truly computed, annotate it with recode_from listing exact ALLOWED_CODES sources (never routing/skip logic).\n\n"
        "HARD CONSTRAINTS\n"
        "- SOURCE OF TRUTH: metadata. Preserve codes and possible_answers exactly; do not add/invent/rename/reformat (case/underscores/hyphens/leading zeros).\n"
        "- question_code values must be EXACT matches from ALLOWED_CODES (allowlist). If unsure, omit.\n"
        "- Do NOT change code formatting: do not alter case, add/remove underscores or hyphens, change digits, reorder parts, or normalize/slugify in any way. Use letter-for-letter exact codes only.\n"
        "- Do NOT invent or rewrite question_text: for variables use metadata question_text verbatim; for group headers use the stem/base (from metadata or questionnaire stem).\n"
        "- Group headers are labels, not variables: use '<STEM>_GROUP' (or '<STEM>_GRID'); never use a real code as header.\n"
        "- Do NOT include range (min/max) variables inside groups.\n"
        "- Use the PDF ONLY to decide grouping and true recodes.\n\n"
        "DERIVED VARIABLES (RECODES)\n"
        "- Only for genuine computed variables (indices/totals/bands).\n"
        "- recode_from must be exact sources from ALLOWED_CODES; never group headers.\n"
        "- Do NOT include routing/skip/display logic as recodes.\n\n"
        "GROUPING GUIDANCE\n"
        "- PRIMARY RULE: Group variables that are the same question repeated across columns. Confirm with the questionnaire: shared stem/wording (common lead-in).\n"
        "- Implicit multi-select groups are allowed if variables share grouping signals. Use ALL applicable signals:\n"
        "  • Shared stem/base prefix (e.g., Q1_1, Q1_2, Q1_97).\n"
        "  • Identical or highly similar answer lists (not only pa_type size, but labels if available).\n"
        "  • Consecutive or patterned codes (e.g., Q5_1..Q5_10).\n"
        "  • Clear grid/layout cues from the PDF (columns/rows under a common stem).\n"
        "  • Also ensure compatibility by pa_type from the compact metadata.\n"
        "- Groups may not always be explicitly labeled in the questionnaire — if the structure (stem, identical answers, consecutive numbering, or layout) clearly implies a group, include it.\n\n"
        "- Do NOT group by section headings. Sections may contain several independent repeated-question groups; only group variables that share the same question/stem and answer sets, consistent with the questionnaire’s intent.\n\n"
        "- Aim for completeness: include all matching members from metadata; groups must have ≥2 members.\n"
        "- EXCLUSIVE/NO-ANSWER variants: include only if the exact code exists in ALLOWED_CODES. Accepted forms include numeric suffixes (Q1_97/Q1_98/Q1_99), token forms (Q2_noanswer/Q2_dontknow/Q2_dk/Q2_na/Q2_refused), and token-first variants (noanswer_Q2) if present.\n\n"
        "OUTPUT SCHEMA (JSON array only, no markdown)\n"
        "- Grouped (multi-select only, grids included as multi-select): {\"question_code\": \"<STEM>_GROUP\", \"question_text\": text, \"question_type\": \"multi-select\", \"sub_questions\": [{\"question_code\": code, \"possible_answers\": {...}}]}\n"
        "- Standalone RECODE ONLY: {\"question_code\": code, \"question_text\": text, \"question_type\": \"single-select\"|\"integer\", \"possible_answers\": {...}, \"recode_from\": [source_codes...]}\n\n"
        "EXAMPLE (format only; use only ALLOWED_CODES in your answer)\n"
        "[\n"
        "  {\n"
        "    \"question_code\": \"Q1_GROUP\",\n"
        "    \"question_text\": \"Q1\",\n"
        "    \"question_type\": \"multi-select\",\n"
        "    \"sub_questions\": [\n"
        "      {\"question_code\": \"Q1_1\", \"possible_answers\": {\"1\": \"Yes\", \"0\": \"No\"}},\n"
        "      {\"question_code\": \"Q1_97\", \"possible_answers\": {\"1\": \"Selected\", \"0\": \"Not selected\"}}\n"
        "    ]\n"
        "  },\n"
        "  {\n"
        "    \"question_code\": \"AGE_BAND\",\n"
        "    \"question_text\": \"Age band\",\n"
        "    \"question_type\": \"single-select\",\n"
        "    \"possible_answers\": {\"1\": \"18-24\", \"2\": \"25-34\"},\n"
        "    \"recode_from\": [\"AGE\"]\n"
        "  }\n"
        "]\n\n"
        "CHECK BEFORE RESPONDING\n"
        "- Every question_code ∈ ALLOWED_CODES (except group headers).\n"
        "- Only emit multi-select groups (treat grids as multi-select); each group has ≥2 members; no ranges in groups.\n"
        "- Do NOT emit standalone items unless they are true recodes (must include recode_from).\n"
        "- Exclusive variants included only if present in ALLOWED_CODES.\n"
        "- recode_from sources ∈ ALLOWED_CODES and are not group headers.\n"
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
    compact_json = json.dumps(compact_items, ensure_ascii=False)

    # Auto-switch to Pro if metadata is large (for better context handling)
    try:
        meta_vars = len(full_meta)
        compact_bytes = len(compact_json.encode("utf-8"))
        too_big = (meta_vars >= 1500) or (compact_bytes >= 1_500_000)
    except Exception:
        too_big = False
    if ("flash" in str(locals().get("model_name", ""))) and too_big:
        print(f"[info] Metadata is large (vars={meta_vars}, size={compact_bytes} bytes). Switching to gemini-2.5-pro.")
        model_name = "gemini-2.5-pro"
        thinking_budget = 1024
        temperature = 0.0

    # Increase thinking budget when input is complex (more variables / larger payload)
    try:
        if meta_vars >= 3000 or compact_bytes >= 3_000_000:
            thinking_budget = max(thinking_budget, 3072)
        elif meta_vars >= 2000 or compact_bytes >= 2_000_000:
            thinking_budget = max(thinking_budget, 2048)
        elif meta_vars >= 1200 or compact_bytes >= 1_200_000:
            thinking_budget = max(thinking_budget, 1536)
        print(f"[info] Thinking budget set to: {thinking_budget}")
    except Exception:
        pass

    # Upload questionnaire (PDF only)
    pdf_path = args.pdf
    if str(pdf_path).lower().endswith(".docx"):
        print("[error] DOCX is no longer supported. Please provide a PDF.")
        raise SystemExit(2)
    with open(pdf_path, "rb") as f:
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
    # Also pass explicit ALLOWED_CODES for exact matching
    allowed_codes = [str(q.get("question_code")) for q in full_meta if q.get("question_code")]
    allowed_codes_json = json.dumps(sorted(allowed_codes), ensure_ascii=False)

    contents = [
        Content(
            role="user",
            parts=[
                Part.from_uri(file_uri=file_obj.uri, mime_type="application/pdf"),
                Part.from_text(text=prompt),
                Part.from_text(text="SPSS_METADATA_COMPACT_JSON:\n" + compact_json),
                Part.from_text(text="ALLOWED_CODES (exact match allowlist):\n" + allowed_codes_json),
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

    print(f"[info] Using model: {model_name}")
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

    # Extract JSON array safely (and helpers for robust recovery)
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

    def extract_code_fence(s: str) -> str:
        tick = "```"
        start = s.find(tick)
        if start == -1:
            return ""
        # Skip optional language tag
        lang_end = s.find("\n", start + len(tick))
        if lang_end == -1:
            return ""
        end = s.find(tick, lang_end + 1)
        if end == -1:
            return ""
        return s[lang_end + 1:end]

    def extract_json_object_str(s: str) -> str:
        start = s.find("{")
        if start == -1:
            raise ValueError("no '{' found")
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        raise ValueError("no matching '}' found")

    def balance_brackets_fragment(fragment: str) -> str:
        # Best-effort: count brackets/braces and append closing ones
        open_sq = fragment.count('[')
        close_sq = fragment.count(']')
        open_br = fragment.count('{')
        close_br = fragment.count('}')
        balanced = fragment
        # Close braces first, then brackets
        balanced += '}' * max(0, open_br - close_br)
        balanced += ']' * max(0, open_sq - balanced.count(']'))
        return balanced

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("not a list")
    except Exception:
        # Try code-fenced content first
        cf = extract_code_fence(text)
        if cf:
            try:
                data = json.loads(cf)
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    raise ValueError
            except Exception:
                pass
        if 'data' not in locals():
            try:
                snippet = extract_json_array_str(text)
                data = json.loads(snippet)
                if not isinstance(data, list):
                    raise ValueError
            except Exception:
                # Try object top-level → wrap in list
                try:
                    obj = extract_json_object_str(text)
                    obj_json = json.loads(obj)
                    data = [obj_json]
                except Exception:
                    # Try balancing missing closers on array fragment
                    try:
                        start_idx = text.find('[')
                        if start_idx != -1:
                            frag = balance_brackets_fragment(text[start_idx:])
                            data = json.loads(frag)
                            if not isinstance(data, list):
                                raise ValueError
                        else:
                            raise ValueError("no '[' found")
                    except Exception as exc2:
                        try:
                            with open(args.output + ".raw.txt", "w", encoding="utf-8") as rf:
                                rf.write(text)
                        except Exception:
                            pass
                        print(f"[error] Failed to parse JSON array from model response: {exc2}")
                        raise SystemExit(2)

    # Warn if model returned no groups (or only standalones)
    def has_groups(items: List[Dict[str, Any]]) -> bool:
        for it in items:
            if isinstance(it, dict):
                subs = it.get("sub_questions")
                if isinstance(subs, list) and len(subs) >= 2:
                    return True
        return False

    def count_groups(items: List[Dict[str, Any]]) -> int:
        n = 0
        for it in items:
            if isinstance(it, dict):
                subs = it.get("sub_questions")
                if isinstance(subs, list) and len(subs) >= 2:
                    n += 1
        return n

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

    print(f"[debug] Groups returned by model (pre-validation): {count_groups(data)}")
    with open(args.output, "w", encoding="utf-8") as f:
        # Post-process: STRICT validation only. Do not auto-add or augment.
        # Build metadata code set
        allowed_codes: set[str] = set()
        for q in full_meta:
            c = str(q.get("question_code", ""))
            if c:
                allowed_codes.add(c)

        unknown_in_groups: Dict[str, List[str]] = {}
        unknown_standalone: List[str] = []
        invalid_recode_groups: Dict[str, List[str]] = {}
        invalid_recode_vars: Dict[str, List[str]] = {}

        # Pre-collect group header codes to forbid using them as recode sources
        group_header_codes: set[str] = set()
        for it in data:
            if isinstance(it, dict) and isinstance(it.get("sub_questions"), list):
                gh = str(it.get("question_code", ""))
                if gh:
                    group_header_codes.add(gh)

        for it in data:
            if not isinstance(it, dict):
                continue
            subs = it.get("sub_questions")
            if isinstance(subs, list):
                grp_code = str(it.get("question_code", "GROUP"))
                bad: List[str] = []
                for sq in subs:
                    if not isinstance(sq, dict):
                        continue
                    scode = str(sq.get("question_code", ""))
                    if scode and scode not in allowed_codes:
                        bad.append(scode)
                if bad:
                    unknown_in_groups[grp_code] = bad
                # Validate group-level recodes
                rec = it.get("recode_from")
                if isinstance(rec, list):
                    bad_rec: List[str] = []
                    for src in rec:
                        s = str(src)
                        if not s or (s not in allowed_codes) or (s in group_header_codes):
                            bad_rec.append(s)
                    if bad_rec:
                        invalid_recode_groups[grp_code] = bad_rec
                # Validate sub-question recodes
                for sq in subs:
                    if not isinstance(sq, dict):
                        continue
                    rec2 = sq.get("recode_from")
                    if isinstance(rec2, list):
                        bad_rec2: List[str] = []
                        for src in rec2:
                            s2 = str(src)
                            if not s2 or (s2 not in allowed_codes) or (s2 in group_header_codes):
                                bad_rec2.append(s2)
                        if bad_rec2:
                            target = str(sq.get("question_code", "")) or "(unknown)"
                            invalid_recode_vars[target] = bad_rec2
            else:
                code = str(it.get("question_code", ""))
                if code and code not in allowed_codes:
                    unknown_standalone.append(code)
                rec = it.get("recode_from")
                if isinstance(rec, list):
                    bad_rec: List[str] = []
                    for src in rec:
                        s = str(src)
                        if not s or (s not in allowed_codes) or (s in group_header_codes):
                            bad_rec.append(s)
                    if bad_rec:
                        target = code or "(unknown)"
                        invalid_recode_vars[target] = bad_rec

        # If any unknowns or invalid recodes exist, attempt ONE targeted correction round via the model
        if unknown_in_groups or unknown_standalone or invalid_recode_groups or invalid_recode_vars:
            try:
                print("[info] Attempting targeted correction for unknown/invalid items using ALLOWED_CODES…")
                # Build minimal correction payload (only offending codes), and request a mapping
                bad_group_members: List[Dict[str, Any]] = []
                for g, codes in unknown_in_groups.items():
                    bad_group_members.append({"group_code": g, "bad_codes": sorted(list(set(codes)))})
                bad_recode_sources: List[str] = []
                for codes in invalid_recode_groups.values():
                    bad_recode_sources.extend(codes)
                for codes in invalid_recode_vars.values():
                    bad_recode_sources.extend(codes)
                bad_recode_sources = sorted(list(set([str(x) for x in bad_recode_sources if str(x)])))
                correction_request = {
                    "bad_standalone_codes": sorted(list(set(unknown_standalone))),
                    "bad_group_members": bad_group_members,
                    "bad_recode_sources": bad_recode_sources,
                }
                correction_instructions = (
                    "CORRECTION TASK: You are given ALLOWED_CODES (exact allowlist) and a list of offending codes.\n"
                    "Goal: Produce a small JSON with how to fix ONLY these offending items.\n"
                    "Output strict JSON object with keys: \n"
                    "- code_replacements: array of {bad: string, good: string} where good ∈ ALLOWED_CODES.\n"
                    "- drop_codes: array of strings for items to drop (no matching code).\n"
                    "- source_replacements: array of {bad: string, good: string} to fix recode_from sources (good ∈ ALLOWED_CODES).\n"
                    "Rules: Do not invent. Do not change formatting of ALLOWED_CODES. Only map to exact codes from ALLOWED_CODES or drop.\n"
                )
                allowed_codes_json2 = json.dumps(sorted(list(allowed_codes)), ensure_ascii=False)
                request_json = json.dumps(correction_request, ensure_ascii=False)
                correction_contents = [
                    Content(
                        role="user",
                        parts=[
                            Part.from_text(text=correction_instructions),
                            Part.from_text(text="ALLOWED_CODES:\n" + allowed_codes_json2),
                            Part.from_text(text="OFFENDING_ITEMS:\n" + request_json),
                        ],
                    )
                ]
                corrected_text = client.models.generate_content(
                    model=model_name,
                    contents=correction_contents,
                    config=generate_cfg,
                ).text or "{}"
                # Parse mapping
                try:
                    mapping = json.loads(corrected_text)
                except Exception:
                    try:
                        snippet = extract_json_object_str(corrected_text)
                        mapping = json.loads(snippet)
                    except Exception:
                        mapping = {}
                code_replacements: Dict[str, str] = {}
                drop_codes: set[str] = set()
                source_replacements: Dict[str, str] = {}
                if isinstance(mapping, dict):
                    for entry in mapping.get("code_replacements", []) or []:
                        bad = str(entry.get("bad", ""))
                        good = str(entry.get("good", ""))
                        if bad and good and good in allowed_codes:
                            code_replacements[bad] = good
                    for d in mapping.get("drop_codes", []) or []:
                        if isinstance(d, str) and d:
                            drop_codes.add(d)
                    for entry in mapping.get("source_replacements", []) or []:
                        bads = str(entry.get("bad", ""))
                        goods = str(entry.get("good", ""))
                        if bads and goods and goods in allowed_codes:
                            source_replacements[bads] = goods
                # Apply mapping to data (only touches offending codes)
                def replace_code(code_value: str) -> str:
                    return code_replacements.get(code_value, code_value)
                new_items: List[Dict[str, Any]] = []
                for it in data:
                    if not isinstance(it, dict):
                        continue
                    subs = it.get("sub_questions")
                    if isinstance(subs, list):
                        new_subs: List[Dict[str, Any]] = []
                        for sq in subs:
                            if not isinstance(sq, dict):
                                continue
                            sc = str(sq.get("question_code", ""))
                            if sc in drop_codes:
                                continue
                            new_code = replace_code(sc)
                            new_sq = dict(sq)
                            new_sq["question_code"] = new_code
                            # Fix sub-question recode sources
                            rec2 = new_sq.get("recode_from")
                            if isinstance(rec2, list):
                                fixed = []
                                for src in rec2:
                                    s2 = str(src)
                                    if s2 in drop_codes:
                                        continue
                                    s2n = source_replacements.get(s2, s2)
                                    fixed.append(s2n)
                                new_sq["recode_from"] = fixed
                            new_subs.append(new_sq)
                        new_it = dict(it)
                        new_it["sub_questions"] = new_subs
                        # Fix group-level recode sources
                        rec = new_it.get("recode_from")
                        if isinstance(rec, list):
                            fixedg = []
                            for src in rec:
                                s = str(src)
                                if s in drop_codes:
                                    continue
                                sn = source_replacements.get(s, s)
                                fixedg.append(sn)
                            new_it["recode_from"] = fixedg
                        new_items.append(new_it)
                    else:
                        codev = str(it.get("question_code", ""))
                        if codev in drop_codes:
                            continue
                        new_codev = replace_code(codev)
                        new_it = dict(it)
                        new_it["question_code"] = new_codev
                        # Fix standalone recode sources
                        rec = new_it.get("recode_from")
                        if isinstance(rec, list):
                            fixeds = []
                            for src in rec:
                                s = str(src)
                                if s in drop_codes:
                                    continue
                                sn = source_replacements.get(s, s)
                                fixeds.append(sn)
                            new_it["recode_from"] = fixeds
                        new_items.append(new_it)
                data = new_items
                print("[info] Targeted correction completed.")
            except Exception as _exc:
                print(f"[warn] Correction round failed: {_exc}")

        # Recompute unknowns after correction (or no-op)
        unknown_in_groups = {}
        unknown_standalone = []
        invalid_recode_groups = {}
        invalid_recode_vars = {}
        for it in data:
            if not isinstance(it, dict):
                continue
            subs = it.get("sub_questions")
            if isinstance(subs, list):
                grp_code = str(it.get("question_code", "GROUP"))
                bad: List[str] = []
                for sq in subs:
                    if not isinstance(sq, dict):
                        continue
                    scode = str(sq.get("question_code", ""))
                    if scode and scode not in allowed_codes:
                        bad.append(scode)
                if bad:
                    unknown_in_groups[grp_code] = bad
                # Validate group-level recodes
                rec = it.get("recode_from")
                if isinstance(rec, list):
                    bad_rec: List[str] = []
                    for src in rec:
                        s = str(src)
                        if not s or (s not in allowed_codes) or (s in group_header_codes):
                            bad_rec.append(s)
                    if bad_rec:
                        invalid_recode_groups[grp_code] = bad_rec
                # Validate sub-question recodes
                for sq in subs:
                    if not isinstance(sq, dict):
                        continue
                    rec2 = sq.get("recode_from")
                    if isinstance(rec2, list):
                        bad_rec2: List[str] = []
                        for src in rec2:
                            s2 = str(src)
                            if not s2 or (s2 not in allowed_codes) or (s2 in group_header_codes):
                                bad_rec2.append(s2)
                        if bad_rec2:
                            target = str(sq.get("question_code", "")) or "(unknown)"
                            invalid_recode_vars[target] = bad_rec2
            else:
                code = str(it.get("question_code", ""))
                if code and code not in allowed_codes:
                    unknown_standalone.append(code)
                rec = it.get("recode_from")
                if isinstance(rec, list):
                    bad_rec: List[str] = []
                    for src in rec:
                        s = str(src)
                        if not s or (s not in allowed_codes) or (s in group_header_codes):
                            bad_rec.append(s)
                    if bad_rec:
                        target = code or "(unknown)"
                        invalid_recode_vars[target] = bad_rec

        if unknown_in_groups or invalid_recode_groups or invalid_recode_vars:
            for g, codes in unknown_in_groups.items():
                print(f"[error] Non-metadata sub-questions in group '{g}': {codes}")
            for g, codes in invalid_recode_groups.items():
                print(f"[error] Invalid recode_from for group header '{g}': {codes}")
            for v, codes in invalid_recode_vars.items():
                print(f"[error] Invalid recode_from for variable '{v}': {codes}")
            print("[fail] Output contains invalid variables or recode sources in groups. Fix the prompt/grouping and retry.")
            raise SystemExit(3)
        # Unknown standalone items are dropped with a warning instead of failing
        if unknown_standalone:
            print(f"[warn] Dropping standalone items not in metadata: {sorted(set(unknown_standalone))}")

        # If validation passed, filter out standalone non-recode items (warn, do not fail)
        non_recode_standalone: List[str] = []
        non_metadata_standalone: List[str] = []
        filtered_data: List[Dict[str, Any]] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            subs = it.get("sub_questions")
            if isinstance(subs, list):
                filtered_data.append(it)
            else:
                code = str(it.get("question_code", ""))
                rec = it.get("recode_from")
                # Drop standalone items not in metadata entirely
                if code and code not in allowed_codes:
                    non_metadata_standalone.append(code)
                    continue
                if isinstance(rec, list) and len(rec) > 0:
                    filtered_data.append(it)
                else:
                    if code:
                        non_recode_standalone.append(code)

        if non_recode_standalone:
            print(f"[warn] Dropping standalone non-recode items: {sorted(set(non_recode_standalone))}")
        if non_metadata_standalone:
            print(f"[warn] Dropped standalone non-metadata items: {sorted(set(non_metadata_standalone))}")

        print(f"[debug] Groups after validation/filtering: {count_groups(filtered_data)}")

        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"[group] Written grouped questions to: {args.output}")


if __name__ == "__main__":
    main()


