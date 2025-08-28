#!/usr/bin/env python3
import argparse
import json
import os
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
        "- Optional recode metadata: If and only if a variable (or group) is a recode/derived from other variable(s), add \"recode_from\": [source_codes...]. Otherwise omit this field entirely. Do not guess: include it only when clearly indicated by the questionnaire or metadata.\n"
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
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generate_cfg,
    )
    text = getattr(response, "text", "") or "[]"

    # Extract JSON array
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("not a list")
    except Exception:
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            data = json.loads(text[s : e + 1])
        else:
            raise SystemExit("Model did not return a JSON array.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=args.indent)
    print(f"[group] Written grouped questions to: {args.output}")


if __name__ == "__main__":
    main()


