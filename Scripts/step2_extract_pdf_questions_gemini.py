#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List

try:
    from google import genai
    from google.genai.types import Content, Part, UploadFileConfig
except Exception as exc:
    raise SystemExit(
        "google-genai is required. Install with: pip install google-genai"
    ) from exc


def wait_until_active(client: "genai.Client", file_obj: Any, timeout_seconds: int = 120) -> Any:
    start = time.time()
    name = getattr(file_obj, "name", None)
    while True:
        refreshed = client.files.get(name=name)
        state = getattr(refreshed, "state", None)
        if state in ("ACTIVE", "SUCCEEDED", "READY"):
            return refreshed
        if state in ("FAILED", "ERROR"):
            raise RuntimeError(f"File processing failed with state={state}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for file to become ACTIVE")
        time.sleep(1.0)


def build_prompt() -> str:
    return (
        "You are given a full survey questionnaire as a PDF. "
        "Extract every question and its possible answers as structured JSON.\n\n"
        "Output MUST be a JSON array. Each element MUST be an object with: \n"
        "- question_code: short stable code (use the code in the questionnaire if present; otherwise infer like Q1, Q2, or Q5_a for sub-items).\n"
        "- question_text: the exact full question text.\n"
        "- question_type: one of [number, single_select, multi_select, text, date, time, matrix_single, matrix_multi, ranking, scale].\n"
        "- possible_answers: \n"
        "  - If enumerated options exist, map option codes/values to labels, e.g., {\"1\": \"Very satisfied\"}.\n"
        "  - If numeric range only, provide {\"min\": number, \"max\": number}.\n"
        "  - If free-text/no predefined answers, use {}.\n\n"
        "Rules:\n"
        "- Represent grid/matrix questions as separate items per row or sub-item, using suffixed codes (e.g., Q10_a, Q10_b).\n"
        "- Preserve any explicit codes/values shown in the questionnaire; if only labels are shown, infer numerical codes starting at 1.\n"
        "- Return ONLY the JSON. No markdown, no comments, no trailing text."
    )


def parse_json_strict(text: str) -> List[Dict[str, Any]]:
    # Try strict parse first
    try:
        return json.loads(text)
    except Exception:
        # Heuristic: extract first JSON array substring
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use Gemini 2.5 Pro to extract survey questions from a PDF without local parsing."
    )
    parser.add_argument("--pdf", required=True, help="Path to the questionnaire PDF")
    parser.add_argument("--output", help="Where to write the JSON output. If omitted, prints to stdout.")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name (default: gemini-2.5-pro)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--print", dest="print_json", action="store_true", help="Print JSON to stdout even if --output is provided.")
    parser.add_argument("--api-key", dest="api_key", help="Optional Gemini API key (otherwise uses GOOGLE_API_KEY env var)")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")

    args = parser.parse_args()

    # Default to user's Gemini key if not provided via flag or env
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or "***REMOVED***"
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY environment variable to your Gemini API key.")

    client = genai.Client(api_key=api_key)

    # Upload PDF directly to Gemini
    print(f"[step2] Uploading PDF to Gemini: {args.pdf}")
    with open(args.pdf, "rb") as f:
        file_obj = client.files.upload(
            file=f,
            config=UploadFileConfig(mime_type="application/pdf"),
        )

    file_obj = wait_until_active(client, file_obj)

    user_prompt = build_prompt()

    print("[step2] Requesting extraction from Gemini...")
    response = client.models.generate_content(
        model=args.model,
        contents=[
            Content(
                role="user",
                parts=[
                    Part.from_uri(file_uri=file_obj.uri, mime_type="application/pdf"),
                    Part.from_text(text=user_prompt),
                ],
            )
        ],
        config={"temperature": args.temperature},
    )

    text = getattr(response, "text", None) or ""
    try:
        data = parse_json_strict(text)
    except Exception as exc:
        # If parsing fails, print the raw text to help debugging
        raise SystemExit(f"Failed to parse JSON from model response: {exc}\nRaw response:\n{text}")

    payload = json.dumps(data, ensure_ascii=False, indent=args.indent)

    if args.output:
        print(f"[step2] Writing extracted questionnaire JSON to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
        print("[step2] Done")

    if (not args.output) or args.print_json:
        print(payload)


if __name__ == "__main__":
    main()


