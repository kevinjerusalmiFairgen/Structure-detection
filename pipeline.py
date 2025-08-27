#!/usr/bin/env python3
import argparse
import os
import json
import shlex
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def run(cmd: str) -> int:
    print(f"$ {cmd}")
    return subprocess.call(cmd, shell=True)


def run_async(cmd: str) -> subprocess.Popen:
    print(f"$ {cmd}")
    return subprocess.Popen(cmd, shell=True)


# Hard-coded default Gemini API key fallback (used if --api-key and env are not set)
DEFAULT_GOOGLE_API_KEY = "***REMOVED***"


def gemini_semantic_merge_no_pdf(
    out_merged: str,
    *,
    gemini_csv: str | None = None,
    openai_csv: str | None = None,
    claude_csv: str | None = None,
    metadata_json: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> int:
    """Use Gemini to semantically deduplicate and merge priors WITHOUT using the PDF.

    - Inputs are only prior CSV rows (Gemini/OpenAI/Claude) and optional metadata names
    - Prefer Gemini rows on conflicts; remove semantically equivalent duplicates
    - Do not invent new rules; only output from provided rows
    """
    print("[merge] gemini-semantic: start (no-pdf)")
    try:
        from google import genai  # type: ignore
        from google.genai.types import Content, Part  # type: ignore
    except Exception:
        print("[merge] google-genai not installed; falling back to simple merge.")
        return 2

    header = ["Question", "Constraint", "Input Questions", "Relationship", "Comment"]

    def read_rows(path: str | None) -> list[dict[str, str]]:
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                import csv as _csv
                r = _csv.DictReader(f)
                return [
                    {
                        "Question": row.get("Question", ""),
                        "Constraint": row.get("Constraint", ""),
                        "Input Questions": row.get("Input Questions", ""),
                        "Relationship": row.get("Relationship", ""),
                        "Comment": row.get("Comment", ""),
                    }
                    for row in r
                ]
        except Exception:
            return []

    gem_rows = read_rows(gemini_csv)
    cla_rows = read_rows(claude_csv)
    oai_rows = read_rows(openai_csv)
    if not any([gem_rows, cla_rows, oai_rows]):
        print("[merge] No prior CSVs found; nothing to merge.")
        return 1

    variable_names: list[str] = []
    if metadata_json and os.path.exists(metadata_json):
        try:
            with open(metadata_json, "r", encoding="utf-8") as f:
                md = json.load(f)
            variable_names = [q.get("question_code", "") for q in md if q.get("question_code")]
        except Exception:
            variable_names = []

    client = genai.Client(api_key=(api_key or os.environ.get("GOOGLE_API_KEY") or DEFAULT_GOOGLE_API_KEY))

    instruction = (
        "You are an expert survey methodologist. Semantically de-duplicate and merge prior rows. "
        "Do NOT invent new rules. Treat two rows as duplicates if they describe the SAME logic relationship: "
        "same target question(s), same constraint type, same gating inputs and effect, even if phrased differently or with reordered lists. "
        "Prefer rows from the GEMINI list when duplicates are detected. Preserve the fields from the chosen row. "
        "Return ONLY JSON array of objects with keys: Question, Constraint, Input Questions, Relationship, Comment."
    )

    parts: list[Part] = []
    if variable_names:
        parts.append(Part.from_text(text="ALLOWED_VARIABLE_NAMES_JSON:\n" + json.dumps(variable_names, ensure_ascii=False)))
    parts.append(Part.from_text(text="GEMINI_PRIOR_ROWS_JSON:\n" + json.dumps(gem_rows, ensure_ascii=False)))
    if cla_rows:
        parts.append(Part.from_text(text="CLAUDE_PRIOR_ROWS_JSON:\n" + json.dumps(cla_rows, ensure_ascii=False)))
    if oai_rows:
        parts.append(Part.from_text(text="OPENAI_PRIOR_ROWS_JSON:\n" + json.dumps(oai_rows, ensure_ascii=False)))
    parts.append(Part.from_text(text="MERGE_TASK_INSTRUCTION:\n" + instruction))

    print("[merge] Calling Gemini to semantic-dedupe…")
    try:
        resp = client.models.generate_content(
            model=(model or "gemini-2.5-pro"),
            contents=[Content(role="user", parts=parts)],
            config={"temperature": 0.0},
        )
        raw_text = getattr(resp, "text", "") or ""
        s = raw_text.find("[")
        e = raw_text.rfind("]")
        data = []
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw_text[s:e+1])
            except Exception:
                data = []
        if not isinstance(data, list):
            try:
                data = json.loads(raw_text)
            except Exception:
                data = []
        merged_rows: list[dict[str, str]] = []
        for obj in (data or []):
            if not isinstance(obj, dict):
                continue
            merged_rows.append({
                "Question": str(obj.get("Question", "")),
                "Constraint": str(obj.get("Constraint", "")),
                "Input Questions": str(obj.get("Input Questions", "")),
                "Relationship": str(obj.get("Relationship", "")),
                "Comment": str(obj.get("Comment", "")),
            })
    except Exception as exc:
        print(f"[merge] Gemini call failed: {exc}; falling back to simple merge.")
        return 2

    # Restrict to union of originals (no invention) and prefer Gemini comments on identical rules
    filtered: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    all_candidates = gem_rows + cla_rows + oai_rows
    for r in merged_rows:
        k4 = (r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""))
        # ensure exists in any candidate by 4-tuple
        exists = any(
            cand.get("Question", "") == r.get("Question", "") and
            cand.get("Constraint", "") == r.get("Constraint", "") and
            cand.get("Input Questions", "") == r.get("Input Questions", "") and
            cand.get("Relationship", "") == r.get("Relationship", "")
            for cand in all_candidates
        )
        if not exists or k4 in seen_keys:
            continue
        # prefer Gemini comment if available
        for g in gem_rows:
            if (
                g.get("Question", "") == r.get("Question", "") and
                g.get("Constraint", "") == r.get("Constraint", "") and
                g.get("Input Questions", "") == r.get("Input Questions", "") and
                g.get("Relationship", "") == r.get("Relationship", "")
            ):
                r["Comment"] = g.get("Comment", r.get("Comment", ""))
                break
        seen_keys.add(k4)
        filtered.append(r)

    with open(out_merged, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in filtered:
            w.writerow(r)
    print(json.dumps({
        "merge": {
            "gemini_rows": len(gem_rows),
            "claude_rows": len(cla_rows),
            "openai_rows": len(oai_rows),
            "merged_rows": len(filtered),
            "out_merged": out_merged,
            "strategy": "gemini-semantic-no-pdf"
        }
    }))
    print("[merge] gemini-semantic: done")
    return 0

def gemini_intelligent_merge(
    pdf_path: str,
    metadata_json: str,
    out_merged: str,
    *,
    gemini_csv: str | None = None,
    openai_csv: str | None = None,
    claude_csv: str | None = None,
    api_key: str | None = None,
    pdf_json: str | None = None,
    model: str | None = None,
) -> int:
    """Use Gemini to reconcile and merge multiple prior CSVs without inventing new rules.

    - Reads PDF (vision) and optional structured JSON
    - Reads metadata to provide variable names
    - Reads candidate rows from provided CSVs
    - In conflicts, prioritizes Gemini rows
    - Writes a merged CSV with the standard header
    """
    print("[merge] gemini-intelligent: start", flush=True)
    try:
        from google import genai  # type: ignore
        from google.genai.types import Content, Part, UploadFileConfig  # type: ignore
    except Exception as exc:
        print("[merge] google-genai not installed; falling back to simple merge.")
        return 2

    header = ["Question", "Constraint", "Input Questions", "Relationship", "Comment"]

    def read_rows(path: str | None) -> list[dict[str, str]]:
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                import csv as _csv
                r = _csv.DictReader(f)
                return [
                    {
                        "Question": row.get("Question", ""),
                        "Constraint": row.get("Constraint", ""),
                        "Input Questions": row.get("Input Questions", ""),
                        "Relationship": row.get("Relationship", ""),
                        "Comment": row.get("Comment", ""),
                    }
                    for row in r
                ]
        except Exception:
            return []

    gem_rows = read_rows(gemini_csv)
    cla_rows = read_rows(claude_csv)
    oai_rows = read_rows(openai_csv)

    # If nothing to merge, bail
    if not any([gem_rows, cla_rows, oai_rows]):
        print("[merge] No prior CSVs found; nothing to merge.")
        return 1

    # Load metadata to provide ALLOWED_VARIABLE_NAMES_JSON
    variable_names: list[str] = []
    try:
        with open(metadata_json, "r", encoding="utf-8") as f:
            md = json.load(f)
        variable_names = [q.get("question_code", "") for q in md if q.get("question_code")]
    except Exception:
        variable_names = []

    # Optional structured questionnaire JSON (compact)
    pdf_json_part: list["Part"] = []
    if pdf_json and os.path.exists(pdf_json):
        try:
            with open(pdf_json, "r", encoding="utf-8") as jf:
                pdf_questions = json.load(jf)
            compact = []
            for item in pdf_questions[:1200]:
                compact.append({
                    "question_code": item.get("question_code"),
                    "question_text": item.get("question_text"),
                    "possible_answers": item.get("possible_answers", {}),
                })
            pdf_json_text = json.dumps(compact, ensure_ascii=False)
            if len(pdf_json_text) > 160000:
                pdf_json_text = pdf_json_text[:160000]
            pdf_json_part = [Part.from_text(text="QUESTIONNAIRE_STRUCTURED_JSON:\n" + pdf_json_text)]
        except Exception:
            pdf_json_part = []

    # Extract full PDF text as additional context
    pdf_text_block: str = ""
    try:
        import fitz  # type: ignore
        with fitz.open(pdf_path) as doc:
            texts: list[str] = []
            for page in doc:
                texts.append(page.get_text())
            pdf_text = "\n".join(texts)
            pdf_text_block = "QUESTIONNAIRE_PDF_TEXT:\n" + (pdf_text[:180000] if len(pdf_text) > 180000 else pdf_text)
    except Exception:
        pdf_text_block = ""

    client = genai.Client(api_key=(api_key or os.environ.get("GOOGLE_API_KEY") or DEFAULT_GOOGLE_API_KEY))

    # Upload PDF for vision context
    pdf_part: list["Part"] = []
    try:
        print(f"[merge] Uploading PDF to Gemini: {pdf_path}")
        with open(pdf_path, "rb") as pf:
            file_obj = client.files.upload(file=pf, config=UploadFileConfig(mime_type="application/pdf"))
        # quick wait loop until ACTIVE
        start = time.time()
        name = getattr(file_obj, "name", None)
        while True:
            fo = client.files.get(name=name)
            state = getattr(fo, "state", None)
            if state in ("ACTIVE", "SUCCEEDED", "READY"):
                pdf_part = [Part.from_uri(file_uri=getattr(fo, "uri", None), mime_type="application/pdf")]
                break
            if time.time() - start > 120:
                print("[merge] PDF upload wait timed out; continuing without file part.")
                pdf_part = []
                break
            time.sleep(1.0)
    except Exception:
        pdf_part = []

    instruction = (
        "You are an expert survey methodologist. Merge the provided candidate prior rows into a single, clean prior. "
        "Do NOT invent any new rules. Only select rows from the provided candidate lists. "
        "When duplicate/conflicting rows describe the same logic, choose the best one using the PDF and your judgment; "
        "if uncertain, prefer the Gemini prior row. De-duplicate exact or near-duplicate entries. Preserve fields verbatim. "
        "Return ONLY a JSON array of objects with keys: Question, Constraint, Input Questions, Relationship, Comment."
    )

    # Build contents
    parts: list["Part"] = []
    parts += pdf_part
    if variable_names:
        parts.append(Part.from_text(text="ALLOWED_VARIABLE_NAMES_JSON:\n" + json.dumps(variable_names, ensure_ascii=False)))
    parts += (pdf_json_part or [])
    if pdf_text_block:
        parts.append(Part.from_text(text=pdf_text_block))
    # Provide the three candidate lists as JSON
    parts.append(Part.from_text(text="GEMINI_PRIOR_ROWS_JSON:\n" + json.dumps(gem_rows, ensure_ascii=False)))
    if cla_rows:
        parts.append(Part.from_text(text="CLAUDE_PRIOR_ROWS_JSON:\n" + json.dumps(cla_rows, ensure_ascii=False)))
    if oai_rows:
        parts.append(Part.from_text(text="OPENAI_PRIOR_ROWS_JSON:\n" + json.dumps(oai_rows, ensure_ascii=False)))
    parts.append(Part.from_text(text="MERGE_TASK_INSTRUCTION:\n" + instruction))

    print("[merge] Calling Gemini to reconcile rows…")
    try:
        resp = client.models.generate_content(
            model=(model or "gemini-2.5-pro"),
            contents=[Content(role="user", parts=parts)],
            config={"temperature": 0.0},
        )
        raw_text = getattr(resp, "text", "") or ""
        s = raw_text.find("[")
        e = raw_text.rfind("]")
        merged_rows: list[dict[str, str]] = []
        data = []
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw_text[s:e+1])
            except Exception:
                data = []
        if not isinstance(data, list):
            try:
                data = json.loads(raw_text)
            except Exception:
                data = []
        for obj in (data or []):
            if not isinstance(obj, dict):
                continue
            merged_rows.append({
                "Question": str(obj.get("Question", "")),
                "Constraint": str(obj.get("Constraint", "")),
                "Input Questions": str(obj.get("Input Questions", "")),
                "Relationship": str(obj.get("Relationship", "")),
                "Comment": str(obj.get("Comment", "")),
            })
    except Exception as exc:
        print(f"[merge] Gemini call failed: {exc}; falling back to simple merge.")
        return 2

    # Optional sanity: restrict to union of originals (no invention)
    union_keys = set()
    for r in (gem_rows + cla_rows + oai_rows):
        union_keys.add((r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""), r.get("Comment", "")))
    filtered: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for r in merged_rows:
        key4 = (r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""))
        # Allow comment differences while keeping rule identity by 4-tuple; prefer Gemini comment if conflict by later pass
        if key4 in seen_keys:
            continue
        # Ensure the 4-tuple exists in at least one candidate; if not, drop (prevents invention)
        exists = False
        for cand in (gem_rows + cla_rows + oai_rows):
            if (
                cand.get("Question", "") == r.get("Question", "") and
                cand.get("Constraint", "") == r.get("Constraint", "") and
                cand.get("Input Questions", "") == r.get("Input Questions", "") and
                cand.get("Relationship", "") == r.get("Relationship", "")
            ):
                exists = True
                # Prefer comment from Gemini candidate if available
                if gem_rows:
                    for g in gem_rows:
                        if (
                            g.get("Question", "") == r.get("Question", "") and
                            g.get("Constraint", "") == r.get("Constraint", "") and
                            g.get("Input Questions", "") == r.get("Input Questions", "") and
                            g.get("Relationship", "") == r.get("Relationship", "")
                        ):
                            r["Comment"] = g.get("Comment", r.get("Comment", ""))
                            break
                break
        if not exists:
            continue
        seen_keys.add(key4)
        filtered.append(r)

    # If model returned nothing usable, fall back to simple concat-dedupe preferring Gemini order
    if not filtered:
        print("[merge] Gemini returned no rows; falling back to simple merge.")
        seen_simple: set[tuple[str, str, str, str]] = set()
        filtered = []
        for r in (gem_rows + cla_rows + oai_rows):
            k = (r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""))
            if k in seen_simple:
                continue
            seen_simple.add(k)
            filtered.append(r)

    with open(out_merged, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in filtered:
            w.writerow(r)
    print(json.dumps({
        "merge": {
            "gemini_rows": len(gem_rows),
            "claude_rows": len(cla_rows),
            "openai_rows": len(oai_rows),
            "merged_rows": len(filtered),
            "out_merged": out_merged,
            "strategy": "gemini-intelligent"
        }
    }))
    print("[merge] gemini-intelligent: done")
    return 0

def step1_extract_metadata(sav_path: str, output_json: str, include_empty: bool = False, indent: int = 2) -> int:
    script = os.path.join(ROOT, "scripts", "step1_extract_spss_metadata.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--input", shlex.quote(sav_path),
        "--output", shlex.quote(output_json),
        "--indent", str(indent),
    ]
    if include_empty:
        args.append("--include-empty")
    return run(" ".join(args))


def step2_extract_llm_pdf(pdf_path: str, output_json: str, indent: int = 2, api_key: str | None = None) -> int:
    script = os.path.join(ROOT, "scripts", "step2_extract_pdf_questions_gemini.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--pdf", shlex.quote(pdf_path),
        "--output", shlex.quote(output_json),
        "--indent", str(indent),
    ]
    if api_key:
        args += ["--api-key", shlex.quote(api_key)]
    return run(" ".join(args))


def step3_group_metadata(metadata_json: str, output_json: str, indent: int = 2, api_key: str | None = None) -> int:
    script = os.path.join(ROOT, "scripts", "step3_group_metadata_gemini.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--metadata", shlex.quote(metadata_json),
        "--output", shlex.quote(output_json),
        "--indent", str(indent),
    ]
    if api_key:
        args += ["--api-key", shlex.quote(api_key)]
    return run(" ".join(args))


def step4_detect_logics(pdf_path: str, metadata_json: str, output_csv: str, api_key: str | None = None, pdf_json: str | None = None) -> int:
    script = os.path.join(ROOT, "scripts", "step4_detect_logics.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--provider", "gemini",
        "--pdf", shlex.quote(pdf_path),
        "--metadata", shlex.quote(metadata_json),
        "--output", shlex.quote(output_csv),
    ]
    if pdf_json:
        args += ["--pdf-json", shlex.quote(pdf_json)]
    if api_key:
        args += ["--api-key", shlex.quote(api_key)]
    return run(" ".join(args))


def step5_enrich_prior(
    metadata_json: str,
    combined_json: str,
    prior_csv: str,
    output_csv: str,
    api_key: str | None = None,
    pdf_json: str | None = None,
    model: str | None = None,
    temperature: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    verbose: bool = False,
    pdf_path: str | None = None,
    full_context: bool = False,
    pdf_text_limit: str | None = None,
    premerge_out: str | None = None,
) -> int:
    raise SystemExit("step5 has been removed from the pipeline.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified pipeline (step1..step5|all)")
    sub = parser.add_subparsers(dest="command", required=True)

    # step1
    p1 = sub.add_parser("step1", help="Extract SPSS metadata to JSON")
    p1.add_argument("--sav", required=True, help="Path to .sav file")
    p1.add_argument("--out", required=True, help="Output JSON path")
    p1.add_argument("--indent", type=int, default=2)
    p1.add_argument("--include-empty", action="store_true")

    # step2
    p2 = sub.add_parser("step2", help="Extract questionnaire from PDF with Gemini")
    p2.add_argument("--pdf", required=True, help="Path to PDF questionnaire")
    p2.add_argument("--out", required=True, help="Output JSON path")
    p2.add_argument("--indent", type=int, default=2)
    p2.add_argument("--api-key")

    # step3
    p3 = sub.add_parser("step3", help="Group metadata into multi-selects using Gemini")
    p3.add_argument("--metadata", required=True, help="Path to metadata questions JSON (from step1)")
    p3.add_argument("--out", required=True, help="Output JSON path")
    p3.add_argument("--indent", type=int, default=2)
    p3.add_argument("--api-key")

    # step4
    p4 = sub.add_parser("step4", help="Detect logic rules (per-model) in parallel and merge")
    p4.add_argument("--pdf", required=True, help="Path to PDF questionnaire")
    p4.add_argument("--metadata", required=True, help="Path to metadata questions JSON (from step1)")
    p4.add_argument("--out-gemini", help="Output CSV for Gemini prior (default in outdir)")
    p4.add_argument("--out-claude", help="Output CSV for Claude prior (default in outdir if Vertex configured)")
    p4.add_argument("--out-merged", help="Output CSV for merged prior (default in outdir)")
    p4.add_argument("--api-key")
    p4.add_argument("--pdf-json", help="Optional path to pdf.questions.llm.json for structured context")
    p4.add_argument("--opus-model", default="claude-opus-4-1", help="Anthropic model ID (default: claude-opus-4-1)")
    p4.add_argument("--opus-project", help="GCP project for Vertex Anthropic (defaults from GCP_PROJECT/GOOGLE_CLOUD_PROJECT)")
    p4.add_argument("--opus-location", default="us-east5", help="GCP region for Anthropic (default: us-east5)")
    # OpenAI
    p4.add_argument("--out-openai", help="Output CSV for OpenAI prior (default in outdir if OpenAI configured)")
    p4.add_argument("--openai-model", default="o4-mini", help="OpenAI model ID (default: o4-mini)")
    p4.add_argument("--openai-api-key", help="OpenAI API key (or OPENAI_API_KEY env)")
    # Gemini-based merge control
    p4.add_argument("--gemini-merge", action="store_true", help="Use Gemini to reconcile and merge priors (no new rules; prefer Gemini)")

    # step5 removed

    # all
    pall = sub.add_parser("all", help="Run all steps (1→5)")
    pall.add_argument("--sav", required=True, help="Path to .sav file")
    pall.add_argument("--pdf", required=True, help="Path to PDF questionnaire")
    default_out = os.path.join(ROOT, "output")
    pall.add_argument("--outdir", required=False, default=default_out, help=f"Directory to write outputs (default: {default_out})")
    pall.add_argument("--indent", type=int, default=2)
    pall.add_argument("--include-empty", action="store_true")
    pall.add_argument("--api-key")
    pall.add_argument("--openai-api-key", help="Optional OpenAI key for step5 (defaults to --api-key or OPENAI_API_KEY)")
    pall.add_argument("--gpt-model", default="claude-opus-4-1-20250805", help="Model for step5 (default: claude-opus-4-1-20250805)")

    args = parser.parse_args()

    if args.command == "step1":
        print("[step] step1: extract metadata (.sav → metadata.questions.json)")
        t0 = time.time()
        rc = step1_extract_metadata(args.sav, args.out, include_empty=args.include_empty, indent=args.indent)
        elapsed = time.time()-t0
        print(f"[time] step1: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        return rc

    if args.command == "step2":
        print("[step] step2: extract questionnaire (PDF → pdf.questions.llm.json)")
        t0 = time.time()
        rc = step2_extract_llm_pdf(args.pdf, args.out, indent=args.indent, api_key=args.api_key)
        elapsed = time.time()-t0
        print(f"[time] step2: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        return rc

    if args.command == "step3":
        print("[step] step3: group metadata into multi-selects (metadata → combined.questions.json)")
        t0 = time.time()
        rc = step3_group_metadata(args.metadata, args.out, indent=args.indent, api_key=args.api_key)
        elapsed = time.time()-t0
        print(f"[time] step3: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        return rc

    if args.command == "step4":
        print("[step] step4: detect logic rules (per-model) in parallel and merge")
        def _dirname_safe(p: object) -> str:
            return os.path.dirname(p) if isinstance(p, str) and p else ""
        outdir = (
            _dirname_safe(getattr(args, "out_gemini", None))
            or _dirname_safe(getattr(args, "out_merged", None))
            or _dirname_safe(getattr(args, "out_claude", None))
            or _dirname_safe(getattr(args, "out_openai", None))
            or os.path.join(ROOT, "output")
        )
        os.makedirs(outdir, exist_ok=True)
        out_gemini = getattr(args, "out_gemini", None) or os.path.join(outdir, "step4_prior_rules.gemini.csv")
        # Disable automatic Claude runs unless --opus-project explicitly provided
        opus_project = getattr(args, "opus_project", None)
        out_claude = getattr(args, "out_claude", None) if opus_project else None
        out_merged = getattr(args, "out_merged", None) or os.path.join(outdir, "step4_prior_rules.merged.csv")
        openai_key = getattr(args, "openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
        # Always run OpenAI; step4 script has its own default key fallback
        out_openai = getattr(args, "out_openai", None) or os.path.join(outdir, "step4_prior_rules.openai.csv")

        gem_script = os.path.join(ROOT, "scripts", "step4_detect_logics.py")
        procs = []
        t0 = time.time()
        gem_parts = [
            sys.executable,
            shlex.quote(gem_script),
            "--provider", "gemini",
            "--pdf", shlex.quote(args.pdf),
            "--metadata", shlex.quote(args.metadata),
            "--output", shlex.quote(out_gemini),
        ]
        if getattr(args, "pdf_json", None):
            gem_parts += ["--pdf-json", shlex.quote(args.pdf_json)]
        if getattr(args, "api_key", None):
            gem_parts += ["--api-key", shlex.quote(args.api_key)]
        procs.append(("gemini", run_async(" ".join(gem_parts))))

        if out_claude and opus_project:
            print("[step] step4-claude: starting Anthropic (Vertex) in parallel…")
            ant_parts = [
                sys.executable,
                shlex.quote(gem_script),
                "--provider", "anthropic",
                "--metadata", shlex.quote(args.metadata),
                "--output", shlex.quote(out_claude),
                "--model", shlex.quote(getattr(args, "opus_model", "claude-opus-4-1")),
                "--vertex-project", shlex.quote(opus_project),
                "--vertex-location", shlex.quote(getattr(args, "opus_location", "us-east5")),
            ]
            if getattr(args, "pdf_json", None):
                ant_parts += ["--pdf-json", shlex.quote(args.pdf_json)]
            ant_parts += ["--pdf", shlex.quote(args.pdf)]
            procs.append(("claude", run_async(" ".join(ant_parts))))

        if out_openai:
            print("[step] step4-openai: starting OpenAI in parallel…")
            oa_parts = [
                sys.executable,
                shlex.quote(gem_script),
                "--provider", "openai",
                "--pdf", shlex.quote(args.pdf),
                "--metadata", shlex.quote(args.metadata),
                "--output", shlex.quote(out_openai),
                "--model", shlex.quote(getattr(args, "openai_model", "gpt-4o-mini")),
            ]
            if openai_key:
                oa_parts += ["--openai-api-key", shlex.quote(openai_key)]
            if getattr(args, "pdf_json", None):
                oa_parts += ["--pdf-json", shlex.quote(args.pdf_json)]
            procs.append(("openai", run_async(" ".join(oa_parts))))

        rc_overall = 0
        for name, p in procs:
            code = p.wait()
            print(f"[step] step4-{name}: exit {code}")
            if code != 0:
                rc_overall = code

        # Merge
        def read_rows(path: str) -> list[dict[str, str]]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    import csv as _csv
                    r = _csv.DictReader(f)
                    return [row for row in r]
            except Exception:
                return []

        def write_rows(path: str, rows: list[dict[str, str]]) -> None:
            with open(path, "w", encoding="utf-8", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=["Question","Constraint","Input Questions","Relationship","Comment"])
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        g_rows = read_rows(out_gemini)
        c_rows = read_rows(out_claude) if out_claude else []
        o_rows = read_rows(out_openai) if out_openai else []
        if getattr(args, "gemini_merge", False):
            print("[step] step4: performing Gemini semantic merge (no-pdf)…")
            rc_merge = gemini_semantic_merge_no_pdf(
                out_merged,
                gemini_csv=out_gemini,
                openai_csv=out_openai,
                claude_csv=out_claude,
                metadata_json=getattr(args, "metadata", None),
                api_key=getattr(args, "api_key", None),
                model=None,
            )
            if rc_merge != 0:
                print("[step] step4: Gemini semantic merge failed; falling back to simple priority merge.")
                seen = set()
                merged = []
                for r in (g_rows + c_rows + o_rows):
                    key = (r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append({
                        "Question": r.get("Question", ""),
                        "Constraint": r.get("Constraint", ""),
                        "Input Questions": r.get("Input Questions", ""),
                        "Relationship": r.get("Relationship", ""),
                        "Comment": r.get("Comment", ""),
                    })
                if merged:
                    write_rows(out_merged, merged)
                    print(json.dumps({"gemini_rows": len(g_rows), "claude_rows": len(c_rows), "openai_rows": len(o_rows), "merged_rows": len(merged), "out_merged": out_merged, "strategy": "priority-prefer-gemini"}))
        else:
            seen = set()
            merged = []
            for r in (g_rows + c_rows + o_rows):
                key = (r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""))
                if key in seen:
                    continue
                seen.add(key)
                merged.append({
                    "Question": r.get("Question", ""),
                    "Constraint": r.get("Constraint", ""),
                    "Input Questions": r.get("Input Questions", ""),
                    "Relationship": r.get("Relationship", ""),
                    "Comment": r.get("Comment", ""),
                })
            if merged:
                write_rows(out_merged, merged)
                print(json.dumps({"gemini_rows": len(g_rows), "claude_rows": len(c_rows), "openai_rows": len(o_rows), "merged_rows": len(merged), "out_merged": out_merged, "strategy": "simple"}))

        elapsed = time.time()-t0
        print(f"[time] step4 (parallel): {elapsed:.2f}s ({elapsed/60:.2f}m)")
        return rc_overall

    if args.command == "step5":
        print("step5 is removed. Use step4 outputs.")
        return 0

    if args.command == "all":
        os.makedirs(args.outdir, exist_ok=True)
        meta_out = os.path.join(args.outdir, "step1_metadata.json")
        pdf_out = os.path.join(args.outdir, "step2_pdf_questions.json")
        combined_out = os.path.join(args.outdir, "step3_grouped_questions.json")
        prior_gemini_csv = os.path.join(args.outdir, "step4_prior_rules.gemini.csv")
        prior_claude_csv = os.path.join(args.outdir, "step4_prior_rules.claude.csv")
        prior_openai_csv = os.path.join(args.outdir, "step4_prior_rules.openai.csv")
        prior_merged_csv = os.path.join(args.outdir, "step4_prior_rules.merged.csv")
        # step5 removed

        print("[step] step1: extract metadata (.sav → metadata.questions.json)")
        t_total0 = time.time()
        t0 = time.time()
        rc = step1_extract_metadata(args.sav, meta_out, include_empty=args.include_empty, indent=args.indent)
        if rc != 0:
            return rc
        t_step1 = time.time()-t0
        print(f"[time] step1: {t_step1:.2f}s ({t_step1/60:.2f}m)")

        print("[step] step2: extract questionnaire (PDF → pdf.questions.llm.json)")
        t0 = time.time()
        rc = step2_extract_llm_pdf(args.pdf, pdf_out, indent=args.indent, api_key=args.api_key)
        if rc != 0:
            return rc
        t_step2 = time.time()-t0
        print(f"[time] step2: {t_step2:.2f}s ({t_step2/60:.2f}m)")

        print("[step] step3: group metadata into multi-selects (metadata → combined.questions.json)")
        t0 = time.time()
        rc = step3_group_metadata(meta_out, combined_out, indent=args.indent, api_key=args.api_key)
        if rc != 0:
            return rc
        t_step3 = time.time()-t0
        print(f"[time] step3: {t_step3:.2f}s ({t_step3/60:.2f}m)")

        print("[step] step4: detect logic rules per-model in parallel and merge")
        t0 = time.time()
        sub = argparse.Namespace(
            pdf=args.pdf,
            metadata=meta_out,
            out_gemini=prior_gemini_csv,
            out_claude=prior_claude_csv,
            out_merged=prior_merged_csv,
            api_key=args.api_key,
            pdf_json=pdf_out,
            opus_model="claude-opus-4-1",
            opus_project=os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            opus_location="us-east5",
        )
        # Reuse the handler logic by crafting commands here
        procs = []
        step4_script = os.path.join(ROOT, "scripts", "step4_detect_logics.py")
        gem_parts = [
            sys.executable,
            shlex.quote(step4_script),
            "--provider", "gemini",
            "--pdf", shlex.quote(sub.pdf),
            "--metadata", shlex.quote(sub.metadata),
            "--output", shlex.quote(sub.out_gemini),
            "--pdf-json", shlex.quote(sub.pdf_json),
        ]
        procs.append(("gemini", run_async(" ".join(gem_parts))))
        # Disable automatic Claude runs in 'all' unless explicitly set
        if False and sub.opus_project:
            claude_parts = [
                sys.executable,
                shlex.quote(step4_script),
                "--provider", "anthropic",
                "--metadata", shlex.quote(sub.metadata),
                "--output", shlex.quote(sub.out_claude),
                "--model", shlex.quote(sub.opus_model),
                "--vertex-project", shlex.quote(sub.opus_project),
                "--vertex-location", shlex.quote(sub.opus_location),
                "--pdf", shlex.quote(sub.pdf),
                "--pdf-json", shlex.quote(sub.pdf_json),
            ]
            procs.append(("claude", run_async(" ".join(claude_parts))))
        # OpenAI provider in parallel if key available
        openai_key_all = os.environ.get("OPENAI_API_KEY")
        if openai_key_all:
            openai_parts = [
                sys.executable,
                shlex.quote(step4_script),
                "--provider", "openai",
                "--pdf", shlex.quote(sub.pdf),
                "--metadata", shlex.quote(sub.metadata),
                "--output", shlex.quote(prior_openai_csv),
                "--model", "gpt-4o-mini",
                "--openai-api-key", shlex.quote(openai_key_all),
                "--pdf-json", shlex.quote(sub.pdf_json),
            ]
            procs.append(("openai", run_async(" ".join(openai_parts))))
        for name, p in procs:
            code = p.wait()
            print(f"[step] step4-{name}: exit {code}")
        # Merge
        def read_rows3(path: str) -> list[dict[str, str]]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    import csv as _csv
                    r = _csv.DictReader(f)
                    return [row for row in r]
            except Exception:
                return []
        g_rows = read_rows3(prior_gemini_csv)
        c_rows = read_rows3(prior_claude_csv)
        o_rows = read_rows3(prior_openai_csv)
        seen = set()
        merged = []
        for r in (g_rows + c_rows + o_rows):
            key = (r.get("Question", ""), r.get("Constraint", ""), r.get("Input Questions", ""), r.get("Relationship", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        with open(prior_merged_csv, "w", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=["Question","Constraint","Input Questions","Relationship","Comment"])
            w.writeheader()
            for r in merged:
                w.writerow(r)
        t_step4 = time.time()-t0
        print(json.dumps({"gemini_rows": len(g_rows), "claude_rows": len(c_rows), "openai_rows": len(o_rows), "merged_rows": len(merged), "out_merged": prior_merged_csv}))
        print(f"[time] step4: {t_step4:.2f}s ({t_step4/60:.2f}m)")

        # finish after step4 (step5 removed)

        t_total = time.time()-t_total0
        summary = {
            "timing": {
                "step1_s": round(t_step1, 2),
                "step1_m": round(t_step1/60, 2),
                "step2_s": round(t_step2, 2),
                "step2_m": round(t_step2/60, 2),
                "step3_s": round(t_step3, 2),
                "step3_m": round(t_step3/60, 2),
                "step4_s": round(t_step4, 2),
                "step4_m": round(t_step4/60, 2),
                
                "total_s": round(t_total, 2),
                "total_m": round(t_total/60, 2),
            }
        }
        print(json.dumps(summary))
        return 0

    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


