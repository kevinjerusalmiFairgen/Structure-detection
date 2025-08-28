#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.dirname(__file__))


def run(cmd: str, quiet: bool = False) -> int:
    if not quiet:
        print(f"$ {cmd}")
    if quiet:
        cmd = f"{cmd} > /dev/null"
    return subprocess.call(cmd, shell=True)


def step1_extract_metadata(sav_path: str, output_json: str, include_empty: bool = False, indent: int = 2, *, quiet: bool = False) -> int:
    script = os.path.join(ROOT, "Scripts", "step1_extract_spss_metadata.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--input", shlex.quote(sav_path),
        "--output", shlex.quote(output_json),
        "--indent", str(indent),
    ]
    if include_empty:
        args.append("--include-empty")
    return run(" ".join(args), quiet=quiet)


def step2_extract_llm_pdf(pdf_path: str, output_json: str, indent: int = 2, api_key: str | None = None, *, quiet: bool = False) -> int:
    script = os.path.join(ROOT, "Scripts", "step2_extract_pdf_questions_gemini.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--pdf", shlex.quote(pdf_path),
        "--output", shlex.quote(output_json),
        "--indent", str(indent),
    ]
    if api_key:
        args += ["--api-key", shlex.quote(api_key)]
    return run(" ".join(args), quiet=quiet)


def step3_group_metadata(metadata_json: str, output_json: str, indent: int = 2, api_key: str | None = None, *, quiet: bool = False) -> int:
    script = os.path.join(ROOT, "Scripts", "step3_group_metadata_gemini.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--metadata", shlex.quote(metadata_json),
        "--output", shlex.quote(output_json),
        "--indent", str(indent),
    ]
    if api_key:
        args += ["--api-key", shlex.quote(api_key)]
    return run(" ".join(args), quiet=quiet)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline (steps 1–3): metadata, pdf extraction, grouping")
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="Run steps 1–3")
    p_all.add_argument("--sav", required=True)
    p_all.add_argument("--pdf", required=True)
    p_all.add_argument("--outdir", required=False, default=os.path.join(ROOT, "Output"))
    p_all.add_argument("--indent", type=int, default=2)
    p_all.add_argument("--include-empty", action="store_true")
    p_all.add_argument("--api-key")
    p_all.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.command == "all":
        os.makedirs(args.outdir, exist_ok=True)
        concise = not getattr(args, "verbose", False)
        meta_out = os.path.join(args.outdir, "step1_metadata.json")
        pdf_out = os.path.join(args.outdir, "step2_pdf_questions.json")
        combined_out = os.path.join(args.outdir, "step3_grouped_questions.json")

        print("Step 1/3: SPSS metadata…", flush=True) if concise else None
        t0 = time.time()
        rc = step1_extract_metadata(args.sav, meta_out, include_empty=args.include_empty, indent=args.indent, quiet=concise)
        if rc != 0:
            return rc
        print(f"done {time.time()-t0:.1f}s", flush=True) if concise else None

        print("Step 2/3: PDF extraction…", flush=True) if concise else None
        t0 = time.time()
        rc = step2_extract_llm_pdf(args.pdf, pdf_out, indent=args.indent, api_key=args.api_key, quiet=concise)
        if rc != 0:
            return rc
        print(f"done {time.time()-t0:.1f}s", flush=True) if concise else None

        print("Step 3/3: Grouping…", flush=True) if concise else None
        t0 = time.time()
        rc = step3_group_metadata(meta_out, combined_out, indent=args.indent, api_key=args.api_key, quiet=concise)
        if rc != 0:
            return rc
        print(f"done {time.time()-t0:.1f}s", flush=True) if concise else None
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())


