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
        cmd = f"{cmd} > /dev/null 2>&1"
    return subprocess.call(cmd, shell=True)


def step1_extract_metadata(sav_path: str, output_json: str, include_empty: bool = False, *, quiet: bool = False) -> int:
    script = os.path.join(ROOT, "Scripts", "step1_extract_spss_metadata.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--input", shlex.quote(sav_path),
        "--output", shlex.quote(output_json),
        # fixed indentation inside script
    ]
    if include_empty:
        args.append("--include-empty")
    return run(" ".join(args), quiet=quiet)


def step2_extract_llm_pdf(pdf_path: str, metadata_path: str, output_json: str, api_key: str | None = None, *, quiet: bool = False, flash: bool = False) -> int:
    script = os.path.join(ROOT, "Scripts", "step2_group_with_pdf_gemini.py")
    args = [
        sys.executable,
        shlex.quote(script),
        "--pdf", shlex.quote(pdf_path),
        "--metadata", shlex.quote(metadata_path),
        "--output", shlex.quote(output_json),
    ]
    if api_key:
        args += ["--api-key", shlex.quote(api_key)]
    if flash:
        args += ["--flash"]
    return run(" ".join(args), quiet=quiet)

def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline (steps 1–2): metadata, group-with-PDF")
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="Run steps 1–2")
    p_all.add_argument("--sav", required=True)
    p_all.add_argument("--pdf", required=True)
    p_all.add_argument("--outdir", required=False, default=os.path.join(ROOT, "Output"))
    # removed indent control; scripts use fixed indentation
    p_all.add_argument("--api-key")
    p_all.add_argument("--verbose", action="store_true")
    p_all.add_argument("--flash", action="store_true", help="Use Gemini 2.5 Flash (thinking_budget 4096)")

    args = parser.parse_args()

    if args.command == "all":
        os.makedirs(args.outdir, exist_ok=True)
        concise = not getattr(args, "verbose", False)
        meta_out = os.path.join(args.outdir, "step1_metadata.json")
        pdf_out = os.path.join(args.outdir, "step2_grouped_questions.json")
        groups_out = os.path.join(args.outdir, "step3_groups.json")

        # Step 1
        print("[step] 1/2 Extract metadata…", flush=True)
        t0 = time.time()
        rc = step1_extract_metadata(args.sav, meta_out, include_empty=True, quiet=True)
        if rc != 0:
            return rc
        t_step1 = time.time()-t0
        print(f"[time] 1/2 Extract metadata: {t_step1:.1f}s", flush=True)

        # Step 2
        print("[step] 2/2 Group with PDF+metadata…", flush=True)
        t0 = time.time()
        rc = step2_extract_llm_pdf(
            args.pdf,
            meta_out,
            pdf_out,
            api_key=args.api_key,
            quiet=True,
            flash=getattr(args, "flash", False),
        )
        t_step2 = time.time()-t0
        print(f"[time] 2/2 Group with PDF+metadata: {t_step2:.1f}s", flush=True)
        # Emit compact groups as final layer
        try:
            rc3 = subprocess.call(
                [
                    "python",
                    os.path.join(ROOT, "Scripts", "step3_emit_groups.py"),
                    "--input", pdf_out,
                    "--output", groups_out,
                ],
                shell=False,
            )
            if rc3 == 0:
                print(f"[time] total: {t_step1 + t_step2:.1f}s", flush=True)
                print(f"[groups] {groups_out}")
            else:
                print(f"[warn] groups emission failed (exit {rc3})")
                print(f"[time] total: {t_step1 + t_step2:.1f}s", flush=True)
        except Exception as exc:
            print(f"[warn] groups emission error: {exc}")
            print(f"[time] total: {t_step1 + t_step2:.1f}s", flush=True)
        if rc != 0:
            return rc
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())


