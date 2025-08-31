#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

import streamlit as st
import sys
import traceback


ROOT = os.path.abspath(os.path.dirname(__file__))
PIPELINE = os.path.join(ROOT, "pipeline_new.py")
SCRIPTS_DIR = os.path.join(ROOT, "Scripts")


def run_step(cmd: list[str]) -> tuple[int, str, float]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        logs = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, logs, elapsed
    except Exception:
        elapsed = time.time() - start
        return 1, traceback.format_exc(), elapsed


def main() -> None:
    st.set_page_config(page_title="Questionnaire Grouper", page_icon="📊", layout="wide")

    st.title("📊 Questionnaire Grouper")
    st.caption("Upload an SPSS .sav and the questionnaire PDF. The app extracts metadata and groups questions (multi-select), optionally using Gemini 2.5 Flash.")

    with st.sidebar:
        st.header("Settings")
        use_flash = st.toggle("Use Gemini 2.5 Flash (faster)", value=False)
        # Fixed JSON indentation in scripts; no user control
        show_tb = st.toggle("Show Python traceback on error", value=True)
        # Read Gemini key from Streamlit secrets or env; no manual entry in UI
        secret_key = ""
        try:
            secret_key = st.secrets.get("google-gemini-key", "")  # type: ignore[attr-defined]
        except Exception:
            secret_key = os.environ.get("google-gemini-key", "") or os.environ.get("GOOGLE_API_KEY", "")
        run_button = st.button("Run Grouping", type="primary")

    c1, c2 = st.columns(2)
    with c1:
        sav_file = st.file_uploader("Data file (.sav or .xlsx)", type=["sav","xlsx"], accept_multiple_files=False)
    with c2:
        pdf_file = st.file_uploader("Questionnaire (PDF)", type=["pdf"], accept_multiple_files=False)

    st.divider()

    if run_button:
        if not sav_file or not pdf_file:
            st.error("Please upload both the data file and the questionnaire.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = tempfile.mkdtemp(prefix=f"ui_run_{ts}_", dir=None)
        outdir_base = os.path.join(ROOT, "Output", "ui_runs")
        outdir = os.path.join(outdir_base, ts)
        os.makedirs(outdir, exist_ok=True)

        sav_path = os.path.join(workdir, sav_file.name)
        pdf_path = os.path.join(workdir, pdf_file.name)
        with open(sav_path, "wb") as f:
            f.write(sav_file.getbuffer())
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.info("Starting pipeline…")
        pipeline_t0 = time.time()
        status = st.empty()
        # Step 1: metadata (.sav or .xlsx)
        status.write("[step] 1/2 Extract metadata…")
        meta_out = os.path.join(outdir, "step1_metadata.json")
        if sav_path.lower().endswith(".xlsx"):
            step1_cmd = [
                sys.executable, os.path.join(SCRIPTS_DIR, "step1_extract_xlsx_metadata.py"),
                "--input", sav_path,
                "--output", meta_out,
                "--include-empty",
            ]
        else:
            step1_cmd = [
                sys.executable, os.path.join(SCRIPTS_DIR, "step1_extract_spss_metadata.py"),
                "--input", sav_path,
                "--output", meta_out,
                "--include-empty",
            ]
        rc1, logs1, t1 = run_step(step1_cmd)
        if rc1 != 0:
            st.error(f"Step 1 failed ({t1:.1f}s)")
            with st.expander("Logs", expanded=True):
                if not logs1.strip():
                    st.write(f"Return code: {rc1}")
                    st.code("(no output)")
                    st.code("CMD: " + " ".join(step1_cmd))
                else:
                    st.code(logs1)
            if show_tb and logs1.strip() == "":
                st.exception(RuntimeError("Step 1 failed"))
            shutil.rmtree(workdir, ignore_errors=True)
            return
        status.write(f"[time] 1/2 Extract metadata: {t1:.1f}s")

        # Step 2: group with PDF+metadata
        status.write("[step] 2/2 Group with PDF+metadata…")
        step2_cmd = [
            sys.executable, os.path.join(SCRIPTS_DIR, "step2_group_with_pdf_gemini.py"),
            "--pdf", pdf_path,
            "--metadata", meta_out,
            "--output", os.path.join(outdir, "step2_grouped_questions.json"),
        ]
        if use_flash:
            step2_cmd.append("--flash")
        else:
            step2_cmd += ["--model", "gemini-2.5-pro"]
        effective_api_key = (secret_key or os.environ.get("GOOGLE_API_KEY", "")).strip()
        if effective_api_key:
            os.environ["GOOGLE_API_KEY"] = effective_api_key
            step2_cmd += ["--api-key", effective_api_key]
        else:
            st.error("Gemini API key is not configured. Set Streamlit secret 'google-gemini-key' or env 'GOOGLE_API_KEY'.")
            shutil.rmtree(workdir, ignore_errors=True)
            return
        rc2, logs2, t2 = run_step(step2_cmd)
        if rc2 != 0:
            st.error(f"Step 2 failed ({t2:.1f}s)")
            # Surface validation errors from step2 (unknown codes not in metadata)
            try:
                import re
                unknown_in_groups: dict[str, list[str]] = {}
                unknown_standalone: list[str] = []
                invalid_recode_groups: dict[str, list[str]] = {}
                invalid_recode_vars: dict[str, list[str]] = {}
                for line in (logs2 or "").splitlines():
                    m = re.search(r"Non-metadata sub-questions in group '([^']+)': \[(.*?)\]", line)
                    if m:
                        grp = m.group(1)
                        raw = m.group(2)
                        items = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
                        if items:
                            unknown_in_groups[grp] = items
                    m2 = re.search(r"Non-metadata standalone items: \[(.*?)\]", line)
                    if m2:
                        raw2 = m2.group(1)
                        items2 = [x.strip().strip("'\"") for x in raw2.split(",") if x.strip()]
                        unknown_standalone.extend(items2)
                    m3 = re.search(r"Invalid recode_from for group header '([^']+)': \[(.*?)\]", line)
                    if m3:
                        grp2 = m3.group(1)
                        raw3 = m3.group(2)
                        items3 = [x.strip().strip("'\"") for x in raw3.split(",") if x.strip()]
                        if items3:
                            invalid_recode_groups[grp2] = items3
                    m4 = re.search(r"Invalid recode_from for variable '([^']+)': \[(.*?)\]", line)
                    if m4:
                        var = m4.group(1)
                        raw4 = m4.group(2)
                        items4 = [x.strip().strip("'\"") for x in raw4.split(",") if x.strip()]
                        if items4:
                            invalid_recode_vars[var] = items4
                if unknown_in_groups or unknown_standalone or invalid_recode_groups or invalid_recode_vars:
                    st.error("Validation failed: output contains variables not present in metadata or invalid recode sources.")
                    if unknown_in_groups:
                        st.subheader("Unknown sub-questions by group")
                        st.json(unknown_in_groups)
                    if unknown_standalone:
                        st.subheader("Unknown standalone variables")
                        st.json(sorted(set(unknown_standalone)))
                    if invalid_recode_groups:
                        st.subheader("Invalid recode_from (group headers)")
                        st.json(invalid_recode_groups)
                    if invalid_recode_vars:
                        st.subheader("Invalid recode_from (variables)")
                        st.json(invalid_recode_vars)
            except Exception:
                pass
            with st.expander("Logs", expanded=True):
                combined = (logs1 or "") + "\n" + (logs2 or "")
                if not combined.strip():
                    st.write(f"Return code: {rc2}")
                    st.code("(no output)")
                    st.code("CMD: " + " ".join(step2_cmd))
                else:
                    st.code(combined)
            if show_tb and (logs2.strip() == ""):
                st.exception(RuntimeError("Step 2 failed"))
            shutil.rmtree(workdir, ignore_errors=True)
            return
        status.write(f"[time] 2/2 Group with PDF+metadata: {t2:.1f}s")

        # Surface warnings for dropped standalones
        if "[warn] Dropping standalone non-recode items:" in (logs2 or "") or "[warn] Dropped standalone non-metadata items:" in (logs2 or ""):
            try:
                import re
                dropped_non_recode = []
                dropped_non_meta = []
                m1 = re.search(r"\[warn\] Dropping standalone non-recode items: \[(.*?)\]", logs2)
                if m1:
                    raw = m1.group(1)
                    dropped_non_recode = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
                m2 = re.search(r"\[warn\] Dropped standalone non-metadata items: \[(.*?)\]", logs2)
                if m2:
                    raw2 = m2.group(1)
                    dropped_non_meta = [x.strip().strip("'\"") for x in raw2.split(",") if x.strip()]
                if dropped_non_recode:
                    st.warning("Dropped standalone non-recode items (kept only multi and recodes):")
                    st.json(sorted(set(dropped_non_recode)))
                if dropped_non_meta:
                    st.warning("Dropped standalone non-metadata items:")
                    st.json(sorted(set(dropped_non_meta)))
            except Exception:
                pass

        # Success: read outputs
        grouped_path = os.path.join(outdir, "step2_grouped_questions.json")
        if not os.path.exists(grouped_path):
            st.warning("Grouping output not found. See logs below.")
            with st.expander("Logs", expanded=True):
                st.code(logs1 + "\n" + logs2)
            shutil.rmtree(workdir, ignore_errors=True)
            return

        # Step 3: emit compact groups JSON (final output shape)
        status.write("[step] Emit compact groups…")
        groups_path = os.path.join(outdir, "step3_groups.json")
        step3_cmd = [
            sys.executable, os.path.join(SCRIPTS_DIR, "step3_emit_groups.py"),
            "--input", grouped_path,
            "--output", groups_path,
        ]
        rc3, logs3, t3 = run_step(step3_cmd)
        if rc3 != 0 or (not os.path.exists(groups_path)):
            st.error(f"Groups emission failed ({t3:.1f}s)")
            with st.expander("Logs", expanded=True):
                combined3 = (logs1 or "") + "\n" + (logs2 or "") + "\n" + (logs3 or "")
                if not combined3.strip():
                    st.write(f"Return code: {rc3}")
                    st.code("(no output)")
                    st.code("CMD: " + " ".join(step3_cmd))
                else:
                    st.code(combined3)
            shutil.rmtree(workdir, ignore_errors=True)
            return

        total_elapsed = time.time() - pipeline_t0
        status.write(f"Completed in {total_elapsed:.1f}s. Showing results…")
        st.success("Done")

        # Load groups.json
        try:
            with open(groups_path, "r", encoding="utf-8") as f:
                groups_obj = json.load(f)
        except Exception as e:
            st.error("Failed to parse groups JSON.")
            if show_tb:
                st.exception(e)
            with st.expander("Logs", expanded=False):
                st.code(logs1 + "\n" + logs2 + "\n" + logs3)
            shutil.rmtree(workdir, ignore_errors=True)
            return

        groups_list = groups_obj.get("groups") if isinstance(groups_obj, dict) else []
        num_groups = len(groups_list) if isinstance(groups_list, list) else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Groups", f"{num_groups}")
        m2.metric("Flash", "Yes" if use_flash else "No")
        m3.metric("Total time", f"{total_elapsed:.1f}s")

        st.subheader("Groups (first 10)")
        st.json(groups_list[:10], expanded=False)

        st.download_button(
            label="Download groups JSON",
            data=json.dumps(groups_obj, ensure_ascii=False, indent=2),
            file_name="groups.json",
            mime="application/json",
        )

        with st.expander("Full logs"):
            st.code(logs1 + "\n" + logs2 + ("\n" + logs3 if logs3 else ""))

        # Provide direct downloads for all three artifacts
        st.subheader("Downloads")
        try:
            with open(meta_out, "r", encoding="utf-8") as f:
                meta_content = f.read()
        except Exception:
            meta_content = ""
        try:
            with open(grouped_path, "r", encoding="utf-8") as f:
                grouped_content = f.read()
        except Exception:
            grouped_content = ""
        try:
            with open(groups_path, "r", encoding="utf-8") as f:
                final_groups_content = f.read()
        except Exception:
            final_groups_content = ""

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                label="Download step1 metadata",
                data=meta_content,
                file_name="step1_metadata.json",
                mime="application/json",
            )
        with d2:
            st.download_button(
                label="Download step2 grouped",
                data=grouped_content,
                file_name="step2_grouped_questions.json",
                mime="application/json",
            )
        with d3:
            st.download_button(
                label="🔴 Download FINAL groups",
                data=final_groups_content,
                file_name="groups.json",
                mime="application/json",
            )

        # Clean up temp uploads
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()


