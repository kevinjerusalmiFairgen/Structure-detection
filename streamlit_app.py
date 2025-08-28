#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

import streamlit as st


ROOT = os.path.abspath(os.path.dirname(__file__))
PIPELINE = os.path.join(ROOT, "pipeline_new.py")
SCRIPTS_DIR = os.path.join(ROOT, "Scripts")


def run_step(cmd: list[str]) -> tuple[int, str, float]:
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    logs = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, logs, elapsed


def main() -> None:
    st.set_page_config(page_title="Questionnaire Grouper", page_icon="📊", layout="wide")

    st.title("📊 Questionnaire Grouper")
    st.caption("Upload an SPSS .sav and the questionnaire PDF. The app extracts metadata and groups questions (multi-select/grid), optionally using Gemini 2.5 Flash.")

    with st.sidebar:
        st.header("Settings")
        use_flash = st.toggle("Use Gemini 2.5 Flash (faster)", value=False)
        indent = st.slider("JSON indent", min_value=0, max_value=4, value=2, step=1)
        outdir_label = st.text_input("Output folder", value=os.path.join(ROOT, "Output", "ui_runs"))
        run_button = st.button("Run Grouping", type="primary")

    c1, c2 = st.columns(2)
    with c1:
        sav_file = st.file_uploader("SPSS .sav", type=["sav"], accept_multiple_files=False)
    with c2:
        pdf_file = st.file_uploader("Questionnaire PDF", type=["pdf"], accept_multiple_files=False)

    st.divider()

    if run_button:
        if not sav_file or not pdf_file:
            st.error("Please upload both the .sav and the PDF.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = tempfile.mkdtemp(prefix=f"ui_run_{ts}_", dir=None)
        outdir = os.path.join(outdir_label, ts)
        os.makedirs(outdir, exist_ok=True)

        sav_path = os.path.join(workdir, sav_file.name)
        pdf_path = os.path.join(workdir, pdf_file.name)
        with open(sav_path, "wb") as f:
            f.write(sav_file.getbuffer())
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.info("Starting pipeline…")
        status = st.empty()
        # Step 1: metadata
        status.write("[step] 1/2 Extract metadata…")
        step1_cmd = [
            "python", os.path.join(SCRIPTS_DIR, "step1_extract_spss_metadata.py"),
            "--input", sav_path,
            "--output", os.path.join(outdir, "step1_metadata.json"),
            "--indent", str(indent),
            "--include-empty",
        ]
        rc1, logs1, t1 = run_step(step1_cmd)
        if rc1 != 0:
            st.error(f"Step 1 failed ({t1:.1f}s)")
            with st.expander("Logs", expanded=True):
                st.code(logs1)
            shutil.rmtree(workdir, ignore_errors=True)
            return
        status.write(f"[time] 1/2 Extract metadata: {t1:.1f}s")

        # Step 2: group with PDF+metadata
        status.write("[step] 2/2 Group with PDF+metadata…")
        step2_cmd = [
            "python", os.path.join(SCRIPTS_DIR, "step2_group_with_pdf_gemini.py"),
            "--pdf", pdf_path,
            "--metadata", os.path.join(outdir, "step1_metadata.json"),
            "--output", os.path.join(outdir, "step2_grouped_questions.json"),
            "--indent", str(indent),
        ]
        if use_flash:
            step2_cmd.append("--flash")
        rc2, logs2, t2 = run_step(step2_cmd)
        if rc2 != 0:
            st.error(f"Step 2 failed ({t2:.1f}s)")
            with st.expander("Logs", expanded=True):
                st.code(logs1 + "\n" + logs2)
            shutil.rmtree(workdir, ignore_errors=True)
            return
        status.write(f"[time] 2/2 Group with PDF+metadata: {t2:.1f}s")

        # Success: read outputs
        meta_path = os.path.join(outdir, "step1_metadata.json")
        grouped_path = os.path.join(outdir, "step2_grouped_questions.json")
        if not os.path.exists(grouped_path):
            st.warning("Grouping output not found. See logs below.")
            with st.expander("Logs", expanded=True):
                st.code(logs1 + "\n" + logs2)
            shutil.rmtree(workdir, ignore_errors=True)
            return

        status.write("Completed. Showing results…")
        st.success("Done")

        # Quick metrics
        try:
            with open(grouped_path, "r", encoding="utf-8") as f:
                grouped = json.load(f)
        except Exception:
            st.error("Failed to parse grouped JSON.")
            with st.expander("Logs", expanded=False):
                st.code(logs)
            shutil.rmtree(workdir, ignore_errors=True)
            return

        num_items = len(grouped) if isinstance(grouped, list) else 0
        num_groups = sum(1 for x in grouped if isinstance(x, dict) and x.get("sub_questions"))

        m1, m2, m3 = st.columns(3)
        m1.metric("Items", f"{num_items}")
        m2.metric("Groups", f"{num_groups}")
        m3.metric("Flash", "Yes" if use_flash else "No")

        st.subheader("Sample (first 15)")
        st.json(grouped[:15], expanded=False)

        st.download_button(
            label="Download grouped JSON",
            data=json.dumps(grouped, ensure_ascii=False, indent=indent),
            file_name="grouped_questions.json",
            mime="application/json",
        )

        with st.expander("Full logs"):
            st.code(logs1 + "\n" + logs2)

        # Clean up temp uploads
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()


