from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fpa_variance_tool.ai_assistant import ask_ai_assistant, build_ai_context
from fpa_variance_tool.commentary import (
    format_strategic_insights,
    generate_full_fpa_report,
    generate_driver_commentary,
    generate_executive_summary,
    generate_strategic_insights,
)
from fpa_variance_tool.data_qa import answer_question_from_fpa_data
from fpa_variance_tool.explainable_insights import generate_explainable_insights
from fpa_variance_tool.pipeline import prepare_analysis_dataset
from fpa_variance_tool.random_file_report import (
    SUPPORTED_RANDOM_FILE_TYPES,
    profile_random_file,
)
from fpa_variance_tool.report_evaluation import evaluate_ai_generated_report
from fpa_variance_tool.reporting import build_filter_summary, build_html_report, write_pdf_report
from fpa_variance_tool.variance_analysis import get_top_variance_drivers_list, summarize_by_dimension
from fpa_variance_tool.visualization import build_chart_pack


DEFAULT_BUDGET = PROJECT_ROOT / "sample_data" / "budget_demo.csv"
DEFAULT_ACTUALS = PROJECT_ROOT / "sample_data" / "actuals_demo.csv"


def inject_brand_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fiq-violet-900: #2d1457;
            --fiq-violet-700: #5a2ca0;
            --fiq-violet-600: #6f3cc4;
            --fiq-violet-500: #8356d8;
            --fiq-violet-100: #f3edff;
            --fiq-white: #ffffff;
            --fiq-ink: #1f1c2c;
            --fiq-muted: #6c6880;
            --fiq-border: rgba(111, 60, 196, 0.14);
            --fiq-shadow: 0 18px 45px rgba(67, 35, 121, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(131, 86, 216, 0.14), transparent 24%),
                radial-gradient(circle at left 10%, rgba(111, 60, 196, 0.10), transparent 20%),
                linear-gradient(180deg, #fbf9ff 0%, #f6f2ff 38%, #ffffff 100%);
            color: var(--fiq-ink);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(45, 20, 87, 0.98) 0%, rgba(63, 30, 112, 0.98) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f7f2ff;
        }

        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stFileUploader label {
            color: #ffffff !important;
            font-weight: 600;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
        }

        [data-testid="stSidebar"] .stCaption {
            color: rgba(247, 242, 255, 0.75) !important;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.4rem;
            max-width: 1340px;
        }

        .fiq-hero {
            position: relative;
            overflow: hidden;
            padding: 1.7rem 1.8rem;
            border-radius: 28px;
            margin-bottom: 1.25rem;
            background:
                linear-gradient(135deg, rgba(45, 20, 87, 0.98) 0%, rgba(111, 60, 196, 0.96) 56%, rgba(147, 106, 228, 0.95) 100%);
            box-shadow: var(--fiq-shadow);
            color: #ffffff;
        }

        .fiq-hero::after {
            content: "";
            position: absolute;
            inset: auto -50px -60px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.02) 70%);
        }

        .fiq-brand {
            font-size: 0.82rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            font-weight: 700;
            opacity: 0.86;
            margin-bottom: 0.7rem;
        }

        .fiq-title {
            font-size: 2.55rem;
            line-height: 1.04;
            font-weight: 800;
            margin: 0;
        }

        .fiq-subtitle {
            max-width: 760px;
            margin-top: 0.7rem;
            font-size: 1rem;
            line-height: 1.6;
            color: rgba(255,255,255,0.9);
        }

        .fiq-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }

        .fiq-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.16);
            font-size: 0.84rem;
            font-weight: 600;
        }

        .fiq-section {
            padding: 1.05rem 1.2rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--fiq-border);
            box-shadow: var(--fiq-shadow);
            margin: 0.5rem 0 1rem 0;
            backdrop-filter: blur(6px);
        }

        .fiq-section-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--fiq-violet-900);
            margin-bottom: 0.2rem;
        }

        .fiq-section-copy {
            color: var(--fiq-muted);
            font-size: 0.95rem;
            margin-bottom: 0.1rem;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--fiq-border);
            border-radius: 20px;
            padding: 0.85rem 1rem;
            box-shadow: var(--fiq-shadow);
        }

        [data-testid="stMetricLabel"] {
            color: var(--fiq-muted);
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--fiq-violet-900);
            font-weight: 800;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            padding: 0.35rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--fiq-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px;
            padding: 0.55rem 1rem;
            font-weight: 700;
            color: var(--fiq-violet-900);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--fiq-violet-700), var(--fiq-violet-500)) !important;
            color: #ffffff !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(111, 60, 196, 0.18);
            background: linear-gradient(135deg, var(--fiq-violet-700), var(--fiq-violet-500));
            color: #ffffff;
            font-weight: 700;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(111, 60, 196, 0.22);
            color: #ffffff;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stCodeBlock"] {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid var(--fiq-border);
            box-shadow: var(--fiq-shadow);
        }

        .fiq-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: rgba(111, 60, 196, 0.08);
            color: var(--fiq-violet-700);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .fiq-alert {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid var(--fiq-border);
            box-shadow: var(--fiq-shadow);
            margin: 0.55rem 0 0.9rem 0;
            font-weight: 600;
            color: var(--fiq-violet-900);
        }

        .fiq-alert-soft {
            background: rgba(111, 60, 196, 0.08);
        }

        .fiq-alert-strong {
            background: linear-gradient(135deg, rgba(90, 44, 160, 0.12), rgba(131, 86, 216, 0.16));
        }

        .fiq-confidence-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 84px;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border: 1px solid rgba(111, 60, 196, 0.16);
        }

        .fiq-confidence-high {
            background: linear-gradient(135deg, #4d2290, #6f3cc4);
            color: #ffffff;
        }

        .fiq-confidence-medium {
            background: linear-gradient(135deg, #a783e9, #c3acf2);
            color: #2d1457;
        }

        .fiq-confidence-low {
            background: #f5f0ff;
            color: #5a2ca0;
            border-color: rgba(90, 44, 160, 0.26);
        }

        .fiq-score-shell {
            padding: 1.1rem 1.15rem;
            border-radius: 22px;
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--fiq-border);
            box-shadow: var(--fiq-shadow);
        }

        .fiq-score-main {
            display: flex;
            align-items: end;
            gap: 0.7rem;
            margin-bottom: 0.8rem;
        }

        .fiq-score-value {
            font-size: 2.5rem;
            line-height: 1;
            font-weight: 900;
            color: var(--fiq-violet-900);
        }

        .fiq-score-grade {
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--fiq-violet-700);
            padding-bottom: 0.3rem;
        }

        .fiq-progress-list {
            display: grid;
            gap: 0.75rem;
            margin-top: 0.8rem;
        }

        .fiq-progress-label {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.28rem;
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--fiq-violet-900);
        }

        .fiq-progress-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(111, 60, 196, 0.10);
            overflow: hidden;
        }

        .fiq-progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #c3acf2 0%, #8356d8 55%, #4d2290 100%);
        }

        .fiq-mini-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--fiq-border);
            box-shadow: var(--fiq-shadow);
            height: 100%;
        }

        .fiq-mini-card-title {
            font-size: 0.88rem;
            font-weight: 800;
            color: var(--fiq-violet-700);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.45rem;
        }

        .fiq-mini-card ul {
            margin: 0;
            padding-left: 1rem;
            color: var(--fiq-ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    st.markdown(
        """
        <section class="fiq-hero">
            <div class="fiq-brand">fintechIQ</div>
            <h1 class="fiq-title">FP&amp;A Intelligence Workspace</h1>
            <div class="fiq-subtitle">
                Variance analysis, management commentary, validation, and ad-hoc file intelligence in one violet-and-white finance cockpit.
            </div>
            <div class="fiq-chip-row">
                <span class="fiq-chip">Budget vs. Actual</span>
                <span class="fiq-chip">Executive Summaries</span>
                <span class="fiq-chip">Driver Analysis</span>
                <span class="fiq-chip">Random File Reports</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, copy: str, kicker: str | None = None) -> None:
    kicker_html = f'<div class="fiq-kicker">{kicker}</div>' if kicker else ""
    st.markdown(
        f"""
        <div class="fiq-section">
            {kicker_html}
            <div class="fiq-section-title">{title}</div>
            <div class="fiq-section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_violet_notice(message: str, tone: str = "soft") -> None:
    tone_class = "fiq-alert-strong" if tone == "strong" else "fiq-alert-soft"
    st.markdown(
        f'<div class="fiq-alert {tone_class}">{message}</div>',
        unsafe_allow_html=True,
    )


def render_confidence_summary_cards(items: list[dict[str, object]]) -> None:
    st.markdown("**Confidence Summary**")
    for item in items:
        level = str(item.get("confidence_level", "Medium")).lower()
        insight = str(item.get("insight", ""))
        reason = str(item.get("confidence_reason", ""))
        st.markdown(
            f"""
            <div class="fiq-mini-card" style="margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start;">
                    <div style="font-weight:700; color:#1f1c2c; line-height:1.45;">{insight}</div>
                    <span class="fiq-confidence-pill fiq-confidence-{level}">{item.get("confidence_level", "Medium")}</span>
                </div>
                <div style="margin-top:0.65rem; color:#6c6880; font-size:0.92rem;">{reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_quality_scorecard(evaluation: dict[str, object]) -> None:
    criteria = evaluation.get("criteria", [])
    weaknesses = evaluation.get("weaknesses", [])
    suggestions = evaluation.get("suggestions", [])
    st.markdown(
        f"""
        <div class="fiq-score-shell">
            <div class="fiq-score-main">
                <div class="fiq-score-value">{evaluation['total_score']}/100</div>
                <div class="fiq-score-grade">{evaluation['grade']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_html = ['<div class="fiq-progress-list">']
    for item in criteria:
        score = int(item["score"])
        progress_html.append(
            f"""
            <div>
                <div class="fiq-progress-label">
                    <span>{item['criterion']}</span>
                    <span>{score}/10</span>
                </div>
                <div class="fiq-progress-track">
                    <div class="fiq-progress-fill" style="width:{score * 10}%;"></div>
                </div>
                <div style="margin-top:0.28rem; color:#6c6880; font-size:0.88rem;">{item['reason']}</div>
            </div>
            """
        )
    progress_html.append("</div>")
    st.markdown("".join(progress_html), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        weakness_items = "".join(f"<li>{item}</li>" for item in weaknesses) or "<li>No major weakness identified.</li>"
        st.markdown(
            f"""
            <div class="fiq-mini-card">
                <div class="fiq-mini-card-title">Top Weaknesses</div>
                <ul>{weakness_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        suggestion_items = "".join(f"<li>{item}</li>" for item in suggestions) or "<li>No immediate improvement suggestion generated.</li>"
        st.markdown(
            f"""
            <div class="fiq-mini-card">
                <div class="fiq-mini-card-title">Improvement Suggestions</div>
                <ul>{suggestion_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def init_ai_state() -> None:
    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []


def dataframe_to_csv_bytes(df) -> bytes:
    export_df = df.copy()
    for column in export_df.columns:
        if hasattr(export_df[column], "dt"):
            try:
                export_df[column] = export_df[column].dt.strftime("%Y-%m-%d")
            except AttributeError:
                pass
    return export_df.to_csv(index=False).encode("utf-8")


def build_export_bundle(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data
def build_management_pack(
    variance_df,
    drivers_df,
    executive_summary: str,
    driver_commentary: str,
    strategic_insights: str,
    system_validation_text: str,
    full_report_text: str,
    filter_summary: str,
    variance_csv: bytes,
    drivers_csv: bytes,
    drilldown_csv: bytes,
    anomalies_csv: bytes,
    quality_csv: bytes,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        chart_paths = build_chart_pack(variance_df, temp_dir)
        html_path = build_html_report(
            variance_df=variance_df,
            drivers_df=drivers_df.head(5),
            chart_paths=chart_paths,
            commentary=executive_summary,
            output_dir=temp_dir,
            report_title="FP&A Management Pack",
            filter_summary=filter_summary,
            driver_commentary=driver_commentary,
            strategic_insights=strategic_insights,
            inline_images=True,
        )

        html_bytes = Path(html_path).read_bytes()
        pdf_buffer = io.BytesIO()
        write_pdf_report(
            drivers_df=drivers_df.head(5),
            chart_paths=chart_paths,
            commentary=executive_summary,
            pdf_target=pdf_buffer,
            report_title="FP&A Management Pack",
            filter_summary=filter_summary,
            driver_commentary=driver_commentary,
            strategic_insights=strategic_insights,
        )
        pdf_bytes = pdf_buffer.getvalue()

        bundle_files = {
            "management_pack.html": html_bytes,
            "management_pack.pdf": pdf_bytes,
            "executive_summary.txt": executive_summary.encode("utf-8"),
            "driver_commentary.txt": driver_commentary.encode("utf-8"),
            "strategic_insights.txt": strategic_insights.encode("utf-8"),
            "system_validation_report.txt": system_validation_text.encode("utf-8"),
            "full_fpa_report.txt": full_report_text.encode("utf-8"),
            "filtered_variance.csv": variance_csv,
            "filtered_drivers.csv": drivers_csv,
            "drilldown.csv": drilldown_csv,
            "filtered_anomalies.csv": anomalies_csv,
            "quality_summary.csv": quality_csv,
        }
        for chart_name, chart_path in chart_paths.items():
            bundle_files[f"charts/{Path(chart_path).name}"] = Path(chart_path).read_bytes()

        return {
            "html": html_bytes,
            "pdf": pdf_bytes,
            "zip": build_export_bundle(bundle_files),
        }


@st.cache_data
def load_analysis_from_paths(budget_path: str, actuals_path: str):
    return prepare_analysis_dataset(budget_path=budget_path, actuals_path=actuals_path)


@st.cache_data
def load_analysis_from_uploads(
    budget_bytes: bytes,
    budget_name: str,
    actuals_bytes: bytes,
    actuals_name: str,
):
    budget_buffer = io.BytesIO(budget_bytes)
    budget_buffer.name = budget_name
    actuals_buffer = io.BytesIO(actuals_bytes)
    actuals_buffer.name = actuals_name
    return prepare_analysis_dataset(budget_path=budget_buffer, actuals_path=actuals_buffer)


@st.cache_data
def load_random_report_from_upload(file_bytes: bytes, file_name: str):
    return profile_random_file(file_bytes=file_bytes, file_name=file_name)


def render_random_file_report(report: dict[str, object], key_prefix: str) -> None:
    st.caption(f"{report['file_type']} file")
    if report["kind"] == "tabular":
        st.markdown("**Dataset Overview**")
        for item in report["dataset_overview"]:
            st.write(f"- {item}")

        st.markdown("**Key Metrics**")
        for item in report["key_metrics"]:
            st.write(f"- {item}")

        st.markdown("**Key Findings**")
        for item in report["key_findings"]:
            st.write(f"- {item}")

        st.markdown("**Root Cause Analysis**")
        for item in report["root_cause_analysis"]:
            st.write(f"- {item}")

        st.markdown("**Anomalies Identified**")
        for item in report["anomalies_identified"]:
            st.write(f"- {item}")

        st.markdown("**Business Impact**")
        for item in report["business_impact"]:
            st.write(f"- {item}")

        st.markdown("**Recommendations**")
        for item in report["recommendations"]:
            st.write(f"- {item}")

        st.markdown("**Recommended Visualizations**")
        for item in report["visualization_recommendations"]:
            st.write(f"- {item}")

        quality_lookup = report["quality_summary"].set_index("metric")["value"].to_dict()
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Rows", f"{int(quality_lookup.get('Rows', 0)):,}")
        metric_col2.metric("Columns", f"{int(quality_lookup.get('Columns', 0)):,}")
        metric_col3.metric("Missing cells", f"{int(quality_lookup.get('Missing cells', 0)):,}")
        metric_col4.metric("Duplicate rows", f"{int(quality_lookup.get('Duplicate rows', 0)):,}")

        st.markdown("**Preview**")
        st.dataframe(report["preview_df"], use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)
        if not report["bar_chart_df"].empty:
            bar_chart_df = report["bar_chart_df"].set_index(report["bar_category"])[[report["bar_value"]]]
            chart_col1.markdown("**Bar Chart**")
            chart_col1.bar_chart(bar_chart_df)
        else:
            chart_col1.info("No suitable category + numeric combination was found for a bar chart.")

        if not report["pie_chart_df"].empty:
            pie_chart_df = report["pie_chart_df"]
            figure, axis = plt.subplots(figsize=(5, 5))
            axis.pie(
                pie_chart_df[report["pie_value"]],
                labels=pie_chart_df[report["pie_category"]],
                autopct="%1.1f%%",
                startangle=90,
            )
            axis.axis("equal")
            chart_col2.markdown("**Pie Chart**")
            chart_col2.pyplot(figure)
            plt.close(figure)
        else:
            chart_col2.info("A pie chart is available when the file contains a stable category split and numeric values.")

        if not report["trend_chart_df"].empty:
            st.markdown("**Trend View**")
            trend_chart_df = report["trend_chart_df"].set_index(report["trend_date"])[[report["trend_value"]]]
            st.line_chart(trend_chart_df)

        detail_col1, detail_col2 = st.columns(2)
        detail_col1.markdown("**Missing Values by Column**")
        if report["missing_by_column"].empty:
            detail_col1.success("No column-level missing values were detected.")
        else:
            detail_col1.dataframe(report["missing_by_column"], use_container_width=True)

        detail_col2.markdown("**Numeric Summary**")
        if report["numeric_summary"].empty:
            detail_col2.info("No numeric columns were detected for descriptive statistics.")
        else:
            detail_col2.dataframe(report["numeric_summary"], use_container_width=True)
    else:
        st.markdown("**Summary**")
        st.write(report["summary"])

        st.markdown("**Key Points**")
        for item in report["key_points"]:
            st.write(f"- {item}")

        st.markdown("**Insights**")
        for item in report["insights"]:
            st.write(f"- {item}")

        st.markdown("**Themes**")
        for item in report["themes"]:
            st.write(f"- {item}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Lines", f"{int(report['line_count']):,}")
        metric_col2.metric("Words", f"{int(report['word_count']):,}")
        metric_col3.metric("Characters", f"{int(report['character_count']):,}")
        st.markdown("**Text Preview**")
        st.code(report["preview_text"] or "No readable text preview is available.", language="text")

    st.download_button(
        "Download quick report",
        data=report["report_text"].encode("utf-8"),
        file_name=f"{Path(str(report['file_name'])).stem}_quick_report.txt",
        mime="text/plain",
        key=f"{key_prefix}_download",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="fintechIQ | FP&A Intelligence", layout="wide")
    init_ai_state()
    inject_brand_theme()
    render_brand_header()

    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.82rem; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 800; opacity: 0.8;">fintechIQ</div>
                <div style="font-size: 1.45rem; font-weight: 800; line-height: 1.1;">Control Center</div>
                <div style="margin-top: 0.35rem; color: rgba(247,242,255,0.76); font-size: 0.92rem;">Load data, adjust views, and generate finance-ready outputs.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Data Sources")
        input_mode = st.radio(
            "Choose data input",
            ["Upload files", "Use sample data", "Use file paths"],
        )

        budget_path = str(DEFAULT_BUDGET)
        actuals_path = str(DEFAULT_ACTUALS)
        budget_upload = None
        actuals_upload = None
        random_uploads = []

        if input_mode == "Upload files":
            budget_upload = st.file_uploader(
                "Budget file",
                type=["csv", "xlsx", "xls"],
                help="Upload a budget export with date, department, region, product_line, account, and amount columns.",
            )
            actuals_upload = st.file_uploader(
                "Actuals file",
                type=["csv", "xlsx", "xls"],
                help="Upload an actuals export with the same structure as the budget file.",
            )
            st.caption(
                "Required columns: date, department, region, product_line, account, amount."
            )
            st.markdown("---")
            st.subheader("Upload Files Here")
            random_uploads = st.file_uploader(
                "Random files for quick report",
                type=list(SUPPORTED_RANDOM_FILE_TYPES),
                accept_multiple_files=True,
                key="random_file_report_uploads",
                help="Upload ad-hoc Excel, CSV, or TXT files to generate a quick structure, quality, and chart report.",
            )
            st.caption(
                "Use this area for one-off files that are not ready for the full budget vs. actual FP&A pipeline."
            )
        elif input_mode == "Use file paths":
            budget_path = st.text_input("Budget file", str(DEFAULT_BUDGET))
            actuals_path = st.text_input("Actuals file", str(DEFAULT_ACTUALS))
        else:
            st.caption("Using the bundled demo files from the sample_data folder.")
        ai_api_key = os.getenv("OPENAI_API_KEY", "")
        ai_model = "gpt-5.1"
        if ai_api_key:
            st.header("AI Assistant")
            ai_model = st.selectbox(
                "AI model",
                ["gpt-5.1", "gpt-5-mini"],
                index=0,
                help="GPT-5.1 is the current flagship default in OpenAI docs; GPT-5-mini is a lower-cost option.",
            )

    random_reports: list[dict[str, object]] = []
    random_report_errors: list[str] = []
    for upload in random_uploads:
        try:
            random_reports.append(
                load_random_report_from_upload(
                    file_bytes=upload.getvalue(),
                    file_name=upload.name,
                )
            )
        except Exception as exc:
            random_report_errors.append(f"{upload.name}: {exc}")

    if random_reports or random_report_errors:
        render_section_intro(
            "Random File Reports",
            "Drop in one-off Excel, CSV, or TXT files and fintechIQ will classify them as document or dataset workflows automatically.",
            kicker="Ad-Hoc Analysis",
        )

        if random_reports:
            report_tabs = st.tabs([str(report["file_name"]) for report in random_reports])
            for index, (tab, report) in enumerate(zip(report_tabs, random_reports, strict=False)):
                with tab:
                    render_random_file_report(report, key_prefix=f"random_report_{index}")

        for error_message in random_report_errors:
            st.error(f"Random file report error: {error_message}")

    prepared = None
    analysis_error = None
    try:
        if input_mode == "Upload files":
            if budget_upload and actuals_upload:
                prepared = load_analysis_from_uploads(
                    budget_bytes=budget_upload.getvalue(),
                    budget_name=budget_upload.name,
                    actuals_bytes=actuals_upload.getvalue(),
                    actuals_name=actuals_upload.name,
                )
        else:
            prepared = load_analysis_from_paths(budget_path, actuals_path)
    except Exception as exc:
        analysis_error = str(exc)

    if analysis_error:
        st.error(f"Unable to load the selected files: {analysis_error}")
        return
    if prepared is None:
        if input_mode == "Upload files":
            st.info("Upload both a budget file and an actuals file to start the FP&A analysis.")
        return

    variance_df = prepared["variance_df"]
    drivers_df = prepared["drivers_df"]
    top_variance_drivers_list = prepared["top_variance_drivers_list"]
    anomaly_df = prepared["anomaly_df"]
    quality_summary = prepared["quality_summary"]
    system_validation_df = prepared["system_validation_df"]
    system_validation_text = prepared["system_validation_text"]
    validation_status = prepared["validation_status"]
    budget_column_mapping = prepared["budget_column_mapping"]
    actual_column_mapping = prepared["actual_column_mapping"]

    periods = sorted(variance_df["date"].dt.strftime("%Y-%m").unique())
    departments = ["All"] + sorted(variance_df["department"].dropna().unique().tolist())
    regions = ["All"] + sorted(variance_df["region"].dropna().unique().tolist())
    product_lines = ["All"] + sorted(variance_df["product_line"].dropna().unique().tolist())
    variance_classes = ["All"] + sorted(variance_df["variance_classification"].dropna().unique().tolist())

    with st.sidebar:
        st.header("Filters")
        selected_periods = st.multiselect("Periods", periods, default=periods)
        selected_department = st.selectbox("Department", departments)
        selected_region = st.selectbox("Region", regions)
        selected_product_line = st.selectbox("Product Line", product_lines)
        selected_variance_class = st.selectbox("Variance Classification", variance_classes)

    filtered = variance_df.copy()
    filtered = filtered[filtered["date"].dt.strftime("%Y-%m").isin(selected_periods)]
    if selected_department != "All":
        filtered = filtered[filtered["department"] == selected_department]
    if selected_region != "All":
        filtered = filtered[filtered["region"] == selected_region]
    if selected_product_line != "All":
        filtered = filtered[filtered["product_line"] == selected_product_line]
    if selected_variance_class != "All":
        filtered = filtered[filtered["variance_classification"] == selected_variance_class]

    filtered_anomalies = anomaly_df.copy()
    if not filtered_anomalies.empty:
        filtered_anomalies = filtered_anomalies[
            filtered_anomalies["date"].dt.strftime("%Y-%m").isin(selected_periods)
        ]
        if selected_department != "All":
            filtered_anomalies = filtered_anomalies[
                filtered_anomalies["department"] == selected_department
            ]
        if selected_region != "All":
            filtered_anomalies = filtered_anomalies[filtered_anomalies["region"] == selected_region]
        if selected_product_line != "All":
            filtered_anomalies = filtered_anomalies[
                filtered_anomalies["product_line"] == selected_product_line
            ]

    summary = summarize_by_dimension(filtered, ["date"]).sort_values("date")
    total_budget = filtered["budget"].sum()
    total_actual = filtered["actual"].sum()
    total_variance = filtered["variance_abs"].sum()
    variance_direction = "Above Plan" if total_variance >= 0 else "Below Plan"
    active_filters = []
    if len(selected_periods) != len(periods):
        active_filters.append(f"{len(selected_periods)} periods")
    if selected_department != "All":
        active_filters.append(selected_department)
    if selected_region != "All":
        active_filters.append(selected_region)
    if selected_product_line != "All":
        active_filters.append(selected_product_line)
    if selected_variance_class != "All":
        active_filters.append(selected_variance_class)
    active_filter_text = " | ".join(active_filters) if active_filters else "All company views active"

    render_section_intro(
        "Current FP&A View",
        f"{variance_direction} with variance of {format_currency(total_variance)}. Active scope: {active_filter_text}.",
        kicker="Live View",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Budget", format_currency(total_budget))
    col2.metric("Actual", format_currency(total_actual))
    col3.metric("Variance", format_currency(total_variance))
    col4.metric("Status", variance_direction)

    st.subheader("Executive Summary")
    filtered_drivers_for_summary = drivers_df.copy()
    if selected_department != "All":
        filtered_drivers_for_summary = filtered_drivers_for_summary[
            filtered_drivers_for_summary["department"] == selected_department
        ]
    if selected_region != "All":
        filtered_drivers_for_summary = filtered_drivers_for_summary[
            filtered_drivers_for_summary["region"] == selected_region
        ]
    if selected_product_line != "All":
        filtered_drivers_for_summary = filtered_drivers_for_summary[
            filtered_drivers_for_summary["product_line"] == selected_product_line
        ]
    if selected_variance_class != "All":
        filtered_drivers_for_summary = filtered_drivers_for_summary[
            filtered_drivers_for_summary["variance_classification"] == selected_variance_class
        ]
    filtered_top_variance_drivers = get_top_variance_drivers_list(filtered, top_n=5)

    executive_summary = generate_executive_summary(filtered, filtered_drivers_for_summary.head(5))
    strategic_insights = generate_strategic_insights(
        filtered,
        filtered_drivers_for_summary.head(5),
        filtered_anomalies,
    )
    strategic_insights_text = format_strategic_insights(strategic_insights)
    full_report_text = generate_full_fpa_report(
        validation_status=validation_status,
        system_validation_text=system_validation_text,
        budget_mapping=budget_column_mapping,
        actual_mapping=actual_column_mapping,
        executive_summary=executive_summary,
        top_drivers_df=filtered_top_variance_drivers,
        strategic_insights_text=strategic_insights_text,
    )
    filter_summary = build_filter_summary(
        {
            "Periods": selected_periods if len(selected_periods) != len(periods) else ["All"],
            "Department": selected_department,
            "Region": selected_region,
            "Product Line": selected_product_line,
            "Classification": selected_variance_class,
        }
    )
    filtered_driver_candidates = filtered_drivers_for_summary.copy()
    driver_commentary = generate_driver_commentary(filtered_driver_candidates.head(5))
    overview_tab, drivers_tab, quality_tab, report_tab, ask_tab, explain_tab, review_tab, downloads_tab = st.tabs(
        ["Overview", "Drivers", "Data Health", "Full Report", "Ask Data", "Explainable", "AI Review", "Downloads"]
    )

    with overview_tab:
        render_section_intro(
            "Executive Summary",
            "Management-ready headline view for the current filter set.",
            kicker="Narrative",
        )
        st.write(executive_summary)

        render_section_intro(
            "Strategic Insights",
            "High-level reasoning on performance quality, concentration, and operating signals.",
            kicker="Interpretation",
        )
        for section, items in strategic_insights.items():
            st.markdown(f"**{section}**")
            for item in items:
                st.write(f"- {item}")

        overview_chart_col1, overview_chart_col2 = st.columns(2)
        with overview_chart_col1:
            render_section_intro(
                "Monthly Trend",
                "Track budget, actual, and variance movement over time.",
                kicker="Trend",
            )
            trend_chart = (
                summary.assign(period=summary["date"].dt.strftime("%Y-%m"))
                .set_index("period")[["budget", "actual", "variance_abs"]]
            )
            st.line_chart(trend_chart)

        with overview_chart_col2:
            render_section_intro(
                "Variance by Account",
                "Spot which accounts are driving the current movement fastest.",
                kicker="Mix",
            )
            account_summary = summarize_by_dimension(filtered, ["account"]).sort_values("variance_abs")
            st.bar_chart(account_summary.set_index("account")[["variance_abs"]])

        render_section_intro(
            "Top 5 Drivers",
            "Fast scan of the highest-impact variance contributors in the current view.",
            kicker="Priorities",
        )
        st.dataframe(filtered_top_variance_drivers, use_container_width=True)

    with drivers_tab:
        render_section_intro(
            "Driver Analysis",
            "Detailed view of the most material drivers with classification and score.",
            kicker="Deep Dive",
        )
        st.dataframe(
            filtered_driver_candidates.head(10)[
                [
                    "department",
                    "region",
                    "product_line",
                    "account",
                    "variance_abs",
                    "variance_pct",
                    "variance_classification",
                    "variance_explanation",
                    "driver_score",
                ]
            ],
            use_container_width=True,
        )
        st.write(driver_commentary)

        render_section_intro(
            "Detailed Drill-Down",
            "Slice the reconciled view by the dimensions most useful for investigation.",
            kicker="Exploration",
        )
        drilldown_dimensions = st.multiselect(
            "Choose dimensions",
            ["department", "region", "product_line", "account"],
            default=["department", "region", "product_line"],
        )
        if drilldown_dimensions:
            drilldown_df = summarize_by_dimension(filtered, drilldown_dimensions).sort_values(
                "variance_abs",
                key=lambda series: series.abs(),
                ascending=False,
            )
            st.dataframe(drilldown_df, use_container_width=True)
        else:
            drilldown_df = filtered[
                [
                    "date",
                    "department",
                    "region",
                    "product_line",
                    "account",
                    "budget",
                    "actual",
                    "variance_abs",
                    "variance_pct",
                    "variance_classification",
                    "variance_explanation",
                ]
            ].copy()
            st.info("Select at least one dimension to view drill-down results.")

        render_section_intro(
            "Variance Classification Detail",
            "Line-level view of what is favorable versus unfavorable.",
            kicker="Classification",
        )
        st.dataframe(
            filtered[
                [
                    "date",
                    "department",
                    "region",
                    "product_line",
                    "account",
                    "variance_abs",
                    "variance_classification",
                    "variance_explanation",
                ]
            ].sort_values("variance_abs", key=lambda series: series.abs(), ascending=False),
            use_container_width=True,
        )

        if ai_api_key:
            render_section_intro(
                "AI Assistant",
                "Ask fintechIQ Copilot follow-up questions about the current FP&A slice.",
                kicker="Copilot",
            )
            ai_context = build_ai_context(
                executive_summary=executive_summary,
                strategic_insights_text=strategic_insights_text,
                driver_commentary=driver_commentary,
                system_validation_text=system_validation_text,
                top_drivers_df=filtered_driver_candidates.head(10),
                filtered_variance_df=filtered,
            )

            for message in st.session_state["ai_messages"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            user_question = st.chat_input("Ask the AI copilot about this FP&A view")
            if user_question:
                st.session_state["ai_messages"].append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.write(user_question)
                with st.chat_message("assistant"):
                    with st.spinner("Generating FP&A guidance..."):
                        try:
                            ai_response = ask_ai_assistant(
                                api_key=ai_api_key,
                                model=ai_model,
                                question=user_question,
                                context=ai_context,
                                chat_history=st.session_state["ai_messages"][:-1],
                            )
                        except Exception as exc:
                            ai_response = f"AI assistant error: {exc}"
                        st.write(ai_response)
                st.session_state["ai_messages"].append({"role": "assistant", "content": ai_response})
    with quality_tab:
        render_section_intro(
            "Data Quality",
            "Review record quality, anomalies, and preprocessing impact before sharing conclusions.",
            kicker="Controls",
        )
        dq_col1, dq_col2 = st.columns([1, 2])
        dq_col1.dataframe(quality_summary, use_container_width=True)
        dq_col2.dataframe(filtered_anomalies.head(25), use_container_width=True)

        render_section_intro(
            "System Validation",
            "Validation signals tell you whether the current output is fully trusted or summary-level only.",
            kicker="Validation",
        )
        if validation_status == "Error":
            render_violet_notice("Validation issues were detected. Review the error report before relying on the final FP&A conclusions.", tone="strong")
        elif validation_status == "Warning":
            render_violet_notice("Validation passed with summary-level limitations. You can use the report, but some drill-down dimensions were not provided in the source data.")
        else:
            render_violet_notice("Validation passed with no material data issues detected.")
        st.text(system_validation_text)
        st.dataframe(system_validation_df, use_container_width=True)

    with report_tab:
        render_section_intro(
            "Full FP&A Report",
            "Consolidated narrative output for the current filtered view.",
            kicker="Management Pack Text",
        )
        st.caption("Downloadable PDF and HTML versions are available below this report.")
        st.text(full_report_text)

    with ask_tab:
        render_section_intro(
            "Ask Data",
            "Ask a direct question about the active uploaded dataset and get a data-bound answer with explanation and supporting values.",
            kicker="Q&A",
        )
        sample_questions = [
            "What is the overall variance?",
            "Which region has the highest variance?",
            "How did variance trend over time?",
            "Why is APAC ahead of plan?",
            "What is the biggest anomaly in the data?",
        ]
        st.caption("Example questions: " + " | ".join(sample_questions))
        data_question = st.text_input(
            "Question",
            placeholder="Ask about trend, comparison, summary, anomaly, or why.",
        )
        if data_question.strip():
            data_summary_for_qa = "\n\n".join(
                [
                    executive_summary,
                    strategic_insights_text,
                    filtered_top_variance_drivers.to_string(index=False),
                ]
            )
            qa_result = answer_question_from_fpa_data(
                question=data_question,
                data_summary=data_summary_for_qa,
                data=filtered,
            )
            st.markdown(qa_result["markdown"])
            st.download_button(
                "Download Answer",
                data=qa_result["markdown"].encode("utf-8"),
                file_name="data_question_answer.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            render_violet_notice("Type a question about the current filtered dataset to get a direct answer.")

    with explain_tab:
        render_section_intro(
            "Explainable Insights",
            "Turn existing analysis into transparent insight, evidence, and reasoning blocks.",
            kicker="Explainability",
        )
        default_insights = strategic_insights_text
        default_source_data = "\n\n".join(
            [
                "Executive Summary:",
                executive_summary,
                "",
                "Top Drivers:",
                filtered_top_variance_drivers.to_string(index=False),
                "",
                "Validation Notes:",
                system_validation_text,
            ]
        )
        original_insights_input = st.text_area(
            "Original Insights",
            value=default_insights,
            height=240,
            help="Paste the original analysis or insight bullets that you want to make more transparent.",
        )
        source_data_input = st.text_area(
            "Source Data",
            value=default_source_data,
            height=240,
            help="Paste the source data summary, supporting metrics, or validated narrative used to justify the insights.",
        )
        if original_insights_input.strip() and source_data_input.strip():
            explainable = generate_explainable_insights(
                insights=original_insights_input,
                source_data=source_data_input,
            )
            render_confidence_summary_cards(explainable["items"])
            weak_insights = explainable.get("weak_insights", [])
            if weak_insights:
                render_violet_notice(
                    f"{len(weak_insights)} insight(s) have weak evidence support. Review those conclusions before reusing them in reporting.",
                    tone="strong",
                )
                for weak_insight in weak_insights:
                    st.write(f"- {weak_insight}")
            else:
                render_violet_notice("All explainable insights found at least moderate evidence support in the supplied source data.")
            st.markdown(explainable["markdown"])
            st.download_button(
                "Download Explainable Insights",
                data=explainable["markdown"].encode("utf-8"),
                file_name="explainable_insights.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            render_violet_notice("Paste both the original insights and the supporting source data to build explainable insights.")

    with review_tab:
        render_section_intro(
            "AI Report Evaluator",
            "Paste an AI-generated analysis and compare it against a source summary using a strict professional rubric.",
            kicker="Quality Control",
        )
        default_source_summary = "\n\n".join(
            [
                "Executive Summary:",
                executive_summary,
                "",
                "Strategic Insights:",
                strategic_insights_text,
                "",
                "Top Drivers:",
                filtered_top_variance_drivers.to_string(index=False),
            ]
        )
        source_summary_input = st.text_area(
            "Source Data Summary",
            value=default_source_summary,
            height=260,
            help="Use the actual source summary, data excerpt, or validated analysis that the AI output should be judged against.",
        )
        ai_output_input = st.text_area(
            "AI Output to Evaluate",
            height=260,
            help="Paste the AI-generated report or narrative you want to evaluate.",
        )
        if ai_output_input.strip() and source_summary_input.strip():
            evaluation = evaluate_ai_generated_report(
                ai_output=ai_output_input,
                data_summary=source_summary_input,
            )
            render_quality_scorecard(evaluation)
            st.markdown(evaluation["report_markdown"])
            st.download_button(
                "Download Evaluation",
                data=evaluation["report_markdown"].encode("utf-8"),
                file_name="ai_report_evaluation.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            render_violet_notice("Paste both the source summary and the AI output to generate the evaluation report.")

    with downloads_tab:
        render_section_intro(
            "Downloads",
            "Export filtered data, management packs, and narrative outputs for leadership reporting.",
            kicker="Export",
        )
    filter_label = "_".join(
        [
            "variance_analysis",
            selected_department.lower().replace("&", "and").replace(" ", "_"),
            selected_region.lower().replace(" ", "_"),
            selected_product_line.lower().replace(" ", "_"),
        ]
    )
    export_files = {
        "executive_summary.txt": executive_summary.encode("utf-8"),
        "driver_commentary.txt": driver_commentary.encode("utf-8"),
        "strategic_insights.txt": strategic_insights_text.encode("utf-8"),
        "system_validation_report.txt": system_validation_text.encode("utf-8"),
        "full_fpa_report.txt": full_report_text.encode("utf-8"),
        "filtered_variance.csv": dataframe_to_csv_bytes(filtered),
        "filtered_drivers.csv": dataframe_to_csv_bytes(filtered_driver_candidates),
        "top_5_variance_drivers.csv": dataframe_to_csv_bytes(filtered_top_variance_drivers),
        "drilldown.csv": dataframe_to_csv_bytes(drilldown_df),
        "quality_summary.csv": dataframe_to_csv_bytes(quality_summary),
        "system_validation_report.csv": dataframe_to_csv_bytes(system_validation_df),
        "filtered_anomalies.csv": dataframe_to_csv_bytes(filtered_anomalies),
    }
    export_bundle = build_export_bundle(export_files)
    management_pack = build_management_pack(
        variance_df=filtered,
        drivers_df=filtered_driver_candidates,
        executive_summary=executive_summary,
        driver_commentary=driver_commentary,
        strategic_insights=strategic_insights_text,
        system_validation_text=system_validation_text,
        full_report_text=full_report_text,
        filter_summary=filter_summary,
        variance_csv=export_files["filtered_variance.csv"],
        drivers_csv=export_files["filtered_drivers.csv"],
        drilldown_csv=export_files["drilldown.csv"],
        anomalies_csv=export_files["filtered_anomalies.csv"],
        quality_csv=export_files["quality_summary.csv"],
    )

    with report_tab:
        report_download_col1, report_download_col2 = st.columns(2)
        report_download_col1.download_button(
            "Download PDF Report",
            data=management_pack["pdf"],
            file_name=f"{filter_label}_management_pack.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        report_download_col2.download_button(
            "Download HTML Report",
            data=management_pack["html"],
            file_name=f"{filter_label}_management_pack.html",
            mime="text/html",
            use_container_width=True,
        )

    with downloads_tab:
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        dl_col1.download_button(
            "Filtered variance CSV",
            data=export_files["filtered_variance.csv"],
            file_name=f"{filter_label}_filtered_variance.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl_col2.download_button(
            "Filtered drivers CSV",
            data=export_files["filtered_drivers.csv"],
            file_name=f"{filter_label}_filtered_drivers.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl_col3.download_button(
            "Drill-down CSV",
            data=export_files["drilldown.csv"],
            file_name=f"{filter_label}_drilldown.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl_col4.download_button(
            "Top 5 drivers CSV",
            data=export_files["top_5_variance_drivers.csv"],
            file_name=f"{filter_label}_top_5_variance_drivers.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download ZIP pack",
            data=export_bundle,
            file_name=f"{filter_label}_export_pack.zip",
            mime="application/zip",
            use_container_width=True,
        )

        mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
        mgmt_col1.download_button(
            "Management pack HTML",
            data=management_pack["html"],
            file_name=f"{filter_label}_management_pack.html",
            mime="text/html",
            use_container_width=True,
        )
        mgmt_col2.download_button(
            "Management pack PDF",
            data=management_pack["pdf"],
            file_name=f"{filter_label}_management_pack.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        mgmt_col3.download_button(
            "Management pack ZIP",
            data=management_pack["zip"],
            file_name=f"{filter_label}_management_pack.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
