from pathlib import Path
import json
import re
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import streamlit as st
from sentinelaid.workflow import run

def create_pdf_report(content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        safe_line = (line.replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
        if line.startswith("# "):
            story.append(Paragraph(safe_line[2:], styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(safe_line[3:], styles["Heading2"]))
        else:
            story.append(Paragraph(safe_line, styles["BodyText"]))
            story.append(Spacer(1, 5))

    doc.build(story)
    return buffer.getvalue()


st.set_page_config(
    page_title="SentinelAid AI",
    page_icon="🕊️",
    layout="wide"
)

DEMO = Path(__file__).parent / "data" / "fictional_scenario.txt"

AGENTS = [
    "News Intelligence Agent",
    "Crisis Detection Agent",
    "Civilian Risk Assessment Agent",
    "Evidence Verification Agent",
    "Humanitarian Report Agent",
]

st.title("🕊️ SentinelAid AI")
st.caption(
    "Multi-Agent Humanitarian Crisis Intelligence "
    "& Civilian Protection Research System"
)

st.warning(
    "Research support only. Not operational intelligence, "
    "targeting, combat guidance, legal adjudication or emergency advice. "
    "All AI-generated findings require human verification."
)

if "sources" not in st.session_state:
    st.session_state.sources = ""

if "question" not in st.session_state:
    st.session_state.question = (
        "What humanitarian needs are reported, "
        "how strong is the supporting evidence, "
        "and what information is missing?"
    )

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("📄 Load fictional demonstration"):
    st.session_state.sources = DEMO.read_text(encoding="utf-8")
    st.session_state.results = None
    st.rerun()

st.divider()

st.subheader("Research workspace")

question = st.text_area(
    "Research question",
    key="question",
    height=90
)

sources = st.text_area(
    "Public reports and source material",
    key="sources",
    height=260,
    placeholder="Paste reports with source labels, such as [S1: publisher, date]"
)

source_count = len(set(re.findall(
    r"\[S\d+\s*:",
    sources,
    flags=re.IGNORECASE
)))

col1, col2, col3 = st.columns(3)

col1.metric("Source labels detected", source_count)
col2.metric("AI agents", "5")
col3.metric("Analysis mode", "Local Ollama")

st.caption(
    "Source-label count is a simple text count, not a measure "
    "of source independence or evidence quality."
)

if st.button(
    "🚀 Run humanitarian assessment",
    type="primary",
    use_container_width=True
):
    if not sources.strip() or not question.strip():
        st.error("Please provide both a research question and source material.")
    else:
        status = st.empty()
        progress_bar = st.progress(0)

        try:
            def update_progress(stage, model):
                status.info(f"Running: {stage} | Model: {model}")
                match = re.match(r"(\d+)/5", stage)
                if match:
                    progress_bar.progress(int(match.group(1)) * 20)

            with st.spinner("Five local AI agents are analysing the material..."):
                results = run(
                    sources,
                    question,
                    progress=update_progress
                )

            st.session_state.results = results
            progress_bar.progress(100)
            status.success(
                "Analysis completed. Human verification is required."
            )

        except Exception as error:
            status.empty()
            st.error(f"Analysis failed: {error}")

results = st.session_state.results

if results:
    st.divider()
    st.header("Humanitarian intelligence dashboard")

    st.success("Five-agent analysis completed")

    st.subheader("Agent execution")

    for agent in AGENTS:
        if agent in results:
            st.write(f"✅ {agent}")
        else:
            st.write(f"⚠️ {agent} — output unavailable")

    st.divider()

    tab_names = [
        "Overview",
        "Crisis Detection",
        "Civilian Risk",
        "Evidence Verification",
        "Final Report",
        "All Agents",
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("Humanitarian situation overview")
        st.markdown(results.get(
            "News Intelligence Agent",
            "No news intelligence output available."
        ))

        st.info(
            "This overview contains AI analysis of supplied material. "
            "It is not independently verified news intelligence."
        )

    with tabs[1]:
        st.subheader("Detected humanitarian concerns")
        st.markdown(results.get(
            "Crisis Detection Agent",
            "No crisis detection output available."
        ))

    with tabs[2]:
        st.subheader("Civilian humanitarian risk assessment")
        st.markdown(results.get(
            "Civilian Risk Assessment Agent",
            "No risk assessment output available."
        ))

        st.caption(
            "AI classifications are preliminary research assessments, "
            "not validated probabilities or definitive safety findings."
        )

    with tabs[3]:
        st.subheader("Evidence verification and conflicting information")
        st.markdown(results.get(
            "Evidence Verification Agent",
            "No evidence verification output available."
        ))

        st.warning(
            "The agent compares only the supplied material. "
            "It does not independently verify external sources."
        )

    with tabs[4]:
        st.subheader("Humanitarian situation report")
        final_report = results.get(
            "Humanitarian Report Agent",
            "No final report available."
        )
        st.markdown(final_report)

    with tabs[5]:
        for agent, output in results.items():
            with st.expander(agent):
                st.markdown(output)

    st.divider()
    st.subheader("Export research results")

    combined_report = "\n\n".join(
        f"# {agent}\n\n{output}"
        for agent, output in results.items()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "⬇️ Download Markdown report",
            data=combined_report,
            file_name="sentinelaid_humanitarian_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        st.download_button(
            "⬇️ Download JSON results",
            data=json.dumps(results, indent=2, ensure_ascii=False),
            file_name="sentinelaid_results.json",
            mime="application/json",
            use_container_width=True
        )

    with col3:
        st.download_button(
            "⬇️ Download PDF report",
            data=create_pdf_report(combined_report),
            file_name="sentinelaid_humanitarian_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.divider()

st.caption(
    "SentinelAid AI | Academic humanitarian research prototype | "
    "Synthetic demonstrations only unless explicitly supplied otherwise."
)
