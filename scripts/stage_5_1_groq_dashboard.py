
from pathlib import Path
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

TEST_SUMMARY_PATH = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_7_final_test"
    / "stage_4_7_test_summary.json"
)

AUDIT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_8_final_model_audit"
    / "FINAL_PROJECT_SUMMARY.json"
)

MAINTENANCE_DIR = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_3_maintenance_intelligence"
)


# --------------------------------------------------
# 2. SAFE FILE LOADING
# --------------------------------------------------

def load_json(path):
    """Load a JSON file without crashing the dashboard."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return {"_loading_error": str(exc)}


def discover_maintenance_files():
    """Discover existing maintenance outputs without assuming filenames."""
    files = []

    if MAINTENANCE_DIR.exists():
        for pattern in ("*.json", "*.csv"):
            files.extend(MAINTENANCE_DIR.glob(pattern))

    return sorted(files)


def load_maintenance_evidence():
    """Read small, useful samples of discovered maintenance outputs."""
    evidence = {}

    for path in discover_maintenance_files():
        try:
            if path.suffix.lower() == ".json":
                data = load_json(path)
                if data is not None:
                    evidence[path.name] = data

            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path, encoding="utf-8-sig")
                evidence[path.name] = {
                    "columns": df.columns.tolist(),
                    "row_count": int(len(df)),
                    "sample_rows": df.head(10).to_dict(
                        orient="records"
                    ),
                }

        except Exception as exc:
            evidence[path.name] = {
                "loading_error": str(exc)
            }

    return evidence


def flatten_dict(data, prefix=""):
    """Convert nested JSON values into readable metric labels."""
    rows = []

    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                rows.extend(flatten_dict(value, name))
            elif isinstance(value, list):
                rows.append((name, json.dumps(value)[:300]))
            else:
                rows.append((name, value))

    elif isinstance(data, list):
        for index, value in enumerate(data[:30]):
            name = f"{prefix}[{index}]"
            if isinstance(value, (dict, list)):
                rows.extend(flatten_dict(value, name))
            else:
                rows.append((name, value))

    else:
        rows.append((prefix or "value", data))

    return rows


def format_context(test_summary, audit_summary, maintenance):
    """Build bounded evidence for the language model."""
    context = {
        "final_test_summary": test_summary,
        "final_model_audit": audit_summary,
        "maintenance_outputs": maintenance,
    }

    # Keep the prompt manageable.
    return json.dumps(context, ensure_ascii=True, default=str)[:24000]


# --------------------------------------------------
# 3. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="BeltCrack9kd Damage Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.title("BeltCrack9kd Damage Intelligence")
st.caption(
    "Segmentation evaluation, maintenance evidence, and "
    "Groq-powered explanations"
)

test_summary = load_json(TEST_SUMMARY_PATH)
audit_summary = load_json(AUDIT_SUMMARY_PATH)
maintenance_evidence = load_maintenance_evidence()

if test_summary is None:
    st.error(
        "Final test summary not found. Expected file:\n\n"
        f"{TEST_SUMMARY_PATH}"
    )

if audit_summary is None:
    st.warning(
        "Final audit summary was not found. "
        "The dashboard can still show any available test results."
    )


# --------------------------------------------------
# 4. MODEL EVALUATION RESULTS
# --------------------------------------------------

st.header("1. Final Segmentation Evaluation")

if isinstance(test_summary, dict):
    flat_metrics = flatten_dict(test_summary)

    # Show likely headline metrics if their keys exist.
    headline_terms = (
        "dice",
        "precision",
        "recall",
        "threshold",
        "coverage",
        "epoch",
    )

    headline = [
        (key, value)
        for key, value in flat_metrics
        if any(term in key.lower() for term in headline_terms)
        and not isinstance(value, (dict, list))
    ]

    if headline:
        cols = st.columns(3)

        for index, (key, value) in enumerate(headline[:12]):
            with cols[index % 3]:
                st.metric(
                    label=key.split(".")[-1].replace("_", " ").title(),
                    value=str(value),
                )
    else:
        st.info(
            "No standard metric keys were detected automatically. "
            "Open the raw summary below to inspect its recorded fields."
        )

    with st.expander("View complete final test summary"):
        st.json(test_summary)

else:
    st.info("Final test results are not currently available.")


# --------------------------------------------------
# 5. FINAL AUDIT
# --------------------------------------------------

st.header("2. Final Model Audit")

if isinstance(audit_summary, dict):
    with st.expander("View final audit summary", expanded=False):
        st.json(audit_summary)
elif audit_summary is not None:
    st.write(audit_summary)
else:
    st.info("No final audit JSON was found at the expected path.")


# --------------------------------------------------
# 6. MAINTENANCE OUTPUTS
# --------------------------------------------------

st.header("3. Maintenance Intelligence Evidence")

maintenance_files = discover_maintenance_files()

if not maintenance_files:
    st.warning(
        "No JSON or CSV files were found directly inside:\n\n"
        f"{MAINTENANCE_DIR}\n\n"
        "This does not mean the maintenance stage failed. "
        "Check the actual output directory and filenames."
    )
else:
    st.write(f"Discovered {len(maintenance_files)} output file(s).")

    for path in maintenance_files:
        with st.expander(path.name):
            if path.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig")
                    st.write(f"Rows: {len(df)}")
                    st.dataframe(df.head(100), use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not read CSV: {exc}")

            elif path.suffix.lower() == ".json":
                data = load_json(path)
                st.json(data if data is not None else {})


# --------------------------------------------------
# 7. GROQ ASSISTANT
# --------------------------------------------------

st.header("4. Ask the Damage Intelligence Assistant")

st.write(
    "Ask about the recorded metrics, segmentation limitations, "
    "maintenance outputs, or appropriate next steps."
)

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: What do the final test metrics tell us about "
        "the reliability of this segmentation model?"
    ),
    height=100,
)

ask_button = st.button(
    "Analyze project evidence",
    type="primary",
)

if ask_button:
    if not question.strip():
        st.warning("Enter a question first.")

    elif test_summary is None and audit_summary is None:
        st.error(
            "No final test or audit summary is available. "
            "Check the file paths before asking the assistant."
        )

    elif not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY was not found. Add it to the project-root "
            ".env file. Do not paste the key into this dashboard."
        )

    else:
        evidence = format_context(
            test_summary,
            audit_summary,
            maintenance_evidence,
        )

        system_prompt = """
You are a careful machine-learning engineering assistant
for a conveyor-belt crack segmentation project.

Use only the supplied project evidence for claims about this project.
Do not invent metrics, files, experiments, dataset properties,
model capabilities, or completed stages.

Clearly distinguish:
1. Recorded results.
2. Interpretation of those results.
3. Suggestions for future work.

The segmentation pilot has known limitations and its measured
performance is limited. Do not call it production-ready or claim
reliable real-world physical damage prediction.

The existing feature-based future crack-area predictor is different
from the image segmentation model. Do not conflate them.

Groq provides explanations only; it does not retrain or validate
the project's models.

If the evidence does not answer a question, say so explicitly.
Give practical, concise answers. Do not expose secrets.
"""

        user_prompt = f"""
PROJECT EVIDENCE:
{evidence}

USER QUESTION:
{question.strip()}
"""

        try:
            with st.spinner("Analyzing the recorded project evidence..."):
                client = Groq(
                    api_key=os.environ["GROQ_API_KEY"]
                )

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=0.2,
                    max_completion_tokens=1200,
                )

                answer = response.choices[0].message.content

            st.subheader("Assistant analysis")
            st.markdown(answer or "The model returned an empty response.")

            st.caption(
                f"Generated using Groq model: {MODEL_NAME}. "
                "This is a language-model explanation, not an "
                "independent model evaluation."
            )

        except Exception as exc:
            st.error(
                "The Groq request failed. Check your API key, model "
                "availability, internet connection, and account limits."
            )
            st.caption(f"Technical error: {type(exc).__name__}")


# --------------------------------------------------
# 8. PROJECT LIMITATIONS
# --------------------------------------------------

st.divider()
st.header("5. Important Limitations")

st.markdown(
    """
- The segmentation pilot has limited measured performance and is
  **not ready for production use**.
- The final test metrics describe the tested dataset and evaluation
  procedure; they do not establish performance on a real conveyor.
- The feature-based future crack-area predictor and the image
  segmentation model are separate components.
- A Groq-generated explanation is not ground truth and does not
  improve model accuracy.
- Real-world deployment would require additional validation on
  representative conveyor footage, appropriate temporal evaluation,
  and monitoring under operating conditions.
"""
)

st.caption("BeltCrack9kd Damage Intelligence | Stage 5.1")
