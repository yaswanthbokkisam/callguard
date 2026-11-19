import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.modeling import evidence_for_transcript, load_bundle, predict_dataframe


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "callguard.joblib"
REPORT_FILE = ROOT / "reports" / "validation_metrics.json"

LABEL_DETAILS = {
    "task_not_completed": {
        "title": "Task not completed",
        "description": "The requested task appears unresolved.",
    },
    "repetition_loop": {
        "title": "Repetition loop",
        "description": "The agent substantially repeated an earlier request.",
    },
    "hallucination": {
        "title": "Unsupported claim",
        "description": "A later response contradicts a claim made in the call.",
    },
    "user_frustration": {
        "title": "User frustration",
        "description": "The conversation contains frustration signals.",
    },
}

EXAMPLE_TRANSCRIPT = (
    "Agent: I'd like to book a table for four at 7:30. "
    "Restaurant: We are completely full tonight. "
    "Agent: Perfect, I have confirmed the table for 7:30. "
    "Restaurant: No, nothing was booked."
)


@st.cache_resource
def get_model():
    return load_bundle(MODEL_FILE)


def format_signal_list(signals):
    if not signals:
        return "None found"
    return ", ".join(f"`{signal}`" for signal in signals)


def show_prediction_cards(row):
    columns = st.columns(2)
    for index, label in enumerate(LABEL_DETAILS):
        details = LABEL_DETAILS[label]
        flagged = bool(row[label])
        with columns[index % 2]:
            status = "Flagged" if flagged else "Clear"
            icon = "⚠️" if flagged else "✅"
            st.subheader(f"{icon} {details['title']}")
            st.write(f"**Status:** {status}")
            st.caption(details["description"])

            score_column = f"{label}_score"
            if score_column in row.index:
                st.progress(float(row[score_column]))
                st.caption(f"Model score: {row[score_column]:.1%}")
            elif label == "repetition_loop":
                st.caption(f"Repeated-turn count: {int(row['repetition_score'])}")
            else:
                st.caption(
                    "Unsupported-claim evidence count: "
                    f"{int(row['hallucination_evidence_score'])}"
                )


def show_evidence(transcript):
    evidence = evidence_for_transcript(transcript)
    with st.expander("Why did CallGuard make these decisions?"):
        st.write(f"**Repetition score:** {evidence['repetition_score']}")
        st.write(
            "**Frustration signals:** "
            + format_signal_list(evidence["frustration_signals"])
        )
        st.write(
            "**Refusal signals:** "
            + format_signal_list(evidence["refusal_signals"])
        )
        st.write(
            "**Contradiction signals:** "
            + format_signal_list(evidence["contradiction_signals"])
        )
        st.write(
            "**Unsupported Agent claims:** "
            + format_signal_list(evidence["unsupported_claim_signals"])
        )
        st.write(
            "**Success signals:** "
            + format_signal_list(evidence["success_signals"])
        )


def single_call_tab(bundle):
    st.write("Paste a phone-call transcript and analyze all four failure types.")
    transcript = st.text_area(
        "Call transcript",
        value=EXAMPLE_TRANSCRIPT,
        height=220,
        help="Use speaker labels such as 'Agent:' and 'Restaurant:'.",
    )

    if st.button("Analyze call", type="primary", use_container_width=True):
        if not transcript.strip():
            st.warning("Enter a transcript before analyzing it.")
            return

        prediction = predict_dataframe(bundle, [transcript]).iloc[0]
        st.divider()
        show_prediction_cards(prediction)
        show_evidence(transcript)


def batch_tab(bundle):
    st.write(
        "Upload a CSV containing a `transcript` column. "
        "CallGuard will analyze every row."
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"The CSV could not be read: {error}")
        return

    if "transcript" not in uploaded_data.columns:
        st.error("The uploaded CSV must contain a column named `transcript`.")
        return

    if uploaded_data.empty:
        st.warning("The uploaded CSV contains no rows.")
        return

    predictions = predict_dataframe(bundle, uploaded_data["transcript"])
    if "call_id" in uploaded_data.columns:
        predictions.insert(0, "call_id", uploaded_data["call_id"].values)

    st.success(f"Analyzed {len(predictions)} calls.")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download results",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="callguard_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def performance_tab():
    st.write("Validation results on 60 realistic, manually labeled calls.")
    if not REPORT_FILE.exists():
        st.info("Run `python -m src.train_models` to create the report.")
        return

    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    rows = []
    for label, metrics in report["metrics"].items():
        rows.append(
            {
                "Detector": LABEL_DETAILS[label]["title"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            }
        )

    metrics_frame = pd.DataFrame(rows).set_index("Detector")
    st.dataframe(
        metrics_frame.style.format("{:.1%}"),
        use_container_width=True,
    )

    st.warning(
        "This is an educational portfolio model trained mainly on synthetic "
        "text. Validate it on representative production calls before real use."
    )


def main():
    st.set_page_config(
        page_title="CallGuard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ CallGuard")
    st.subheader("Failure detection for voice-AI call transcripts")
    st.caption(
        "Hybrid NLP system: TF-IDF + Logistic Regression + transparent rules"
    )

    if not MODEL_FILE.exists():
        st.error(
            "The trained model is missing. Run `python -m src.train_models` "
            "from the project folder."
        )
        st.stop()

    bundle = get_model()
    single_tab, batch_upload_tab, model_tab = st.tabs(
        ["Single call", "Batch CSV", "Model performance"]
    )

    with single_tab:
        single_call_tab(bundle)
    with batch_upload_tab:
        batch_tab(bundle)
    with model_tab:
        performance_tab()


if __name__ == "__main__":
    main()
