from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.features import (
    CONTRADICTION_SIGNALS,
    FRUSTRATION_SIGNALS,
    REFUSAL_SIGNALS,
    SUCCESS_SIGNALS,
    contradiction_score,
    matched_signals,
    repeat_score,
    structured_features,
    unsupported_claim_score,
    unsupported_claim_signals,
)


LABELS = (
    "task_not_completed",
    "repetition_loop",
    "hallucination",
    "user_frustration",
)

ML_LABELS = (
    "task_not_completed",
    "user_frustration",
)

DEFAULT_THRESHOLDS = {
    "task_not_completed": 0.30,
    "user_frustration": 0.50,
}


def create_vectorizer():
    """Create the word-level TF-IDF converter used by CallGuard."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=5000,
    )


def build_feature_matrix(vectorizer, transcripts, fit=False):
    """Combine learned TF-IDF features with transparent numeric features."""
    text_values = pd.Series(transcripts).fillna("").astype(str)
    if fit:
        text_matrix = vectorizer.fit_transform(text_values)
    else:
        text_matrix = vectorizer.transform(text_values)

    numeric_matrix = csr_matrix(structured_features(text_values))
    return hstack([text_matrix, numeric_matrix]).tocsr()


def train_bundle(training_data):
    """Train the deployable hybrid CallGuard model bundle."""
    vectorizer = create_vectorizer()
    matrix = build_feature_matrix(
        vectorizer,
        training_data["transcript"],
        fit=True,
    )

    models = {}
    for label in ML_LABELS:
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        model.fit(matrix, training_data[label])
        models[label] = model

    return {
        "version": 1,
        "vectorizer": vectorizer,
        "models": models,
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "labels": LABELS,
    }


def predict_dataframe(bundle, transcripts):
    """Predict all four CallGuard labels for one or more transcripts."""
    text_values = pd.Series(transcripts).fillna("").astype(str).reset_index(drop=True)
    matrix = build_feature_matrix(bundle["vectorizer"], text_values)

    results = pd.DataFrame({"transcript": text_values})

    for label in ML_LABELS:
        probabilities = bundle["models"][label].predict_proba(matrix)[:, 1]
        threshold = bundle["thresholds"][label]
        results[f"{label}_score"] = probabilities
        results[label] = (probabilities >= threshold).astype(int)

    repetition_scores = np.asarray(
        [repeat_score(text) for text in text_values],
        dtype=int,
    )
    contradiction_scores = np.asarray(
        [contradiction_score(text) for text in text_values],
        dtype=int,
    )
    unsupported_claim_scores = np.asarray(
        [unsupported_claim_score(text) for text in text_values],
        dtype=int,
    )
    hallucination_evidence_scores = (
        contradiction_scores + unsupported_claim_scores
    )

    results["repetition_score"] = repetition_scores
    results["repetition_loop"] = (repetition_scores > 0).astype(int)
    results["hallucination_evidence_score"] = hallucination_evidence_scores
    results["hallucination"] = (
        hallucination_evidence_scores > 0
    ).astype(int)

    ordered_columns = ["transcript"]
    for label in LABELS:
        ordered_columns.append(label)
        if label in ML_LABELS:
            ordered_columns.append(f"{label}_score")
        elif label == "repetition_loop":
            ordered_columns.append("repetition_score")
        else:
            ordered_columns.append("hallucination_evidence_score")

    return results[ordered_columns]


def evidence_for_transcript(transcript):
    """Return human-readable evidence used by the hybrid detectors."""
    return {
        "repetition_score": repeat_score(transcript),
        "frustration_signals": matched_signals(
            transcript,
            FRUSTRATION_SIGNALS,
        ),
        "refusal_signals": matched_signals(
            transcript,
            REFUSAL_SIGNALS,
        ),
        "contradiction_signals": matched_signals(
            transcript,
            CONTRADICTION_SIGNALS,
        ),
        "unsupported_claim_signals": unsupported_claim_signals(transcript),
        "success_signals": matched_signals(
            transcript,
            SUCCESS_SIGNALS,
        ),
    }


def save_bundle(bundle, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)


def load_bundle(path):
    return joblib.load(Path(path))
