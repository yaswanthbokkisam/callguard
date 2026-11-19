import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Load the synthetic call data.
df = pd.read_csv("data/synthetic_calls.csv")


# X contains the original transcript text.
X_text = df["transcript"]


# y contains the correct frustration answers.
y = df["user_frustration"]


# Split the raw text and answers into training and testing groups.
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Create the text-to-number converter.
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
)


# Learn the vocabulary from training text and convert it to numbers.
X_train = vectorizer.fit_transform(X_train_text)


# Convert test text using the vocabulary learned from training data.
X_test = vectorizer.transform(X_test_text)


print("Training text count:", len(X_train_text))
print("Testing text count:", len(X_test_text))

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nVocabulary size:", len(vectorizer.get_feature_names_out()))

print("\nFirst 20 learned features:")
print(vectorizer.get_feature_names_out()[:20])

print("\nTraining label counts:")
print(y_train.value_counts())

print("\nTesting label counts:")
print(y_test.value_counts())


# Create the Logistic Regression classification model.
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)


# Train the model using TF-IDF features and correct labels.
model.fit(X_train, y_train)


# Ask the trained model to predict the test calls.
predictions = model.predict(X_test)


# Compare the predictions with the correct test answers.
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\nFrustration model evaluation:")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 score:  {f1:.2%}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))


# Load a separate dataset containing more realistic conversations.
realistic_df = pd.read_csv("data/realistic_test.csv")


# Convert realistic transcripts using the vocabulary learned from training data.
# We use transform(), not fit_transform(), to avoid test-data leakage.
X_realistic = vectorizer.transform(realistic_df["transcript"])


# Get the correct realistic answers.
y_realistic = realistic_df["user_frustration"]


# Predict frustration for the realistic calls.
realistic_predictions = model.predict(X_realistic)


# Calculate realistic evaluation measurements.
realistic_accuracy = accuracy_score(
    y_realistic,
    realistic_predictions,
)
realistic_precision = precision_score(
    y_realistic,
    realistic_predictions,
)
realistic_recall = recall_score(
    y_realistic,
    realistic_predictions,
)
realistic_f1 = f1_score(
    y_realistic,
    realistic_predictions,
)


print("\nRealistic frustration evaluation:")
print(f"Accuracy:  {realistic_accuracy:.2%}")
print(f"Precision: {realistic_precision:.2%}")
print(f"Recall:    {realistic_recall:.2%}")
print(f"F1 score:  {realistic_f1:.2%}")

print("\nRealistic confusion matrix:")
print(confusion_matrix(y_realistic, realistic_predictions))


# Get the probability of class 1: user frustration.
realistic_probabilities = model.predict_proba(X_realistic)[:, 1]


# Lower the decision threshold because missing a failure is expensive.
selected_threshold = 0.45


# Convert probabilities into 0/1 predictions using the selected threshold.
threshold_predictions = (
    realistic_probabilities >= selected_threshold
).astype(int)


# Evaluate the selected threshold.
threshold_accuracy = accuracy_score(
    y_realistic,
    threshold_predictions,
)
threshold_precision = precision_score(
    y_realistic,
    threshold_predictions,
)
threshold_recall = recall_score(
    y_realistic,
    threshold_predictions,
)
threshold_f1 = f1_score(
    y_realistic,
    threshold_predictions,
)


print(
    f"\nRealistic evaluation at threshold "
    f"{selected_threshold}:"
)
print(f"Accuracy:  {threshold_accuracy:.2%}")
print(f"Precision: {threshold_precision:.2%}")
print(f"Recall:    {threshold_recall:.2%}")
print(f"F1 score:  {threshold_f1:.2%}")

print("\nThreshold confusion matrix:")
print(confusion_matrix(y_realistic, threshold_predictions))


# Find real frustration calls that the model missed.
false_negatives = realistic_df.loc[
    (y_realistic == 1) & (realistic_predictions == 0)
]


# Find normal calls that the model incorrectly flagged.
false_positives = realistic_df.loc[
    (y_realistic == 0) & (realistic_predictions == 1)
]


print("\nFalse negatives: missed frustration calls")
print("Count:", len(false_negatives))

for _, row in false_negatives.iterrows():
    print(f"\nCall ID: {row['call_id']}")
    print(row["transcript"])


print("\nFalse positives: incorrect warnings")
print("Count:", len(false_positives))

for _, row in false_positives.iterrows():
    print(f"\nCall ID: {row['call_id']}")
    print(row["transcript"])


# Match every TF-IDF feature with its learned Logistic Regression weight.
feature_weights = pd.DataFrame(
    {
        "feature": vectorizer.get_feature_names_out(),
        "weight": model.coef_[0],
    }
)


# Large positive weights push predictions toward frustration.
top_frustration_features = feature_weights.sort_values(
    by="weight",
    ascending=False,
).head(15)


# Large negative weights push predictions toward a normal call.
top_normal_features = feature_weights.sort_values(
    by="weight",
    ascending=True,
).head(15)


print("\nTop features indicating frustration:")
print(top_frustration_features.to_string(index=False))

print("\nTop features indicating a normal call:")
print(top_normal_features.to_string(index=False))
