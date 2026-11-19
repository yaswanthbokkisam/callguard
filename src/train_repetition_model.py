import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from features import repeat_score


# Load the 500 call transcripts
df = pd.read_csv("data/synthetic_calls.csv")


# Convert every transcript into a numerical repeat score
df["repeat_score"] = df["transcript"].apply(repeat_score)


# X contains the input clue
X = df[["repeat_score"]]


# y contains the correct answer
y = df["repetition_loop"]


# Divide X and y into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("Total calls:", len(df))

print("\nTraining inputs:", X_train.shape)
print("Training answers:", y_train.shape)

print("\nTesting inputs:", X_test.shape)
print("Testing answers:", y_test.shape)

print("\nTraining label counts:")
print(y_train.value_counts())

print("\nTesting label counts:")
print(y_test.value_counts())
# Create an empty decision-tree model
model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42,
)


# Train the model using only the training data
model.fit(X_train, y_train)


# Display the rule learned by the model
tree_rules = export_text(
    model,
    feature_names=["repeat_score"],
)

print("\nLearned decision-tree rule:")
print(tree_rules)
# Ask the trained model to predict the 100 test calls
predictions = model.predict(X_test)


# Compare the predictions with the real test answers
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\nModel evaluation:")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 score:  {f1:.2%}")


print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
# Load a separate dataset containing more realistic conversations
realistic_df = pd.read_csv("data/realistic_test.csv")


# Use the same feature logic used during training
realistic_df["repeat_score"] = realistic_df["transcript"].apply(
    repeat_score
)


# Prepare the realistic model inputs and correct answers
X_realistic = realistic_df[["repeat_score"]]
y_realistic = realistic_df["repetition_loop"]


# Use the already-trained model
realistic_predictions = model.predict(X_realistic)


# Calculate realistic-test measurements
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


print("\nRealistic-data evaluation:")
print(f"Accuracy:  {realistic_accuracy:.2%}")
print(f"Precision: {realistic_precision:.2%}")
print(f"Recall:    {realistic_recall:.2%}")
print(f"F1 score:  {realistic_f1:.2%}")

print("\nRealistic-data confusion matrix:")
print(
    confusion_matrix(
        y_realistic,
        realistic_predictions,
    )
)