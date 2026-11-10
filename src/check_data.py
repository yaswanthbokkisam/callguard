import pandas as pd

df = pd.read_csv("data/sample_calls.csv")
print(df.shape)
labels = ["task_not_completed", "repetition_loop", "hallucination", "user_frustration"]
print(df[labels].sum())