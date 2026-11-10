import random
import pandas as pd

SCENARIOS = [
    {"goal": "book a table for two at 7pm", "business": "Restaurant"},
    {"goal": "book a haircut for Friday at 3pm", "business": "Salon"},
    {"goal": "order a large pepperoni pizza", "business": "Shop"},
    {"goal": "schedule an appointment for next Monday", "business": "Clinic"},
]

FAILURES = ["none", "task_not_completed", "repetition_loop", "hallucination", "user_frustration"]

# Order of numbers: task_not_completed, repetition_loop, hallucination, user_frustration
LABELS = {
    "none":               [0, 0, 0, 0],
    "task_not_completed": [1, 0, 0, 0],
    "repetition_loop":    [1, 1, 0, 0],
    "hallucination":      [0, 0, 1, 0],
    "user_frustration":   [1, 0, 0, 1],
}


def build_call(scenario, failure):
    goal = scenario["goal"]
    business = scenario["business"]

    lines = [f"Agent: Hi, I'd like to {goal}."]

    if failure == "task_not_completed":
        lines.append(f"{business}: Sorry, we cannot help with that.")
        lines.append("Agent: Okay, thank you, goodbye.")
    elif failure == "repetition_loop":
        lines.append(f"{business}: Sorry, that is not available.")
        lines.append(f"Agent: Could I {goal}?")
        lines.append(f"{business}: Sorry, that is not available.")
        lines.append(f"Agent: Okay, I'd like to {goal}.")
        lines.append(f"{business}: Still not available.")
    elif failure == "hallucination":
        lines.append(f"{business}: Sure, that works.")
        lines.append("Agent: Great. Your confirmation number is 4821 and the total is 18 dollars.")
    elif failure == "user_frustration":
        lines.append(f"{business}: Sorry, we cannot help with that.")
        lines.append("Agent: Please, can you check anyway?")
        lines.append(f"{business}: This is useless, let me talk to a human.")
    else:
        lines.append(f"{business}: Sure, that works.")
        lines.append("Agent: Great, thank you, goodbye.")

    return " ".join(lines)


def make_dataset(n):
    rows = []
    for i in range(1, n + 1):
        scenario = random.choice(SCENARIOS)
        failure = random.choice(FAILURES)
        labels = LABELS[failure]
        rows.append({
            "call_id": i,
            "transcript": build_call(scenario, failure),
            "task_not_completed": labels[0],
            "repetition_loop": labels[1],
            "hallucination": labels[2],
            "user_frustration": labels[3],
        })
    return pd.DataFrame(rows)


random.seed(42)
df = make_dataset(500)
df.to_csv("data/synthetic_calls.csv", index=False)
print(df.shape)
print(df.head())