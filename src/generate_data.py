import random
import pandas as pd

SCENARIOS = [
    {"goal": "book a table for two at 7pm", "business": "Restaurant"},
    {"goal": "book a haircut for Friday at 3pm", "business": "Salon"},
    {"goal": "order a large pepperoni pizza", "business": "Shop"},
    {"goal": "schedule an appointment for next Monday", "business": "Clinic"},
]

FAILURES = [
    "none",
    "good_sorry_then_ok",
    "good_confirmed",
    "task_not_completed",
    "repetition_loop",
    "hallucination",
    "user_frustration",
]

# Order of numbers: task_not_completed, repetition_loop, hallucination, user_frustration
LABELS = {
    "none":               [0, 0, 0, 0],
    "good_sorry_then_ok": [0, 0, 0, 0],
    "good_confirmed":     [0, 0, 0, 0],
    "task_not_completed": [1, 0, 0, 0],
    "repetition_loop":    [1, 1, 0, 0],
    "hallucination":      [0, 0, 1, 0],
    "user_frustration":   [1, 0, 0, 1],
}

REFUSALS = [
    "Sorry, we cannot help with that.",
    "I'm afraid that is not possible today.",
    "We don't have anything like that right now.",
    "Unfortunately that is not available.",
]
NOT_AVAILABLE = [
    "Sorry, that is not available.",
    "Still not available.",
    "We have nothing open.",
    "No, I already told you that is not possible.",
    "We are fully reserved this week.",
    "I cannot fit anyone in right now.",
    "That item is out of stock.",
    "We stopped offering that.",
    "The schedule is packed until next month.",
    "Nothing left for that day.",
    "We do not have that option.",
    "I am afraid we cannot do that.",
    "There are no spots left.",
    "That service is paused at the moment.",
]

REPEAT_ASKS = [
    "Could I {goal}?",
    "Okay, I'd like to {goal}.",
    "Let me ask again, can I {goal}?",
    "Just to confirm, I want to {goal}.",
]

MADE_UP = [
    "Great. Your confirmation number is 4821 and the total is 18 dollars.",
    "Perfect, you are confirmed with reference 7390 and the fee is 25 dollars.",
    "All set. The booking code is B552 and it will cost 40 dollars.",
    "Wonderful, the owner Maria approved it, and there is no charge.",
]

SHOP_CONFIRMS = [
    "Sure, your confirmation number is 4821 and the total is 18 dollars.",
    "Yes, the booking code is B552 and it will cost 40 dollars.",
    "Done. Your reference is 7390 and the fee is 25 dollars.",
]

CHECK_ANYWAY = [
    "Please, can you check anyway?",
    "Could you try once more?",
    "Can you look again, please?",
]

FRUSTRATED = [
    "This is useless, let me talk to a human.",
    "You are not helping. Can I speak to a real person?",
    "Forget it, put someone else on the line.",
    "This is ridiculous, I want a human.",
]

GOOD_REPLIES = [
    "Sure, that works.",
    "Yes, we can do that.",
    "Absolutely, that is fine.",
]

GOODBYES = [
    "Great, thank you, goodbye.",
    "Perfect, thanks a lot, bye.",
    "Wonderful, have a good day.",
]


def build_call(scenario, failure):
    goal = scenario["goal"]
    business = scenario["business"]

    lines = [f"Agent: Hi, I'd like to {goal}."]

    if failure == "task_not_completed":
        lines.append(f"{business}: {random.choice(REFUSALS)}")
        lines.append("Agent: Okay, thank you, goodbye.")
    elif failure == "repetition_loop":
        lines.append(f"{business}: {random.choice(NOT_AVAILABLE)}")
        lines.append("Agent: " + random.choice(REPEAT_ASKS).format(goal=goal))
        lines.append(f"{business}: {random.choice(NOT_AVAILABLE)}")
        lines.append("Agent: " + random.choice(REPEAT_ASKS).format(goal=goal))
        lines.append(f"{business}: {random.choice(NOT_AVAILABLE)}")
    elif failure == "hallucination":
        lines.append(f"{business}: {random.choice(GOOD_REPLIES)}")
        lines.append("Agent: " + random.choice(MADE_UP))
    elif failure == "user_frustration":
        lines.append(f"{business}: {random.choice(REFUSALS)}")
        lines.append("Agent: " + random.choice(CHECK_ANYWAY))
        lines.append(f"{business}: {random.choice(FRUSTRATED)}")
    elif failure == "good_sorry_then_ok":
        lines.append(f"{business}: Sorry, that exact time is taken, but we have another slot.")
        lines.append("Agent: Yes, that other slot works for me.")
        lines.append(f"{business}: Done, you are booked.")
        lines.append("Agent: " + random.choice(GOODBYES))
    elif failure == "good_confirmed":
        lines.append(f"{business}: {random.choice(SHOP_CONFIRMS)}")
        lines.append("Agent: Perfect, I have those details. Thank you, goodbye.")
    else:
        lines.append(f"{business}: {random.choice(GOOD_REPLIES)}")
        lines.append("Agent: " + random.choice(GOODBYES))

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