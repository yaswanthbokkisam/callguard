import re

import numpy as np


FRUSTRATION_SIGNALS = (
    "useless",
    "ridiculous",
    "unacceptable",
    "not acceptable",
    "annoying",
    "frustrating",
    "impossible",
    "unhappy",
    "losing patience",
    "supervisor",
    "manager",
    "human",
    "real person",
    "stop repeating",
    "third time",
    "third call",
    "hanging up",
    "not helping",
    "going nowhere",
    "had enough",
    "terrible experience",
    "waste of my time",
)

REFUSAL_SIGNALS = (
    "not available",
    "cannot",
    "can't",
    "no openings",
    "fully booked",
    "sold out",
    "closed",
    "nothing available",
    "not possible",
    "do not have",
    "don't have",
    "system is down",
    "out of stock",
    "no quote",
    "not at the moment",
    "need the account",
    "need the address",
    "without it",
    "call back",
)

CONTRADICTION_SIGNALS = (
    "didn't say",
    "did not say",
    "never said",
    "no, nothing",
    "no booking",
    "nothing has been",
    "didn't mention",
    "did not mention",
    "don't see any note",
    "separate fee",
    "isn't approved",
    "not approved",
    "there's no booking",
    "we didn't have",
    "no, there's no",
    "no, i said",
)

UNSUPPORTED_CLAIM_SIGNALS = (
    "confirmation number",
    "booking code",
    "reference is",
    "total is",
    "fee is",
    "approved it",
    "no charge",
    "i have confirmed",
    "i've confirmed",
    "i have booked",
    "i've got my client booked",
    "balance is",
    "arrives tomorrow",
    "ready for pickup at",
    "is changed",
)

SUCCESS_SIGNALS = (
    "you are booked",
    "booked",
    "confirmed",
    "all set",
    "everything is confirmed",
    "booking is complete",
    "see you",
    "sounds good",
    "works for me",
    "done",
    "scheduled",
    "it is updated",
    "it's updated",
)

STRUCTURED_FEATURE_NAMES = (
    "repeat_score",
    "frustration_score",
    "refusal_score",
    "contradiction_score",
    "unsupported_claim_score",
    "success_score",
    "agent_turn_count",
)


def agent_turns(transcript):
    """Return each Agent turn from a transcript."""
    return re.findall(
        r"Agent:\s*(.*?)(?=\s[A-Z][\w ]{0,25}:\s|$)",
        str(transcript),
    )


def same_words(a, b):
    """Return Jaccard word overlap from 0 (none) to 1 (identical)."""
    words_a = set(re.findall(r"[a-z']+", a.lower()))
    words_b = set(re.findall(r"[a-z']+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def repeat_score(transcript):
    """Count Agent turns that substantially repeat an earlier Agent turn."""
    turns = agent_turns(str(transcript))
    repeats = 0
    for current_index in range(1, len(turns)):
        for earlier_index in range(current_index):
            if same_words(turns[current_index], turns[earlier_index]) >= 0.4:
                repeats += 1
                break
    return repeats


def matched_signals(transcript, signals):
    """Return configured phrases that occur in a transcript."""
    text = str(transcript).lower()
    return [signal for signal in signals if signal in text]


def signal_score(transcript, signals):
    """Count distinct configured phrases found in a transcript."""
    return len(matched_signals(transcript, signals))


def frustration_score(transcript):
    return signal_score(transcript, FRUSTRATION_SIGNALS)


def refusal_score(transcript):
    return signal_score(transcript, REFUSAL_SIGNALS)


def contradiction_score(transcript):
    return signal_score(transcript, CONTRADICTION_SIGNALS)


def unsupported_claim_signals(transcript):
    """Find claim phrases introduced by the Agent without earlier support."""
    text = str(transcript)
    found = []
    pattern = r"Agent:\s*(.*?)(?=\s[A-Z][\w ]{0,25}:\s|$)"

    for match in re.finditer(pattern, text):
        earlier_text = text[: match.start()].lower()
        agent_text = match.group(1).lower()
        for signal in UNSUPPORTED_CLAIM_SIGNALS:
            if (
                signal in agent_text
                and signal not in earlier_text
                and signal not in found
            ):
                found.append(signal)
    return found


def unsupported_claim_score(transcript):
    return len(unsupported_claim_signals(transcript))


def success_score(transcript):
    return signal_score(transcript, SUCCESS_SIGNALS)


def structured_features(transcripts):
    """Convert transcripts into the six transparent numeric features."""
    rows = []
    for transcript in transcripts:
        rows.append(
            [
                repeat_score(transcript),
                frustration_score(transcript),
                refusal_score(transcript),
                contradiction_score(transcript),
                unsupported_claim_score(transcript),
                success_score(transcript),
                len(agent_turns(transcript)),
            ]
        )
    return np.asarray(rows, dtype=float)
