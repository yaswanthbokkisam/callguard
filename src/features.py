import re


def agent_turns(transcript):
    # finds everything the Agent said, one piece per turn
    return re.findall(r"Agent:\s*(.*?)(?=\s[A-Z][\w ]{0,25}:\s|$)", transcript)


def same_words(a, b):
    # how much two sentences overlap, from 0 (nothing) to 1 (identical)
    words_a = set(re.findall(r"[a-z']+", a.lower()))
    words_b = set(re.findall(r"[a-z']+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def repeat_score(transcript):
    # counts how many Agent turns are a near-copy of an earlier Agent turn
    turns = agent_turns(transcript)
    repeats = 0
    for i in range(1, len(turns)):
        for j in range(i):
            if same_words(turns[i], turns[j]) >= 0.6:
                repeats += 1
                break
    return repeats