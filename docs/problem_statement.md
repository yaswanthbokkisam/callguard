# Problem Statement

## 1. Who uses this tool?
A QA engineer at a company that runs voice AI agents. They review
about 200 phone call transcripts every day and cannot read all of them.

## 2. What does it help them decide?
Which calls to review first, and which kind of problem to fix.
The tool flags each call with the failure types it found.

## 3. What does "good enough" mean?
The model should find at least 8 out of 10 real failures
(recall of 0.80 or higher) with an F1 score of at least 0.75.
Missing a real failure is worse than flagging a good call, because
a missed failure means a bad call goes unnoticed.

## 4. What is out of scope?
No audio analysis, only text transcripts. No live detection during
a call. No automatic fixing of the agent.