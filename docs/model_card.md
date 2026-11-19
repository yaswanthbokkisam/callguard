# CallGuard Model Card

## Intended use

CallGuard is a portfolio demonstration for prioritizing voice-AI transcripts
for human QA review. It is designed for offline text analysis, not live call
control.

## Outputs

The system returns four binary flags:

- `task_not_completed`
- `repetition_loop`
- `hallucination` (displayed as unsupported claim)
- `user_frustration`

The two Logistic Regression detectors also return model scores. The two rule
detectors return evidence counts rather than pretending that a rule count is a
calibrated probability.

## Training data

The model is trained on 500 reproducibly generated synthetic calls. Synthetic
data allows controlled labels but cannot represent the full variety of real
language.

## Evaluation data

Validation uses 60 manually labeled, more realistic transcripts. This dataset
was used during feature and threshold development, so it is validation data,
not a fully untouched production test.

## Modeling approach

- TF-IDF unigrams and bigrams represent transcript text.
- Logistic Regression predicts task completion and frustration.
- Structured numeric features capture domain signals.
- Repetition uses repeated-turn word overlap.
- Unsupported claims use explicit contradiction evidence.

## Risks and limitations

- Unseen wording can cause missed failures.
- Known phrases can cause false alarms in a different context.
- Transcript errors and missing speaker labels can reduce performance.
- Scores are not guarantees and should not trigger punitive action by
  themselves.
- Sensitive transcripts require appropriate access control and retention
  policies.

## Human oversight

Use flags to prioritize review. A human should read the full transcript before
making operational, employment, compliance, or customer-impacting decisions.

## Recommended next work

- collect consented, representative production examples
- create an untouched final test set
- measure performance by call domain and customer group
- calibrate model scores
- monitor drift and retrain when language changes
