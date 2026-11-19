# CallGuard

CallGuard is an educational machine-learning portfolio project that analyzes
voice-AI call transcripts and flags four failure types:

- task not completed
- repetition loop
- unsupported claim (hallucination)
- user frustration

It helps a QA engineer prioritize calls for human review. It does not replace
human judgment or automatically change an AI agent.

## Why this project exists

A company may receive hundreds of AI-agent call transcripts each day. Reading
every call manually is slow. CallGuard turns each transcript into a small set
of review flags and displays the evidence behind transparent rule decisions.

## Architecture

```text
Transcript
   |
   +--> TF-IDF word and phrase features
   |
   +--> structured features
        - repeated-turn count
        - frustration signals
        - refusal signals
        - contradiction signals
        - success signals
        - agent-turn count
   |
   +--> Logistic Regression
        - task_not_completed
        - user_frustration
   |
   +--> transparent rules
        - repetition_loop
        - hallucination / unsupported claim
   |
   +--> flags, model scores, and evidence
```

This is a hybrid system: machine learning handles language patterns while
rules handle failure types that require clear, auditable evidence.

## Validation results

The saved report evaluates 60 realistic, manually labeled calls.

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| Task not completed | 92.0% | 85.2% | 88.5% |
| Repetition loop | 100.0% | 100.0% | 100.0% |
| Unsupported claim | 90.0% | 100.0% | 94.7% |
| User frustration | 72.7% | 80.0% | 76.2% |

These are validation results for a small educational dataset, not a production
performance guarantee.

## Project structure

```text
callguard/
|-- app.py                         # Streamlit entry point
|-- data/
|   |-- synthetic_calls.csv        # generated training data
|   |-- realistic_test.csv         # realistic validation data
|   `-- sample_calls.csv            # small batch-upload example
|-- docs/
|   |-- problem_statement.md
|   `-- model_card.md
|-- models/
|   `-- callguard.joblib            # trained deployable model bundle
|-- reports/
|   `-- validation_metrics.json
|-- src/
|   |-- app.py                     # Streamlit interface
|   |-- features.py                # transparent feature logic
|   |-- generate_data.py           # reproducible synthetic-data generator
|   |-- modeling.py                # reusable train/predict functions
|   `-- train_models.py             # final training and evaluation command
`-- requirements.txt
```

The older `train_repetition_model.py` and `train_frustration_model.py` files
are learning experiments. The deployed application uses `modeling.py`.

## Run locally on Windows

From `AI-Engineer-ML-Course\callguard`:

```powershell
..\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train_models
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

## Use the application

### Single call

Paste a transcript using speaker labels such as `Agent:` and `Restaurant:`.
The app displays all four flags, model scores, and rule evidence.

### Batch CSV

Upload a CSV containing a column named `transcript`. A `call_id` column is
optional. Download the completed predictions as `callguard_results.csv`.

## Regenerate and retrain

```powershell
python src\generate_data.py
python -m src.train_models
```

`random.seed(42)` makes synthetic generation reproducible.

## Deploy with Streamlit Community Cloud

1. Push the `callguard` folder to a GitHub repository.
2. Sign in at `share.streamlit.io` using GitHub.
3. Choose the repository and branch.
4. Set the main file to `app.py`.
5. Select **Deploy**.

The app requires no API key. `models/callguard.joblib` must be committed with
the project so the cloud app can load it.

## Limitations

- Most training examples are synthetic.
- The validation dataset is small.
- Rules depend on speaker formatting and known phrases.
- The system analyzes text only, not audio, tone, or timing.
- Real deployment requires representative data, privacy review, monitoring,
  threshold review, and repeated human evaluation.

See [docs/model_card.md](docs/model_card.md) for responsible-use details.
