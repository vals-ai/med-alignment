## Agreement calculations (vibe-coded)
- Fleiss’ kappa helpers (`fleiss_kappa`, `pairwise_agreement`) are provided for measuring inter-rater consistency. Only `pairwise_agreement` is used in the CLI run today.
- Computes the human majority vote per index, skipping ties.
- Compares each auto evaluator’s labels to the human majority and prints its percent agreement.

## Data

```
alignment_test/
├── data/
│   ├── human_review/*.csv         # human QA exports
│   └── custom_operator/*.csv      # auto-eval exports
├── main.py
└── README.md
```

## Running the Script

1. Create/activate a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install pandas
```

2. Execute `main.py`:

```bash
python main.py
```
