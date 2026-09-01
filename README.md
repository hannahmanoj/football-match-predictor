# 26' world cup match & tournament simulator

a streamlit football analytics app that predicts individual match outcomes and runs Monte Carlo tournament simulations

## what it does

- predicts win/draw/loss probabilities for any two teams.
- explains predictions using rating, ranking, form, attack, and defense factors.
- simulates group-stage matches.
- advances teams using the 2026-style format: top two in each group plus the best third-place teams.
- builds a seeded round of 32 bracket from group-stage performance instead of randomly shuffling qualifiers.
- runs Monte Carlo simulations to estimate each team's chance of reaching each tournament stage.
- shows champion probabilities in an interactive Streamlit dashboard.

## project structure

```text
app/
  streamlit_app.py
data/
  sample_team_ratings.csv
src/
  predictor.py
  simulator.py
EPL_match_predictor.ipynb
matches.csv
requirements.txt
```

## run the app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app/streamlit_app.py
```

## update the data

download the latest international match results:

```bash
python3 src/download_international_data.py
```

Build team ratings from completed matches:

```bash
python3 src/build_team_ratings.py
```

this creates `data/team_ratings.csv`, which the streamlit app uses automatically. The generated ratings include an Elo-style rating, an Elo-derived ranking, recent form points, attack rating, defense rating, goals scored per match, and goals conceded per match.

train the match prediction model:

```bash
python3 src/train_model.py
```

this creates `models/match_model.joblib` and `models/model_report.txt`. when the model file exists, `src/predictor.py` uses the selected trained model probabilities. If the model file is missing, it falls back to the simpler rating formula.

the training script also writes:

- `models/model_metrics.json` for dashboard metrics.
- `models/calibration_curve.csv` for the reliability chart.
- `models/feature_importance.csv` for model explainability.
- brier score, ranked probability score, and expected calibration error in the model report.
