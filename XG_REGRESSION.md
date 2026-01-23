# Codex 5.2 Prompt — XGBoost Regression on PricePredictionCleanedUp.csv (Notebook + Cars.py)

You are Codex 5.2 acting as a careful ML engineer.  
Goal: Train an **XGBoost regressor** on `PricePredictionCleanedUp.csv` with target `selling_price`, produce a **Jupyter notebook** (`Cars.ipynb`) plus a runnable **Python script** (`Cars.py`), and save plots + metrics. Use **conda** to create a fresh environment and **uv** to install packages inside it.

## Critical fixes + constraints (must follow)
- The library is **xgboost** (not “xcboost”). Install `xgboost`.
- This is **regression**: there is no true “accuracy” or “confusion matrix” in the classic classification sense.
  - You MUST still provide the requested “accuracy/confusion” *as regression-friendly analogs*:
    - “Accuracy” → **R²** (and optionally explained variance).
    - “Confusion matrix” → **binned confusion matrix**: bin true and predicted `selling_price` into quantiles, then compute confusion matrix over bins.
- XGBoost doesn’t expose epoch-by-epoch “training accuracy” like deep nets, but it does support evaluation per boosting round:
  - Track and plot **training + validation loss** across boosting rounds using `eval_set` and `evals_result_`.
  - “Training accuracy per epoch” → plot **training + validation R² per boosting round** by computing predictions at checkpoints OR approximate by reporting per-round metric if available. If not feasible, plot only loss per round and also report final R².
- Use **seaborn + matplotlib** for visuals.
- Use **scikit-learn** for train/test split and metrics.
- Keep everything reproducible: set `random_state=42`.
- Assume the input is already numeric-heavy (AutoML style). Still, you must:
  - Drop rows with missing target.
  - Ensure all feature columns are numeric. If any non-numeric remain, encode or drop with documentation.

## Files to create
1. `Cars.py` (CLI runnable)
2. `Cars.ipynb` (same workflow, richer narrative)
3. `outputs/` folder containing saved plots (PNG) and `metrics.json`

## Environment setup (document exactly, in notebook + optionally in README cells)
Use conda + uv. Commands must be included verbatim:

```bash
conda create -n cars-xgb python=3.11 -y
conda activate cars-xgb

python -m pip install -U uv
uv pip install -U pandas numpy scikit-learn xgboost seaborn matplotlib jupyter ipykernel

python -m ipykernel install --user --name cars-xgb --display-name "Python (cars-xgb)"
jupyter notebook
