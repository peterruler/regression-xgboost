# Codex 5.2 Prompt — Google Colab Notebook (Drive CSV import + Train + Joblib Save/Load + Sample Prediction)

You are Codex 5.2 acting as a careful ML engineer.  
Create a **Google Colab–ready Jupyter notebook** (`Cars_Colab.ipynb`) that trains a regression model from a CSV stored in **Google Drive**, saves the trained model with **joblib**, reloads it as **`loaded_model`**, and performs a prediction on **`sample_input`**.

## Hard requirements (must follow)
- Notebook must run on **Google Colab** without local dependencies.
- Must include:
  - A clear **“Open in Colab”** badge/button (markdown cell).
  - Google Drive mount code (`drive.mount('/content/drive')`).
  - A file picker OR a clearly editable path variable for the CSV in Drive.
- Use `pandas` to load the CSV.
- Use `scikit-learn` for train/test split and metrics.
- Use `joblib` to:
  - save model to Drive
  - load model into variable **`loaded_model`**
- Create `sample_input` and run `loaded_model.predict(sample_input)` successfully.
- Handle feature alignment robustly: `sample_input` must have the **same feature columns** as training data (order and names).
- Keep everything reproducible: `random_state=42`.

## Assumptions
- Dataset is `PricePredictionCleanedUp.csv` with target column `selling_price`.
- If the target column is missing, stop with a clear error message.

## Deliverable
- Create **one notebook**: `Cars_Colab.ipynb`

---

## Notebook structure (must implement in this order)

### 0) Title + Colab import button
Add a markdown cell with:
- Title: “Car Price Regression (Colab)”
- “Open in Colab” badge linking to the notebook’s GitHub path (use a placeholder URL and comment `# TODO: replace with your repo link`)

### 1) Colab setup + imports
Code cell:
- Standard imports:
  - `pandas`, `numpy`
  - `train_test_split`
  - `mean_absolute_error`, `mean_squared_error`, `r2_score`
  - `joblib`
- If you use XGBoost or other libs, include a cell:
  ```python
  !pip -q install xgboost
