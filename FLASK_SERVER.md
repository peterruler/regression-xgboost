# Codex 5.2 Prompt — Flask Web App for Car Price Prediction (WTForms + Model Inference)

You are Codex 5.2 acting as a careful full-stack engineer.  
Build a small **Flask** web app that lets a user enter car features in a **web form** (WTForms / Flask-WTF), sends them to a prediction endpoint, and shows the predicted `selling_price`. The form fields must match the **feature columns** used to train the model (i.e., the columns of `PricePredictionCleanedUp.csv` excluding `selling_price`). Update `requirements.txt`.

## Hard requirements (must follow)
- Create `app.py` Flask server.
- Use **Flask-WTF + WTForms** for the form.
- Use templates (`templates/` folder) with `render_template`.
- Load a saved model with **joblib** (assume model file exists, default: `models/cars_model.joblib`).
- Use **the same input schema as one row** of `PricePredictionCleanedUp.csv` (features only, no target).
- Generate correct form fields automatically from the CSV schema:
  - Numeric columns → `FloatField` or `IntegerField`
  - 0/1 one-hot columns → `BooleanField` but convert to **int 0/1** before prediction OR use `IntegerField` with validation (0 or 1).
  - Category id / freq columns → numeric fields.
  - Do **not** hardcode dozens of fields manually unless the dataset is tiny; implement dynamic form generation.
- Create a robust preprocessing step to ensure the final `pandas.DataFrame` passed to the model has **exactly the same column order** as training features.
  - Use a persisted list of feature names if available (preferred). If not, infer from `PricePredictionCleanedUp.csv`.
- Add CSRF protection via Flask-WTF (needs `SECRET_KEY`).
- Update `requirements.txt` with required packages.

## Project layout to create
- `app.py`
- `requirements.txt`
- `templates/`
  - `base.html`
  - `index.html` (form page + prediction output)
- `models/`
  - (assume `cars_model.joblib` exists; do not create the model here)
- Optional: `static/` (basic styling)

## Assumptions / Inputs
- CSV file: `PricePredictionCleanedUp.csv` in project root (used to read schema / feature columns).
- Target column: `selling_price` (exclude from form).
- Model file: `models/cars_model.joblib`

## Flask behavior
- Route `/` (GET/POST):
  - GET: render form
  - POST: validate form, build feature row DataFrame, run prediction, render result on same page
- Optional route `/reset`:
  - clears session and redirects back to `/`

## Implementation details (must implement)
### 1) Schema extraction
- Read header of `PricePredictionCleanedUp.csv` with pandas.
- Determine feature columns: all columns except `selling_price`.
- Determine field types:
  - If column dtype numeric: numeric WTForms field
  - If column name looks like one-hot (contains patterns like `brand_`, `model_`, etc.) AND values are {0,1}: treat as binary
- For binary fields: show as checkbox (BooleanField) but convert `True/False` to `1/0` before prediction.

### 2) Dynamic WTForm
- Create a function `make_dynamic_form(feature_specs)` that returns a FlaskForm subclass with fields added dynamically.
- Include `SubmitField("Predict")`.
- Use validators:
  - numeric: `InputRequired()` (or allow optional with default 0) depending on what makes sense
  - binary: default `False`
- Add user-friendly labels by converting column names to title case.

### 3) Model loading
- Load model once at startup (global), not per request.
- If model file missing, show a helpful error page/message.

### 4) Prediction input building
- On POST:
  - Create dict `{col: value}` for every feature column
  - Convert booleans to int
  - Fill missing fields with 0 or median:
    - Prefer default 0 for binary and engineered columns
  - Create `X = pd.DataFrame([row_dict], columns=feature_cols)` to enforce order
- Predict:
  - `pred = model.predict(X)[0]`
  - Render predicted selling price (format nicely, e.g. 2 decimals)

### 5) Templates
- `base.html`: basic layout, includes flash message area
- `index.html`: renders form in a grid (so many fields are usable)
  - show prediction result area if available
  - show validation errors

### 6) requirements.txt
Add at minimum:
- flask
- flask-wtf
- wtforms
- pandas
- numpy
- scikit-learn
- joblib
- xgboost (only if your saved model requires it; include it to be safe)

## Run instructions (must include in README comment at top of app.py)
```bash
pip install uv
pip install --upgrade pip
python -m venv .venv
source .venv/bin/activate  # mac/linux
uv pip install -r requirements.txt
export FLASK_APP=app.py
flask run --reload
