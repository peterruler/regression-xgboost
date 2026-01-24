You are Codex. Create a complete, reproducible end-to-end ML pipeline for car price prediction (REGRESSION) from a CSV with columns:

brand, model, min_cost_price, max_cost_price, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats, selling_price

SOURCE File: /training/PricePrediction.csv
TARGET File: /training/PricePredictionCleanedUp.csv

TARGET:
- selling_price (float) is the regression target to predict.

DELIVERABLES:
1) TARGET File: /training/PricePredictionCleanedUp.csv
2) training/Cars.ipynb (Colab-friendly) that:
   - mounts Google Drive OR provides a file upload option
   - loads the CSV from google drive csv_path = '/content/drive/MyDrive/ML/PricePredictionCleanedUp.csv'
   - runs all preprocessing + training + evaluation
   - saves the trained pipeline with 
model.get_booster().save_model(booster_path)
   - loads it back as loaded_model
   via 
booster = xgb.Booster()
booster.load_model('xgboost_price_model.json')
   - predicts for a sample_input (single-row DataFrame)
3) training/Cars.py that provides:
   - load_data(path)
   - build_preprocessor(reference_date=None)
   - train_model(df)
   - evaluate_model(model, X_test, y_test)
   - save_model(model, path)
   - load_model(path) -> loaded_model
   - predict_sample(loaded_model, sample_input)
4) requirements.txt

MODEL REQUIREMENT:
- use 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
to split the data in train and test part
- Use XGBoost for the primary model:
  - Prefer xgboost.XGBRegressor as the final model.
  - Also train a simple baseline model (Ridge or LinearRegression) for comparison.
- If xgboost is not installed, install it in the notebook (pip) and include it in requirements.txt.

CONSTRAINTS:
- Use Python, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib.
- Use ColumnTransformer + Pipeline so preprocessing is included in the saved model.
- Use OneHotEncoder(handle_unknown="ignore") for categoricals.
- Use SimpleImputer(strategy="median") for numerics, and SimpleImputer(strategy="most_frequent") for categoricals (or constant "Unknown").
- Use train_test_split with random_state fixed.
- EVALUATION MUST USE REGRESSION METRICS ONLY: MAE, RMSE, R2.
- DO NOT create a confusion matrix (classification only).
- Include plots: predicted vs actual scatter, residual distribution/plot, and feature importance (XGBoost).

DATA ENGINEERING / PREPROCESSING TASKS (implement all):
A) Basic validation & cleaning
- Load CSV -> DataFrame.
- Standardize column names to lowercase + strip spaces.
- Assert required columns exist.
- Remove duplicate rows.
- Replace common missing tokens ("", "NA", "N/A", "null", "-", "--") with NaN.

B) Parse and coerce numeric columns (robust cleaning)
Numeric candidate columns:
- min_cost_price, max_cost_price, km_driven, mileage, engine, max_power, seats
- vehicle_age will be handled specially (see section C)
Tasks:
- Coerce to numeric with errors='coerce'.
- If any numeric fields may contain units (examples: "18 kmpl", "1197 CC", "74 bhp"), strip non-numeric characters safely (keep decimal points) before coercion.
- Clip obvious invalid values (e.g., km_driven < 0, seats <= 0) -> set to NaN then impute.

C) vehicle_age timestamp logic (MUST)
vehicle_age may either be:
1) already an age in years (small numeric like 0–30), OR
2) a Unix timestamp in seconds (~1e9–2e9) or milliseconds (~1e12–2e12).

Implement detection + conversion:
- If median(vehicle_age) >= 1e11 -> treat as milliseconds Unix timestamp.
- Else if median(vehicle_age) >= 1e9 -> treat as seconds Unix timestamp.
- Else treat as "age in years" already.

If timestamp:
- Convert to datetime using pandas.to_datetime with correct unit.
- Compute vehicle_age_years = (reference_date - vehicle_date).days / 365.25
- reference_date:
  - default: fixed at runtime as pd.Timestamp.today().normalize()
  - but store/print it and allow user to pass a custom reference_date for reproducibility.
- Drop raw vehicle_age timestamp column after deriving vehicle_age_years, and use vehicle_age_years as the numeric feature.
- Optionally also derive vehicle_year = vehicle_date.dt.year (numeric) and include it.

If already years:
- Rename/standardize it as vehicle_age_years and keep as numeric.

Handle invalid timestamps/ages:
- Coerce invalid to NaN; impute vehicle_age_years using median.

D) Missing values
- Numeric: impute median.
- Categorical: impute "Unknown" or most_frequent.

E) Redundant / leakage columns
- min_cost_price and max_cost_price might encode market price bounds that could be too close to selling_price.
Implement BOTH options:
1) Default: keep them but add engineered features cost_mid and cost_range.
2) Provide a clearly labeled switch (e.g., DROP_COST_BOUNDS = False). If True, drop min_cost_price and max_cost_price to reduce leakage risk.
Always document why this matters.

Feature engineering (if cost bounds kept):
- cost_mid = (min_cost_price + max_cost_price)/2
- cost_range = max_cost_price - min_cost_price

Also add:
- km_per_year = km_driven / max(vehicle_age_years, 1e-6)

F) Encoding categoricals
Categorical columns:
- brand, model, seller_type, fuel_type, transmission_type
- OneHotEncode with handle_unknown="ignore".
Optional: reduce model cardinality by grouping rare models (< N occurrences) to "Other" (implement as a parameter RARE_MODEL_MIN_COUNT, default 20).

G) Train/test split and model training
- Split X/y into train/test 80/20 with random_state=42.
- Train baseline: Ridge (or LinearRegression).
- Train final: XGBRegressor (xgboost) inside the Pipeline.
  - Use sensible defaults and set random_state.
  - Use early stopping if you create a validation set; otherwise skip early stopping.
  - Optionally add RandomizedSearchCV for a small hyperparameter search (nice-to-have).
- Select XGBRegressor pipeline as the final saved model.

H) Evaluation outputs
- Print MAE, RMSE, R2 on test set for BOTH baseline and XGBoost.
- Plot predicted vs actual (XGBoost).
- Plot residuals (XGBoost).
- Plot feature importances for XGBoost (top 20), using get_booster().get_score() or model.feature_importances_ with correct feature names from the preprocessor.

I) Save/load and sample prediction
- Save final pipeline with joblib.dump to "car_price_pipeline.joblib"
- Load it as loaded_model
- Create sample_input as a single-row DataFrame with realistic values for all original raw input columns (excluding target).
- Run loaded_model.predict(sample_input) and print result.

OUTPUT FORMATTING:
- Provide the full content for Cars.py, requirements.txt, and the notebook code cells in Cars.ipynb.
- In the notebook, include clear section headings and make it runnable top-to-bottom in Google Colab.