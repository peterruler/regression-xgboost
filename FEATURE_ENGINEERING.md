# Codex 5.2 Prompt — Feature Engineering + Clean CSV for Google AutoML

You are Codex 5.2 acting as a meticulous data engineer.  
Goal: take `Price_Prediction.csv`, perform feature engineering (numeric normalization + bucketization, categorical one-hot + embedding-friendly encodings, date/time extraction, text-derived features), and write `PricePredictionCleanedUp.csv` optimized for Google AutoML. Also **update THIS markdown file** with an explicit, reproducible “Steps performed” section that matches what the code actually did.

## Hard requirements (must follow)
- Use **Python + pandas** (numpy allowed). Keep dependencies minimal.
- Use **uv** to install packages (document exact commands).
- Output file: **`PricePredictionCleanedUp.csv`**
- Input file: **`Price_Prediction.csv`**
- One-hot encoding must be **0/1 integers** (NOT True/False). Enforce integer dtype.
- Always update **this .md file** whenever you change logic (steps list must stay accurate).
- Optimize for **Google AutoML**:
  - Prefer **flat, tabular**, mostly numeric columns.
  - Avoid non-serializable objects, nested structures.
  - Ensure consistent column names, no duplicate columns, no mixed dtypes.
  - Handle missing values (no NaNs left if possible).
- Preserve the target column (assume it is `selling_price`) unchanged.
- Make the pipeline **reproducible** and **idempotent**.

## Deliverables
1. A script (suggested: `feature_engineer.py`) that:
   - Reads `Price_Prediction.csv`
   - Engineers features per rules below
   - Writes `PricePredictionCleanedUp.csv`
   - Prints a short summary (rows/cols, missing values after processing)
2. Update this markdown file with:
   - Exact `uv` install/run commands
   - “Steps performed” bullet list (truthfully reflecting the final implementation)
   - Brief schema notes: target column, what new feature families were added
   - Any detected date/time columns and what was extracted (or note none found)

## Feature engineering rules

### 1) Load + basic cleaning
- Load with pandas (`pd.read_csv`).
- Standardize column names (snake_case, strip spaces) **only if needed**; if you rename, document it and be consistent.
- Identify target column: `selling_price` (do not transform it).
- Infer column groups:
  - Numeric: `select_dtypes(include=[number])` excluding target
  - Categorical: object/category columns
  - Timestamp/date candidates: columns whose name contains any of: `date`, `time`, `timestamp`, `ts`, `year`, `month`, `day` OR values parse cleanly via `pd.to_datetime` with a high success ratio.

### 2) Missing values (AutoML-friendly)
- Numeric: impute with median.
- Categorical: impute with string `"unknown"`.
- Document how many missing values were present before/after.

### 3) Normalize + bucketize numeric features
For each numeric feature column (excluding `selling_price`):
- Add z-score normalized column: `<col>_z`
  - z = (x - mean) / std; if std==0, set z to 0.
- Add bucketized column: `<col>_bucket`
  - Use `pd.qcut(col, 5, duplicates="drop")` → integer codes
  - If qcut fails, fallback to `pd.cut(col, 5)` → integer codes
- Ensure bucket columns are integers.

### 4) Categorical features: one-hot + embeddings
For each categorical feature column:
- One-hot encode via `pd.get_dummies(..., dtype=int)` so values are **0/1 ints**.
- Add embedding-friendly encodings:
  - `<col>_cat_id`: integer category id via `.astype("category").cat.codes` (ensure `"unknown"` included)
  - `<col>_freq`: frequency encoding: count(category)/N as float

### 5) Date/time features
For each detected timestamp/date column:
- Parse with `pd.to_datetime(errors="coerce", utc=False)`.
- Add:
  - `<col>_year`, `<col>_month`, `<col>_day`
  - `<col>_dayofweek`, `<col>_is_weekend`
  - If time is present: `<col>_hour`, `<col>_minute`
- Optionally drop the original raw timestamp string column if it harms AutoML (only do this if you justify it in the md update).

### 6) Text-derived features
From relevant text columns (at minimum: `brand`, `model`, and a combined `brand_model` if both exist):
- Create:
  - length in chars: `<col>_len`
  - word count: `<col>_words`
- Keep them numeric (ints).

### 7) Output hygiene
- Ensure:
  - No boolean columns remain (convert to 0/1 ints).
  - No object columns remain unless you intentionally keep them (prefer numeric for AutoML since we already one-hot + encodings).
  - No NaNs (or document any unavoidable ones).
- Write `PricePredictionCleanedUp.csv` with `index=False`.

## Acceptance checklist (must pass)
- `PricePredictionCleanedUp.csv` exists and loads without dtype issues.
- One-hot columns contain only 0/1 ints, not True/False.
- Target `selling_price` is present and unchanged.
- Script runs end-to-end with only uv-installed deps.
- This markdown file contains an updated “Steps performed” that exactly matches the final script.

## Now do the work
- Create/update the script.
- Run it locally (or simulate run outputs if execution isn’t available), but make sure logic is correct.
- Save the cleaned CSV.
- Update this markdown file’s “Steps performed” section to reflect the final implementation (replace any outdated bullets).
