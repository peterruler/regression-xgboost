"""
Run instructions:
```bash
pip install uv
pip install --upgrade pip
python -m venv .venv
source .venv/bin/activate  # mac/linux
uv pip install -r requirements.txt
export FLASK_APP=app.py
flask run --reload
```
"""

import os

import pandas as pd
import xgboost as xgb
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import Optional

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(APP_ROOT, "training", "PricePredictionCleanedUp.csv")
FALLBACK_CSV_PATH = os.path.join(APP_ROOT, "PricePredictionCleanedUp.csv")
CSV_PATH = os.environ.get("CSV_PATH", DEFAULT_CSV_PATH)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(APP_ROOT, "models", "xgboost_price_model.json"))


def load_schema(csv_path: str):
    if not os.path.exists(csv_path) and os.path.exists(FALLBACK_CSV_PATH):
        csv_path = FALLBACK_CSV_PATH
    df = pd.read_csv(csv_path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    if "selling_price" not in df.columns:
        raise ValueError("Expected selling_price column in CSV.")

    feature_cols = [c for c in df.columns if c != "selling_price"]
    feature_specs = []
    one_hot_cols = set()
    for col in feature_cols:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        uniques = set(series.dropna().unique().tolist()) if is_numeric else set()
        if bool(uniques) and uniques.issubset({0, 1, 0.0, 1.0}):
            one_hot_cols.add(col)

    group_prefixes = [
        ("brand", "brand__"),
        ("model", "model__"),
        ("seller_type", "seller_type__"),
        ("fuel_type", "fuel_type__"),
        ("transmission_type", "transmission_type__"),
    ]
    group_specs = []
    grouped_cols = set()
    for name, prefix in group_prefixes:
        group_columns = sorted([c for c in one_hot_cols if c.startswith(prefix)])
        if group_columns:
            group_specs.append(
                {
                    "name": name,
                    "field_name": f"{name}_choice",
                    "prefix": prefix,
                    "columns": group_columns,
                }
            )
            grouped_cols.update(group_columns)

    for col in feature_cols:
        if col in grouped_cols:
            continue
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        uniques = set(series.dropna().unique().tolist()) if is_numeric else set()
        is_binary = bool(uniques) and uniques.issubset({0, 1, 0.0, 1.0})
        if is_binary:
            field_type = "binary"
        elif pd.api.types.is_integer_dtype(series):
            field_type = "int"
        else:
            field_type = "float"
        feature_specs.append((col, field_type))

    return feature_cols, feature_specs, group_specs, csv_path


def make_dynamic_form(feature_specs, group_specs):
    class DynamicForm(FlaskForm):
        pass

    for group in group_specs:
        choices = [("", "None")]
        for col in group["columns"]:
            label = col.replace(group["prefix"], "", 1).replace("_", " ").title()
            choices.append((col, label))
        label = group["name"].replace("_", " ").title()
        field = SelectField(label, choices=choices, default="")
        setattr(DynamicForm, group["field_name"], field)

    for col, field_type in feature_specs:
        label = col.replace("_", " ").title()
        if field_type == "binary":
            field = SelectField(
                label,
                choices=[("0", "No"), ("1", "Yes")],
                default="0",
            )
        elif field_type == "int":
            field = IntegerField(label, validators=[Optional()], default=0)
        else:
            field = FloatField(label, validators=[Optional()], default=0.0)
        setattr(DynamicForm, col, field)

    setattr(DynamicForm, "submit", SubmitField("Predict"))
    return DynamicForm


def load_samples(csv_path: str, feature_cols):
    df = pd.read_csv(csv_path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    samples = [
        {
            "id": "bmw_5",
            "label": "BMW 5",
            "filters": {"brand__BMW": 1, "model__5": 1},
        },
        {
            "id": "vw_polo",
            "label": "Volkswagen Polo",
            "filters": {"brand__Volkswagen": 1, "model__Polo": 1},
        },
        {
            "id": "mb_c_class",
            "label": "Mercedes-Benz C-Class",
            "filters": {"brand__Mercedes-Benz": 1, "model__C-Class": 1},
        },
        {
            "id": "ford_ecosport",
            "label": "Ford Ecosport",
            "filters": {"brand__Ford": 1, "model__Ecosport": 1},
        },
    ]
    options = []
    sample_map = {}
    for sample in samples:
        missing = [c for c in sample["filters"] if c not in df.columns]
        if missing:
            continue
        mask = pd.Series(True, index=df.index)
        for col, value in sample["filters"].items():
            mask &= df[col] == value
        rows = df[mask]
        if rows.empty:
            continue
        row = rows.iloc[0][feature_cols].to_dict()
        sample_map[sample["id"]] = row
        options.append({"id": sample["id"], "label": sample["label"]})
    return options, sample_map


def row_to_form_data(row_dict, feature_specs, group_specs):
    data = {}
    for group in group_specs:
        selected = ""
        for col in group["columns"]:
            value = row_dict.get(col, 0)
            if value == 1 or value == 1.0:
                selected = col
                break
        data[group["field_name"]] = selected

    for col, field_type in feature_specs:
        value = row_dict.get(col, 0)
        if field_type == "binary":
            data[col] = str(int(value)) if value in (0, 1, 0.0, 1.0) else "0"
        else:
            data[col] = value
    return data


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

MODEL = None
MODEL_ERROR = None
FEATURE_COLS = []
FEATURE_SPECS = []
GROUP_SPECS = []
SCHEMA_PATH = None
SAMPLE_OPTIONS = []
SAMPLE_MAP = {}
MODEL_FEATURES = []
MODEL_FEATURE_MISMATCH = None

try:
    FEATURE_COLS, FEATURE_SPECS, GROUP_SPECS, SCHEMA_PATH = load_schema(CSV_PATH)
except Exception as exc:
    MODEL_ERROR = f"Schema error: {exc}"
    GROUP_SPECS = []
else:
    try:
        MODEL = xgb.Booster()
        MODEL.load_model(MODEL_PATH)
    except Exception as exc:
        MODEL_ERROR = f"Model load error: {exc}"
    else:
        num_features = MODEL.num_features() if hasattr(MODEL, "num_features") else None
        if MODEL.feature_names:
            MODEL_FEATURES = list(MODEL.feature_names)
        elif num_features is not None:
            MODEL_FEATURES = FEATURE_COLS[:num_features]
            if num_features != len(FEATURE_COLS):
                MODEL_FEATURE_MISMATCH = (
                    f"Model expects {num_features} features, but CSV has {len(FEATURE_COLS)}."
                )
        else:
            MODEL_FEATURES = FEATURE_COLS.copy()

        if MODEL_FEATURES and len(MODEL_FEATURES) < len(FEATURE_COLS):
            allowed = set(MODEL_FEATURES)
            FEATURE_COLS = [c for c in FEATURE_COLS if c in allowed]
            FEATURE_SPECS = [(c, t) for c, t in FEATURE_SPECS if c in allowed]
            filtered_groups = []
            for group in GROUP_SPECS:
                cols = [c for c in group["columns"] if c in allowed]
                if cols:
                    filtered = dict(group)
                    filtered["columns"] = cols
                    filtered_groups.append(filtered)
            GROUP_SPECS = filtered_groups

    try:
        SAMPLE_OPTIONS, SAMPLE_MAP = load_samples(SCHEMA_PATH, FEATURE_COLS)
    except Exception:
        SAMPLE_OPTIONS, SAMPLE_MAP = [], {}


@app.route("/", methods=["GET", "POST"])
def index():
    DynamicForm = make_dynamic_form(FEATURE_SPECS, GROUP_SPECS)
    sample_id = request.args.get("sample")
    if sample_id in SAMPLE_MAP:
        prefill = row_to_form_data(SAMPLE_MAP[sample_id], FEATURE_SPECS, GROUP_SPECS)
        form = DynamicForm(data=prefill)
    else:
        form = DynamicForm()
    prediction = None

    if request.method == "POST":
        if MODEL_ERROR:
            flash(MODEL_ERROR, "error")
        elif form.validate_on_submit():
            row = {col: 0 for col in MODEL_FEATURES or FEATURE_COLS}
            for col, field_type in FEATURE_SPECS:
                field = getattr(form, col)
                value = field.data
                if field_type == "binary":
                    value = int(value) if value is not None else 0
                elif value is None:
                    value = 0
                if col in row:
                    row[col] = value

            for group in GROUP_SPECS:
                selected = getattr(form, group["field_name"]).data
                if selected in group["columns"]:
                    if selected in row:
                        row[selected] = 1

            X = pd.DataFrame([row], columns=MODEL_FEATURES or FEATURE_COLS)
            try:
                dmat = xgb.DMatrix(X, feature_names=MODEL_FEATURES or FEATURE_COLS)
                prediction = float(MODEL.predict(dmat)[0])
            except Exception as exc:
                flash(f"Prediction error: {exc}", "error")
        else:
            flash("Please correct the highlighted fields.", "error")

    return render_template(
        "index.html",
        form=form,
        prediction=prediction,
        model_error=MODEL_ERROR,
        schema_path=SCHEMA_PATH,
        model_path=MODEL_PATH,
        sample_options=SAMPLE_OPTIONS,
        current_sample_id=sample_id if sample_id in SAMPLE_MAP else "",
        feature_mismatch=MODEL_FEATURE_MISMATCH,
    )


@app.route("/reset")
def reset():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
