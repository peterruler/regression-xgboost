"""
Run instructions:
python -m venv .venv
source .venv/bin/activate  # mac/linux
pip install -r requirements.txt
export FLASK_APP=app.py
flask run --reload
"""

import os

import pandas as pd
import xgboost as xgb
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from wtforms import BooleanField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import Optional

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_ROOT, "PricePredictionCleanedUp.csv")
MODEL_PATH = os.path.join(APP_ROOT, "models", "xgboost_price_model.json")


def load_schema(csv_path: str):
    df = pd.read_csv(csv_path)
    if "selling_price" not in df.columns:
        raise ValueError("Expected selling_price column in CSV.")

    feature_cols = [c for c in df.columns if c != "selling_price"]
    feature_specs = []
    one_hot_cols = set()
    for col in feature_cols:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_binary = False
        if is_numeric:
            uniques = set(series.dropna().unique().tolist())
            if uniques and uniques.issubset({0, 1, 0.0, 1.0}):
                is_binary = True
        if is_binary:
            one_hot_cols.add(col)

    allowed_groups = {"brand", "model", "fuel", "seller", "transmission"}
    grouped_cols = set()
    group_specs = []
    for prefix in sorted(allowed_groups):
        group_columns = sorted(
            [c for c in one_hot_cols if c.startswith(f"{prefix}_")]
        )
        if len(group_columns) > 1:
            group_specs.append(
                {
                    "name": prefix,
                    "field_name": f"{prefix}_choice",
                    "columns": group_columns,
                }
            )
            grouped_cols.update(group_columns)

    for col in feature_cols:
        if col in grouped_cols:
            continue
        series = df[col]
        if col in one_hot_cols:
            field_type = "binary"
        elif pd.api.types.is_integer_dtype(series):
            field_type = "int"
        else:
            field_type = "float"
        feature_specs.append((col, field_type))
    return feature_cols, feature_specs, group_specs


def make_dynamic_form(feature_specs, group_specs):
    class DynamicForm(FlaskForm):
        pass

    for group in group_specs:
        choices = [("", "None")]
        prefix = f"{group['name']}_"
        for col in group["columns"]:
            label = col.replace(prefix, "", 1).replace("_", " ").title()
            choices.append((col, label))
        label = group["name"].replace("_", " ").title()
        field = SelectField(label, choices=choices, default="")
        setattr(DynamicForm, group["field_name"], field)

    for col, field_type in feature_specs:
        label = col.replace("_", " ").title()
        if field_type == "binary":
            field = BooleanField(label, default=False)
        elif field_type == "int":
            field = IntegerField(label, validators=[Optional()], default=0)
        else:
            field = FloatField(label, validators=[Optional()], default=0.0)
        setattr(DynamicForm, col, field)

    setattr(DynamicForm, "submit", SubmitField("Predict"))
    return DynamicForm


def load_samples(csv_path: str, feature_cols):
    df = pd.read_csv(csv_path)
    targets = [
        ("ferrari", "brand_Ferrari", "Ferrari"),
        ("audi", "brand_Audi", "Audi"),
        ("bmw", "brand_BMW", "BMW"),
        ("mercedes_benz", "brand_Mercedes-Benz", "Mercedes-Benz"),
        ("ford", "brand_Ford", "Ford"),
        ("maruti", "brand_Maruti", "Maruti"),
    ]
    options = []
    sample_map = {}
    for sample_id, col, label in targets:
        if col not in df.columns:
            continue
        rows = df[df[col] == 1]
        if rows.empty:
            continue
        row = rows.iloc[0][feature_cols].to_dict()
        sample_map[sample_id] = row
        options.append({"id": sample_id, "label": label})
        if len(options) >= 5:
            break
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
            data[col] = bool(value)
        else:
            data[col] = value
    return data


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

MODEL = None
MODEL_ERROR = None
SAMPLE_OPTIONS = []
SAMPLE_MAP = {}
DEFAULT_SAMPLE_ID = None
DEFAULT_SAMPLE_LABEL = None

try:
    FEATURE_COLS, FEATURE_SPECS, GROUP_SPECS = load_schema(CSV_PATH)
except Exception as exc:
    FEATURE_COLS, FEATURE_SPECS, GROUP_SPECS = [], [], []
    MODEL_ERROR = f"Schema error: {exc}"
else:
    try:
        MODEL = xgb.Booster()
        MODEL.load_model(MODEL_PATH)
    except Exception as exc:
        MODEL_ERROR = f"Model load error: {exc}"
    try:
        SAMPLE_OPTIONS, SAMPLE_MAP = load_samples(CSV_PATH, FEATURE_COLS)
        if "ferrari" in SAMPLE_MAP:
            DEFAULT_SAMPLE_ID = "ferrari"
            DEFAULT_SAMPLE_LABEL = "Ferrari sample"
        elif SAMPLE_OPTIONS:
            DEFAULT_SAMPLE_ID = SAMPLE_OPTIONS[0]["id"]
            DEFAULT_SAMPLE_LABEL = f"{SAMPLE_OPTIONS[0]['label']} sample"
    except Exception:
        SAMPLE_OPTIONS, SAMPLE_MAP = [], {}
        DEFAULT_SAMPLE_ID = None
        DEFAULT_SAMPLE_LABEL = None


@app.route("/", methods=["GET", "POST"])
def index():
    DynamicForm = make_dynamic_form(FEATURE_SPECS, GROUP_SPECS)
    sample_id = request.args.get("sample")
    current_sample_id = sample_id if sample_id in SAMPLE_MAP else None
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
            row = {col: 0 for col in FEATURE_COLS}
            for col, field_type in FEATURE_SPECS:
                field = getattr(form, col)
                value = field.data
                if field_type == "binary":
                    value = 1 if value else 0
                elif value is None:
                    value = 0
                row[col] = value

            for group in GROUP_SPECS:
                selected = getattr(form, group["field_name"]).data
                if selected in group["columns"]:
                    row[selected] = 1

            X = pd.DataFrame([row], columns=FEATURE_COLS)
            try:
                dmat = xgb.DMatrix(X, feature_names=FEATURE_COLS)
                prediction = float(MODEL.predict(dmat)[0])
            except Exception as exc:
                flash(f"Prediction error: {exc}", "error")
        else:
            flash("Please correct the highlighted fields.", "error")

    return render_template(
        "index.html",
        form=form,
        feature_cols=FEATURE_COLS,
        feature_specs=FEATURE_SPECS,
        group_specs=GROUP_SPECS,
        sample_options=SAMPLE_OPTIONS,
        default_sample_id=DEFAULT_SAMPLE_ID,
        default_sample_label=DEFAULT_SAMPLE_LABEL,
        current_sample_id=current_sample_id,
        prediction=prediction,
        model_error=MODEL_ERROR,
    )


@app.route("/reset")
def reset():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
