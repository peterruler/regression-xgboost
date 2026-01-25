# regression-xgboost

🚗 **XGBoost-basierte Fahrzeugpreis-Vorhersage** mit Flask Web-Interface

- Generated with ChatGPT+ GPT-Codex-5.2 and VS Code Plugin
- Refined with Google Colab built-in Gemini3 AI

## 🌐 Demo

[Live Demo auf Render (ca. 2 Minuten Startzeit)](https://regression-xgboost.onrender.com)

---

## 📋 Projektbeschreibung

Dieses Projekt implementiert ein **Machine Learning Modell zur Vorhersage von Gebrauchtwagenpreisen** basierend auf verschiedenen Fahrzeugeigenschaften wie:

- **Marke & Modell** (Brand, Model)
- **Fahrzeugalter** (Vehicle Age)
- **Kilometerstand** (km driven)
- **Kraftstofftyp** (Fuel Type: Benzin, Diesel, etc.)
- **Getriebeart** (Transmission: Manuell, Automatik)
- **Motorleistung** (Engine, Max Power)
- **Verbrauch** (Mileage)
- **Sitzplätze** (Seats)

### Technologie-Stack

| Komponente | Technologie |
|------------|-------------|
| ML-Modell | XGBoost Regressor |
| Web-Framework | Flask |
| Frontend | Jinja2 Templates, WTForms |
| Datenverarbeitung | Pandas, NumPy, Scikit-learn |
| Visualisierung | Matplotlib, Seaborn |

---

## 🚀 Schnellstart

### Voraussetzungen

- Python 3.11+
- pip oder uv (empfohlen)

### Installation & Start

```bash
# Repository klonen
git clone https://github.com/your-username/regression-xgboost.git
cd regression-xgboost

# Virtuelle Umgebung erstellen
python -m venv .venv

# Umgebung aktivieren
source .venv/bin/activate  # macOS/Linux
# oder
.venv\Scripts\activate     # Windows

# Abhängigkeiten installieren (mit uv - schneller)
pip install uv
uv pip install -r requirements.txt

# ODER klassisch mit pip
pip install -r requirements.txt

# Flask-App starten
export FLASK_APP=app.py    # macOS/Linux
# set FLASK_APP=app.py     # Windows

flask run --reload
```

Die App ist dann erreichbar unter: **http://127.0.0.1:5000**

### Alternative: Mit Gunicorn (Produktion)

```bash
# gunicorn local standard port 8000 on render 5000 port open on production
gunicorn -w 4 -b 0.0.0.0:5000 app:app 
```

---

## 📁 Projektstruktur

```
regression-xgboost/
├── app.py                 # Flask Web-Anwendung
├── requirements.txt       # Python-Abhängigkeiten
├── models/
│   └── xgboost_price_model.json  # Trainiertes XGBoost-Modell
├── templates/
│   ├── base.html          # Basis-Template
│   └── index.html         # Hauptseite mit Formular
├── training/
│   ├── Cars.ipynb         # Jupyter Notebook für Training
│   ├── Cars.py            # Python-Skript für Training
│   ├── Price_Prediction.csv       # Rohdaten
│   ├── PricePredictionCleanedUp.csv  # Bereinigte Daten
│   └── SPEC.md            # Training-Spezifikation
└── legacy/                # Ältere Spezifikationen
```

---

## 🧠 Modell-Training

Das Modell kann mit dem Jupyter Notebook oder Python-Skript neu trainiert werden:

### Mit Jupyter Notebook (Google Colab kompatibel)

```bash
cd training
jupyter notebook Cars.ipynb
```

### Mit Python-Skript

```bash
cd training
python Cars.py
```

### Training-Metriken

Das Modell wird mit folgenden Regressionsmetriken evaluiert:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Bestimmtheitsmaß)

---

## ⚙️ Umgebungsvariablen

| Variable | Beschreibung | Standard |
|----------|--------------|----------|
| `FLASK_APP` | Flask-Anwendung | `app.py` |
| `CSV_PATH` | Pfad zur CSV-Datei | `training/PricePredictionCleanedUp.csv` |
| `MODEL_PATH` | Pfad zum Modell | `models/xgboost_price_model.json` |

---

## 📚 Weitere Dokumentation

- [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) - Feature Engineering Pipeline
- [FLASK_SERVER.md](FLASK_SERVER.md) - Flask Server Details
- [XG_REGRESSION.md](XG_REGRESSION.md) - XGBoost Regression Details
- [GOOGLE_COLAB.md](GOOGLE_COLAB.md) - Google Colab Anleitung
- [REQUIREMENTS.md](REQUIREMENTS.md) - Anforderungen

---

## 📄 Lizenz

MIT License