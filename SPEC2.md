with XCboost du a regression ot the PricePredictionCleanedUp.csv
create a jupiter notebook if possible to perform al the action and a Cars.py file.
use uv package manager to install xcboost in an newly created conda environment.
Then perform a regression with target column the selling_price
use seaborn and matpoltlib to visualize scatterplot the data, show training error and training accuracy and confusion matrix on training data etc. use sklearn to train test split the data

Actions performed:
- Created `automl/Cars.py` to load `PricePredictionCleanedUp.csv`, train an XGBoost regressor with `selling_price` as target, and report MAE/RMSE/R2.
- Added seaborn/matplotlib plots: actual vs predicted scatter, training error/accuracy bar chart, and a binned confusion matrix saved to `automl/outputs/`.
- Created `automl/Cars.ipynb` with the same workflow plus setup instructions using conda + `uv pip install`.
