use PricePrediction.csv modify with pandas python library save to PricePredictionCleanedUp.csv in this md file add the steps that were performed to feature engineer the original csv file
Please perform feature engineering
 use uv to install pip packages
- Normalize and bucketsize numeric features
- create onehot encoding and embeddings for categorical features
- perform date and text features
- extract date- and time-related features from timestamp columns
- automl/PricePredictionCleanedUp.csv please optimize for google automl
- use 1 and 0 instead of the and false for one hot encoding 
- always update md file with the changes made
Steps performed:
- Loaded `automl/Price_Prediction.csv` with pandas.
- Identified numeric feature columns (excluding `selling_price`) and created z-score normalized versions with `_z` suffix.
- Bucketized numeric feature columns into 5 bins using quantiles (fallback to equal-width), with `_bucket` suffix.
- One-hot encoded categorical columns (`brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`) via `pd.get_dummies`.
- Added embedding-friendly categorical encodings: integer category ids (`*_cat_id`) and frequency encodings (`*_freq`).
- Added text features from `brand`, `model`, and combined `brand model` (character length and word count).
- Searched for date/time columns by name and extracted date parts when present; none were found in this dataset.
- Saved the engineered dataset to `automl/PricePredictionCleanedUp.csv`.
