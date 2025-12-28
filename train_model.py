import pandas as pd
import joblib
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split

# load dataset
dataframe = pd.read_csv(
    r"C:\Users\sumit\OneDrive\Documents\SampleSuperstore.csv",
    encoding="latin1"
)

inputs = dataframe[["Sales", "Quantity", "Discount"]]
output = dataframe["Profit"]
#split the dataset for the train and test the model

x_train, x_test, y_train, y_test = train_test_split(
    inputs, output, train_size=0.8, random_state=42
)

# train model
model =  XGBRFRegressor(tree_method="hist",        # ✅ CPU ONLY
    predictor="cpu_predictor",
    device="cpu",              # ✅ important for newer versions
    random_state=42)
model.fit(x_train, y_train)

# SAVE MODEL (THIS CREATES THE FILE)
joblib.dump(model, "profit_model.pkl")

print("✅ profit_model.pkl created")
