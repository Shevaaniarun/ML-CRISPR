from catboost import CatBoostRegressor
from src.feature_engineering import extract_features

model = CatBoostRegressor()
model.load_model("models/catboost_crispr.cbm")

seq = input("Enter sgRNA sequence (20nt): ")

X = extract_features([seq])
pred = model.predict(X)

print("Predicted Efficiency Score:", pred[0])
