from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from src.feature_engineering import extract_features
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def train_model(df, seq_col, target_col):
    """
    Train CatBoost regressor on sgRNA dataset.
    """

    sequences = df[seq_col].tolist()
    y = df[target_col].values

    # Feature extraction
    X = extract_features(sequences)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # CatBoost model
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="RMSE",
        verbose=100
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Spearman correlation
    spearman_score = spearmanr(y_test, y_pred).correlation

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    pearson = pearsonr(y_test, y_pred)[0]


    print("\n✅ Model Training Complete")

    model.save_model("models/catboost_crispr.cbm")
    print("✅ Model saved in models/catboost_crispr.cbm")


    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Pearson Correlation:", pearson)
    print("Spearman Correlation:", spearman_score)

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Efficiency")
    plt.ylabel("Predicted Efficiency")
    plt.title("True vs Predicted sgRNA Activity")
    plt.show()

    importances = model.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(importances[:10])

    return model
