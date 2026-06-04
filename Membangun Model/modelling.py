import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_experiment(experiment_name="anime_score_prediction")

df = pd.read_csv("anime_preprocessing (5).csv")

X = df.drop(columns=["score"])
y = df["score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

print(importance.sort_values(
    by="importance",
    ascending=False
).head(20))

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

print(importance_df.sort_values(
    by="importance",
    ascending=False
).head(20))