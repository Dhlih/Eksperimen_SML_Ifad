import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mengatur repository dagshub
dagshub.init(
    repo_owner="Dhlih",
    repo_name="anime-score-prediction",
    mlflow=True
)

# Konfigurasi MLflow
mlflow.set_experiment("anime_score_prediction")

# Load dataset
df = pd.read_csv("anime_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["score"])
y = df["score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Hyperparameter yang akan dicoba
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Random Search
search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=3,       
    cv=3,
    scoring="r2",
    n_jobs=5,
    random_state=42
)

with mlflow.start_run(run_name="RF_RandomSearch"):
    # Training + Hyperparameter Tuning
    search.fit(X_train, y_train)

    # Ambil model terbaik
    best_model = search.best_estimator_

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Log parameter 
    mlflow.log_params(search.best_params_)
    mlflow.log_param("estimator_class", "sklearn.ensemble._forest.RandomForestRegressor")
    mlflow.log_param("search_estimator_class", "sklearn.model_selection._search.RandomizedSearchCV")
    mlflow.log_param("cv", 3)
    mlflow.log_param("n_iter", 3)
    mlflow.log_param("scoring", "r2")

    # Evaluasi pada Data Training 
    y_train_pred = best_model.predict(X_train)
    training_mae = mean_absolute_error(y_train, y_train_pred)
    training_mse = mean_squared_error(y_train, y_train_pred)
    training_rmse = training_mse ** 0.5
    training_r2 = r2_score(y_train, y_train_pred)

    # Evaluasi pada Data Testing 
    y_test_pred = best_model.predict(X_test)
    testing_mae = mean_absolute_error(y_test, y_test_pred)
    testing_mse = mean_squared_error(y_test, y_test_pred)
    testing_rmse = testing_mse ** 0.5
    testing_r2 = r2_score(y_test, y_test_pred)

    # Metrik CV Search
    mlflow.log_metric("best_cv_score", search.best_score_)

    # Kelompok Metrik Training (Kloning Autolog)
    mlflow.log_metric("training_mae", training_mae)
    mlflow.log_metric("training_mse", training_mse)
    mlflow.log_metric("training_rmse", training_rmse)
    mlflow.log_metric("training_r2_score", training_r2)
    
    # Kelompok Metrik Testing (Nilai Tambah Eksklusif)
    mlflow.log_metric("testing_mae", testing_mae)
    mlflow.log_metric("testing_mse", testing_mse)
    mlflow.log_metric("testing_rmse", testing_rmse)
    mlflow.log_metric("testing_r2_score", testing_r2)

    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")

    # Log dataset
    mlflow.log_artifact("anime_preprocessing.csv")

    # Log model terbaik
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # Log prediksi vs aktual
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig("prediction_vs_actual.png")
    plt.close()

    mlflow.log_artifact("prediction_vs_actual.png")

    with open("best_params.txt", "w") as f:
        f.write(str(search.best_params_))

    mlflow.log_artifact("best_params.txt")

    print("Best Params :", search.best_params_)
    print("Best CV R²  :", search.best_score_)
    print("MAE         :", mae)
    print("MSE         :", mse)
    print("RMSE        :", rmse)
    print("R² Test     :", r2)

    print("Proses training dan manual logging ke DagsHub berhasil!")