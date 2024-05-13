import mlflow
import numpy as np
from sklearn import datasets, metrics
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV


def eval_metrics(pred, actual):
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)
    return rmse, mae, r2


# Set th experiment name
mlflow.set_experiment("wine-quality")

# Enable auto-logging to MLflow
mlflow.sklearn.autolog()

# Load wine quality dataset
X, y = datasets.load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


lr = ElasticNet()

# Define distribution to pick parameter values from
distributions = dict(
    alpha=uniform(loc=0, scale=10),  # sample alpha uniformly from [-5.0, 5.0]
    l1_ratio=uniform(),  # sample l1_ratio uniformlyfrom [0, 1.0]
)

# Initialize random search instance
clf = RandomizedSearchCV(
    estimator=lr,
    param_distributions=distributions,
    # Optimize for mean absolute error
    scoring="neg_mean_absolute_error",
    # Use 5-fold cross validation
    cv=5,
    # Try 100 samples. Note that MLflow only logs the top 5 runs.
    n_iter=100,
)

# Start a parent run
with mlflow.start_run(run_name="wine-model"):
    search = clf.fit(X_train, y_train)

    # Evaluate the best model on test dataset
    y_pred = clf.best_estimator_.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_pred, y_test)
    mlflow.log_metrics(
        {
            "mean_squared_error_X_test": rmse,
            "mean_absolute_error_X_test": mae,
            "r2_score_X_test": r2,
        }
    )

# mlflow models serve -m runs:/<run_id_for_your_best_run>/model -p 1234 --enable-mlserver
# mlflow models serve -m runs:/7eded846d14e471f962ddc4dc170ce44/model -p 1234 --enable-mlserver
#               or
# mlflow models serve -m ./mlruns/701576440205981691/7eded846d14e471f962ddc4dc170ce44/artifacts/model -p 1234 --no-conda
# Request for prediction using following
# curl -X POST -H "Content-Type:application/json" --data '{"inputs": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]}' http://127.0.0.1:1234/invocations
# {"predictions": [-0.05977032080946154]}


# mlflow models build-docker -m runs:/7eded846d14e471f962ddc4dc170ce44/model -n satyamsc0/mlflow-wine-classifier --enable-mlserver
# mlflow models build-docker -m ./mlruns/701576440205981691/7eded846d14e471f962ddc4dc170ce44/artifacts/model -n satyamsc0/mlflow-wine-classifier --no-conda

# docker run -p 1234:1234 wine-classification

# mlflow models serve -m "models:/wine-quality@wine_quality"