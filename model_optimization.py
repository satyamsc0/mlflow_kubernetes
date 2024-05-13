import mlflow
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from model_train import eval_metrics, X_train, X_test, y_train, y_test

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
with mlflow.start_run(run_name="hyperparameter-tuning"):
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
