# mlflow with ml_models

1. First create a virtual environment and activate it:
 - To create: `python -m venv env_mlflow`

 - To activate:
   - for linux/mac: `source env_mlflow/bin/activate`
   - for windows: `env_mlflow/bin/activate`

2. Create a folder or git repository for the project and move to that folder:
 - `cd mlflow_project`

3. Install MLflow and additional dependencies:
 - `pip install mlflow[extras]`
 - check if installed properly: `mlflow --version`

4. Model training and hyperparameter tuning:

5. Run the model:
 - `python ml_model.py`

6. To check if the parameters and metrics are logged correctly, via the MLflow UI, run the following command in your terminal:
 - `mlflow ui --port 5000`
 - then visit http://localhost:5000 to open the UI.

7. Testing model serving locally:
 - `mlflow models serve -m ./mlruns/<runid of your model>/artifacts/model -p 1234 --no-conda`
 - Example: `mlflow models serve -m ./mlruns/701576440205981691/7eded846d14e471f962ddc4dc170ce44/artifacts/model -p 1234 --no-conda`
 