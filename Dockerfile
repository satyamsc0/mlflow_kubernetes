
# Use Python base image
FROM --platform=linux/amd64 python:3.10

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt to container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MLFLOW app files to the container
COPY ml_model.py /app

# Add MLflow model and any other required artifacts to the Docker image
COPY ./mlruns/701576440205981691/7eded846d14e471f962ddc4dc170ce44/artifacts/model /app/model


# Expose port 1234 for MLFLOW app
EXPOSE 1234

# Command to start the MLFLOW app
CMD ["mlflow", "models", "serve", "-m", "./model", "-p", "1234", "--no-conda"]
