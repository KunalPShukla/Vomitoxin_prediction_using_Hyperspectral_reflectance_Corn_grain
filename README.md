# Vomitoxin prediction using Hyperspectral for reflectance for Corn grain

This project provides a FastAPI-based REST API for deploying machine learning models. It is designed to be easily containerized and deployed to AWS.

## Project Structure

- `main.py`: The entry point of the FastAPI application.
- `Dockerfile`: Configuration for building the Docker image.
- `requirements.txt`: Python dependencies.
- `venv/`: Virtual environment (should be excluded from git).

## Setup Instructions

### 1. Local Development

1.  **Create and Activate Virtual Environment:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Run Locally:**
    ```powershell
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

### 2. Docker Setup

1.  **Build the Docker Image:**
    ```bash
    docker build -t imagoai-api .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -p 8000:8000 imagoai-api
    ```

## AWS Deployment Plan

This application can be deployed to AWS using:
- **AWS App Runner:** (Simplest) Connect this GitHub repository directly to App Runner.
- **AWS ECS (Elastic Container Service):** Deploy the Docker container to ECS with Fargate.
- **AWS Lambda:** Using a wrapper like `mangum`.

## API Endpoints

- `GET /`: Welcome message.
- `GET /health`: Health check for deployment monitoring.
- `POST /predict`: Endpoint for receiving data and returning predictions.
