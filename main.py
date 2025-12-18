from fastapi import FastAPI

app = FastAPI(title="ImagoAI API", description="FastAPI for ImagoAI ML Model Deployment")

@app.get("/")
async def root():
    return {"message": "Welcome to ImagoAI API. Use /docs to see the API documentation."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data: dict):
    # This is a placeholder for actual ML prediction logic
    return {"prediction": "This is a placeholder", "received_data": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
