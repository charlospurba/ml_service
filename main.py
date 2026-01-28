from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Load model permanen
MODEL_PATH = "gmm_model.pkl"
CLUSTERS = ["Achievers", "Free Spirits", "Players", "Disruptors"]

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model GMM Loaded Successfully.")
else:
    print("CRITICAL ERROR: gmm_model.pkl not found!")

class UserFeatures(BaseModel):
    vector: list

@app.post("/predict")
async def predict(data: UserFeatures):
    X = np.array(data.vector).reshape(1, -1)
    
    # Menghitung probabilitas tiap klaster
    probs = model.predict_proba(X)[0]
    cluster_id = np.argmax(probs)
    
    return {
        "cluster": CLUSTERS[cluster_id],
        "confidence": float(np.max(probs)),
        "all_probabilities": probs.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)