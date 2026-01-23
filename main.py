from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import os

app = FastAPI()

# Definisi nama klaster berdasarkan profil gamifikasi
CLUSTERS = ["Achievers", "Free Spirits", "Players", "Disruptors"]

class UserFeatures(BaseModel):
    vector: list

# Endpoint untuk memprediksi klaster user
@app.post("/predict")
async def predict(data: UserFeatures):
    # Data input: [0.18, 0.1, 0.15, 0]
    X = np.array(data.vector).reshape(1, -1)
    
    # Catatan: Dalam sistem produksi, model GMM harus dilatih (fit) terlebih dahulu
    # Untuk keperluan awal, kita gunakan bobot inisialisasi
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    
    # Dummy fit agar model bisa berjalan (Gunakan data simulasi nantinya)
    dummy_data = np.random.rand(10, 4)
    gmm.fit(dummy_data)
    
    # Prediksi klaster
    cluster_id = gmm.predict(X)[0]
    probs = gmm.predict_proba(X)[0] # Confidence score
    
    return {
        "cluster": CLUSTERS[cluster_id],
        "confidence": float(np.max(probs)),
        "all_probabilities": probs.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)