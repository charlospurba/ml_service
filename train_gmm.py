import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

def generate_simulated_data():
    # Menggunakan standar deviasi (scale) 0.3 agar klaster lebih halus dan beririsan
    # Mean diturunkan ke 0.6 agar tidak terlalu ekstrem
    achievers = np.random.normal(loc=[0.6, 0.3, 0.3, 0.2], scale=0.3, size=(200, 4))
    free_spirits = np.random.normal(loc=[0.3, 0.6, 0.3, 0.2], scale=0.3, size=(200, 4))
    players = np.random.normal(loc=[0.3, 0.3, 0.6, 0.2], scale=0.3, size=(200, 4))
    disruptors = np.random.normal(loc=[0.2, 0.2, 0.2, 0.7], scale=0.3, size=(200, 4))
    
    return np.vstack([achievers, free_spirits, players, disruptors])

print("Melatih ulang model GMM dengan klaster yang lebih stabil...")
X_train = np.clip(generate_simulated_data(), 0, 1)

# Menggunakan 4 komponen klaster
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_train)

with open('gmm_model.pkl', 'wb') as f:
    pickle.dump(gmm, f)

print("Selesai! Model lebih stabil telah disimpan.")