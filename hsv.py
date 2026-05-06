import cv2
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import joblib

# Paths
bleached_folder = r"C:\Users\MRIDULA\Downloads\bleached"
healthy_folder = r"C:\Users\MRIDULA\Downloads\unbleached"

def load_hsv_samples(folder, max_samples=50000):
    hsv_pixels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv = hsv.reshape(-1, 3)
            hsv_pixels.append(hsv)
    all_pixels = np.vstack(hsv_pixels)
    if len(all_pixels) > max_samples:
        idx = np.random.choice(len(all_pixels), max_samples, replace=False)
        all_pixels = all_pixels[idx]
    return all_pixels

bleached_hsv = load_hsv_samples(bleached_folder)
healthy_hsv = load_hsv_samples(healthy_folder)

# Gaussian Mixture Models
bleached_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
healthy_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)

bleached_gmm.fit(bleached_hsv)
healthy_gmm.fit(healthy_hsv)

# Save models
joblib.dump(bleached_gmm, "bleached_gmm.pkl")
joblib.dump(healthy_gmm, "healthy_gmm.pkl")

print(" HSV model training completed and saved.")
