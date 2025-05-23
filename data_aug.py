import sys
import os
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python data_aug.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

X = np.load("dataset/X_kepler.npy")


np.random.seed(42)
noise = np.random.normal(0, sigma, size=X.shape)
X_noisy = X + sigma * noise

save_path = f"aug_dataset/X_kepler_augmented_sigma_{sigma:.5f}.npy"
np.save(save_path, X_noisy)

print(f"Augmented data saved to: {save_path}")
print(f"Shape: {X.shape} | Noise σ={sigma}")
