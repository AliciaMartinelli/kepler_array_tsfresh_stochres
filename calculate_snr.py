import os
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python calculate_snr.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

X_original_path = "dataset/X_kepler.npy"
X_augmented_path = os.path.join("aug_dataset", f"X_kepler_augmented_sigma_{sigma:.5f}.npy")
output_path = os.path.join("results", "snr", f"snr_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

X_original = np.load(X_original_path)
X_augmented = np.load(X_augmented_path)

if X_original.shape != X_augmented.shape:
    print("Shape mismatch between original and augmented data!")
    sys.exit(1)

signal_var = np.var(X_original, axis=1)
noise_var = np.var(X_augmented - X_original, axis=1)
snr_values = 10 * np.log10(signal_var / noise_var)
snr_values = np.where(noise_var == 0, np.inf, snr_values)

df = pd.DataFrame({
    "sample_index": np.arange(len(snr_values)),
    "snr": snr_values
})

df.to_csv(output_path, index=False)
print(f"\nSNR results saved to {output_path}")
print(f"Mean SNR: {np.mean(snr_values):.2f} dB | Std: {np.std(snr_values):.2f} dB")
