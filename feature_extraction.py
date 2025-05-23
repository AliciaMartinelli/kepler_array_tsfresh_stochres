import os
import numpy as np
import pandas as pd
from tsfresh import extract_features
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

RAW_DIR = "raw"
OUTPUT_DIR = "dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_lightcurve(lc, original_interval=30, target_interval=60):
    time = np.arange(0, len(lc) * original_interval, original_interval)
    interp_func = interp1d(time, lc, kind='linear', fill_value='extrapolate')
    resampled_time = np.arange(0, time[-1], target_interval)
    return interp_func(resampled_time)

def extract_tsfresh_features(raw_folder):
    feature_data = []
    labels = []

    for file in sorted(os.listdir(raw_folder)):
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(raw_folder, file)
        data = np.load(file_path, allow_pickle=True).item()

        if "lightcurve" not in data or "label" not in data:
            print(f"Skipping {file} (missing keys)")
            continue

        lc = data["lightcurve"]
        label = data["label"]
        global_lc = resample_lightcurve(lc[:2001])
        local_lc = resample_lightcurve(lc[2001:])

        df_global = pd.DataFrame({"id": [file]*len(global_lc), "time": range(len(global_lc)), "flux": global_lc})
        df_local  = pd.DataFrame({"id": [file]*len(local_lc),  "time": range(len(local_lc)),  "flux": local_lc})

        features_g = extract_features(df_global, column_id="id", column_sort="time", n_jobs=0)
        features_l = extract_features(df_local,  column_id="id", column_sort="time", n_jobs=0)

        combined = pd.concat([features_g, features_l], axis=1)
        feature_data.append(combined)
        labels.append(label)

    all_features = pd.concat(feature_data)
    all_features = all_features.interpolate(method='linear', axis=0).fillna(all_features.median()).fillna(0)

    const_features = all_features.columns[all_features.nunique() == 1]
    final_features = all_features.drop(columns=const_features)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(final_features)
    y = np.array(labels)

    np.save(os.path.join(OUTPUT_DIR, "X_kepler.npy"), X_scaled)
    np.save(os.path.join(OUTPUT_DIR, "y_kepler.npy"), y)

    print("DONE")
    print(f"Samples: {X_scaled.shape[0]} | Features: {X_scaled.shape[1]}")
    print(f" Removed {len(const_features)} constant features")

extract_tsfresh_features(RAW_DIR)