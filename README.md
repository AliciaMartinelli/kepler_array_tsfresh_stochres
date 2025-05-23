# Kepler - Stochastic Resonance - TSFresh + KNN Classification

This repository contains an experiment in which Gaussian noise is injected into statistical features extracted from Kepler light curves using TSFresh. The impact of this noise on classification performance is evaluated using a KNN classifier.

This experiment is part of the Bachelor's thesis **"Machine Learning for Exoplanet Detection: Investigating Feature Engineering Approaches and Stochastic Resonance Effects"** by Alicia Martinelli (2025).

## Folder Structure

```
kepler_array_tsfresh_stochres/
├── feature_extraction.py      # Extracts TSFresh features from the light curves in the raw folder and saves them in the dataset folder
├── data_aug.py                # Loads the X_kepler.npy and adds Gaussian noise with a noise intensitiy parameter sigma
├── calculate_snr.py           # Calculates the SNR
├── train_KNN.py               # KNN training
├── run_pipeline.sh            # Run the pipeline after feature extraction to get the X_kepler.npy with noise and train the KNN classifier per sigma
└── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Preprocessed Kepler dataset
The preprocessed Kepler dataset used in this project is based on the public release from Shallue & Vanderburg (2018) and is available via the AstroNet GitHub repository (Google Drive) [https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE](https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE)

Download the TFRecords from the Google Drive, convert them into .npy files and save them in the raw folder.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/AliciaMartinelli/kepler_array_tsfresh_stochres.git
    cd kepler_array_tsfresh_stochres
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    You may need to install `scikit-learn`, `tsfresh`, `matplotlib`, `numpy`, and `tensorflow` (and more).

## Usage

1. Extract the TSFresh features:
```bash
python feature_extraction.py
```
This extracts the TSFresh features from the light curves in the raw folder and saves the resulting X_kepler.npy and y_kepler.npy into the dataset folder.

2. Run the pipeline:
```bash
./run_pipeline.sh
```
Start the pipeline to create the X_kepler_sigma.npy feature vectors with added noise (aug_dataset) and train a KNN per sigma. Also the SNR will be calculated per sigma and saved in the results/snr folder.

3. Plot the AUC vs noise intensity parameter sigma:
```bash
python visualize_results.py
```
This will visualize the results in a plot with AUC vs. noise intensity parameter sigma

## Thesis Context

This repository corresponds to the experiment described in:
- **Section 6.1**: TSFresh feature noise injection and evaluation with KNN

**Author**: Alicia Martinelli  
**Email**: alicia.martinelli@stud.unibas.ch  
**Year**: 2025