import os
import pandas as pd
import matplotlib.pyplot as plt

size = 20
plt.rcParams.update({
    "font.size": size,
    "axes.titlesize": size,
    "axes.labelsize": size,
    "xtick.labelsize": size,
    "ytick.labelsize": size,
    "legend.fontsize": 15
})

sigmas = [0.00001, 0.0001, 0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.1, 0.3, 1]

def load_metrics_knn():
    result_dir = os.path.join("results", "knn")
    rows = []

    for sigma in sigmas:
        filename = f"results_knn_sigma_{sigma:.5f}.csv"
        filepath = os.path.join(result_dir, filename)

        if not os.path.exists(filepath):
            print(f"warning: missing file {filepath}")
            continue

        df = pd.read_csv(filepath)
        df["sigma"] = sigma
        rows.append(df)

    if not rows:
        raise RuntimeError("no KNN data found.")

    return pd.concat(rows, ignore_index=True)

def load_snr_data():
    result_dir = os.path.join("results", "snr")
    snr_rows = []

    for sigma in sigmas:
        filename = f"snr_sigma_{sigma:.5f}.csv"
        filepath = os.path.join(result_dir, filename)

        if not os.path.exists(filepath):
            print(f"warning: missing snr file {filepath}")
            continue

        df = pd.read_csv(filepath)
        df["sigma"] = sigma
        snr_rows.append(df)

    if not snr_rows:
        raise RuntimeError("no SNR data found.")

    return pd.concat(snr_rows, ignore_index=True)

def summarize(df, value_col):
    return df.groupby("sigma")[value_col].agg(["mean", "std"]).reset_index()

def plot_auc_and_snr(auc_df, snr_df):
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        auc_df["sigma"], auc_df["mean"], yerr=auc_df["std"],
        label="AUC (KNN)", marker='o', capsize=4, color='tab:blue'
    )

    ax1 = plt.gca()
    ax1.set_xscale("log")
    ax1.set_xlabel(r"noise intensity parameter ($\sigma$)")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.5, 1.0)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    snr_summary = snr_df.groupby("sigma")["snr"].agg(["mean", "std"]).reset_index()
    ax2.errorbar(
        snr_summary["sigma"], snr_summary["mean"], yerr=snr_summary["std"],
        label="SNR", marker='s', color='tab:red', capsize=4
    )
    ax2.set_ylabel("SNR (dB)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower left')

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/auc_snr_knn_vs_sigma.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Loading KNN results and SNR values...")
    df_knn = summarize(load_metrics_knn(), "auc")
    df_snr = load_snr_data()

    print("Generating plot...")
    plot_auc_and_snr(df_knn, df_snr)
    print("Plot saved to results/plots/auc_snr_knn_vs_sigma.png")
