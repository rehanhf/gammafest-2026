#menghitung dan visualisasi distribusi gol menggunakan IQR.
#input: series (pandas Series), label (nama kolom).
#output: print outlier bounds dan plot distribusi.

import matplotlib.pyplot as plt

def deteksi_outlier_iqr(series, label):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    print(f"{label} - lower_bound: {lower_bound}, upper_bound: {upper_bound}")
    print(f"jumlah outlier: {len(outliers)}")

    plt.figure(figsize=(8, 4))
    series.plot(kind='hist', bins=30, edgecolor='black', alpha=0.7)
    plt.title(f"Distribusi {label}")
    plt.xlabel(label)
    plt.ylabel("frekuensi")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers)
    }
