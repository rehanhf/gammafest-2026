#visualisasi heatmap korelasi antar fitur numerik.
#input: dataframe (df).
#output: plot heatmap dengan anotasi nilai korelasi.

import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap_korelasi(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                linewidths=0.5, linecolor='white')
    plt.title("Heatmap Korelasi antar Fitur")
    plt.tight_layout()
    plt.show()
