import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.cluster.hierarchy import linkage, dendrogram

file_path = input("Please enter the CSV file path: ")
n_clusters = int(input("Please enter the number of clusters: "))

df = pd.read_csv(file_path)
df.set_index('Features', inplace=True)

df_transposed = df.T

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_transposed)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df_transposed['PCA1'] = pca_result[:, 0]
df_transposed['PCA2'] = pca_result[:, 1]

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df_transposed['Cluster'] = clusters

mean_passages = df_transposed.iloc[:, :-3].mean().round(2)
std_passages = df_transposed.iloc[:, :-3].std().round(2)

cv_passages = ((std_passages / mean_passages) * 100).round(2)

passage_significance_df = pd.DataFrame({
    'Mean': mean_passages,
    'Standard Deviation': std_passages,
    'Coefficient of Variation (%)': cv_passages
})

max_deviation_passage = passage_significance_df['Coefficient of Variation (%)'].idxmax()

cv_features = df.apply(lambda x: np.std(x) / np.mean(x) * 100, axis=1).round(2)

def find_max_deviation_passage(feature_name):
    feature_data = df.loc[feature_name]
    deviations = (feature_data - feature_data.mean()) / feature_data.std()
    max_deviation_index = np.argmax(np.abs(deviations))
    return df.columns[max_deviation_index]

feature_significance_df = pd.DataFrame({
    'Feature': df.index,
    'Coefficient of Variation (%)': cv_features,
    'Passage with Highest Deviation': df.index.map(find_max_deviation_passage)
}).reset_index(drop=True)

print("\nPassage Significance (Mean, Standard Deviation, and Coefficient of Variation):")
print(tabulate(passage_significance_df, headers='keys', tablefmt='pretty'))

print(f"\nPassage with the highest deviation: {max_deviation_passage}")

print("\nFeature Significance (Coefficient of Variation and Passage with Highest Deviation):")
print(tabulate(feature_significance_df, headers='keys', tablefmt='pretty'))

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_transposed, palette='Set1')

plt.title('PCA and K-means Clustering')
plt.legend()
plt.show()

correlation_matrix = df_transposed.iloc[:, :-3].corr().round(2)

linkage_matrix = linkage(correlation_matrix, method='ward')
dendro = dendrogram(linkage_matrix, labels=correlation_matrix.columns, no_plot=True)
ordered_columns = dendro['ivl']

reordered_corr_matrix = correlation_matrix.loc[ordered_columns, ordered_columns]

plt.figure(figsize=(10, 8))
sns.heatmap(reordered_corr_matrix, annot=False, cmap='coolwarm', cbar=False)
plt.title('Correlation Matrix')

for i, feature in enumerate(reordered_corr_matrix.columns):
    plt.text(len(reordered_corr_matrix) + 0.5, i + 0.5, feature, va='center', ha='left', fontsize=10, color='black')

plt.show()
