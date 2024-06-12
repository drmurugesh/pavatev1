library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(pheatmap)
library(cluster)
library(factoextra)

file_path <- readline(prompt="Please enter the CSV file path: ")
n_clusters <- as.integer(readline(prompt="Please enter the number of clusters: "))

df <- read_csv(file_path)

df <- df %>% column_to_rownames("Features")

df_transposed <- t(df)

scaled_data <- scale(df_transposed)

pca <- prcomp(scaled_data, scale. = TRUE)
pca_result <- data.frame(pca$x[, 1:2])
df_transposed <- cbind(df_transposed, PCA1 = pca_result$PC1, PCA2 = pca_result$PC2)

kmeans_result <- kmeans(scaled_data, centers = n_clusters, nstart = 25)
df_transposed <- cbind(df_transposed, Cluster = as.factor(kmeans_result$cluster))

mean_passages <- colMeans(df_transposed[, -((ncol(df_transposed)-2):ncol(df_transposed))]) %>% round(2)
std_passages <- apply(df_transposed[, -((ncol(df_transposed)-2):ncol(df_transposed))], 2, sd) %>% round(2)

cv_passages <- ((std_passages / mean_passages) * 100) %>% round(2)

passage_significance_df <- data.frame(
  Mean = mean_passages,
  Standard_Deviation = std_passages,
  Coefficient_of_Variation = cv_passages
)

max_deviation_passage <- names(which.max(passage_significance_df$Coefficient_of_Variation))

cv_features <- apply(df, 1, function(x) (sd(x) / mean(x) * 100)) %>% round(2)

find_max_deviation_passage <- function(feature_name) {
  feature_data <- df[feature_name, ]
  deviations <- (feature_data - mean(feature_data)) / sd(feature_data)
  max_deviation_index <- which.max(abs(deviations))
  colnames(df)[max_deviation_index]
}

feature_significance_df <- data.frame(
  Feature = rownames(df),
  Coefficient_of_Variation = cv_features,
  Passage_with_Highest_Deviation = sapply(rownames(df), find_max_deviation_passage)
)

print("\nPassage Significance (Mean, Standard Deviation, and Coefficient of Variation):")
print(passage_significance_df)

cat("\nPassage with the highest deviation:", max_deviation_passage, "\n")

print("\nFeature Significance (Coefficient of Variation and Passage with Highest Deviation):")
print(feature_significance_df)

pca_plot <- ggplot(df_transposed, aes(x = PCA1, y = PCA2, color = Cluster)) +
  geom_point() +
  labs(title = "PCA and K-means Clustering") +
  theme_minimal()

print(pca_plot)

correlation_matrix <- cor(df_transposed[, -((ncol(df_transposed)-2):ncol(df_transposed))])

dist_matrix <- dist(1 - correlation_matrix)
hclust_result <- hclust(dist_matrix, method = "ward.D2")
ordered_columns <- colnames(correlation_matrix)[hclust_result$order]

reordered_corr_matrix <- correlation_matrix[ordered_columns, ordered_columns]

corr_plot <- pheatmap(reordered_corr_matrix, cluster_rows = F, cluster_cols = F, display_numbers = FALSE)

print(corr_plot)
