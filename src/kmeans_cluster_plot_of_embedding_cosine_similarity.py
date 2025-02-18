import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Define file paths
file_path = "/Users/carlstahmer/Desktop/plosone_embeddings.csv"
outfile = "/Users/carlstahmer/Desktop/plos_kmeans_cluster_embeddings.png"

# Define number of embeddings features
num_features = 1024

# Read DataFrame with the labels as both index and columns.
corpus_df = pd.read_csv(file_path, dtype={'index': str}, index_col=0)

# Assign the index to a label vector for ease of use in plotting
labels = corpus_df.index

# Write a function to process the contents of the ebeddings colum
# of the dataframe and convert it into a vector
def vecrtorize_embeddings(embed_string):
    cleaned_string = embed_string.replace("[", "")
    cleaned_string = cleaned_string.replace("]", "")
    embedding_vector = cleaned_string.split(",")
    return embedding_vector

# Call our vectorizing function on each row of the loaded data.
# Will return a list of lists
embeddings_list = list(map(vecrtorize_embeddings, corpus_df['embeddings']))

# Convert the returned list of list to a dataframe
embeddings_df = pd.DataFrame(embeddings_list)

# Add index
embeddings_df.index = corpus_df.index

# Add column names
embeddings_df.columns = range(1, embeddings_df.shape[1] + 1)

# Set the number of clusters that we want
n_clusters = 10

# Create a k-means object
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit k-means on the rows of embeddings matrix.
clusters = kmeans.fit_predict(embeddings_df.values)

# Optionally, add the cluster labels to the corpus DataFrame 
# (if you want to inspect them later).
corpus_df['cluster'] = clusters

# -------------------------------
# Dimensionality Reduction for Visualization
# -------------------------------
# Since our data lives in a high-dimensional space (each observation is a vector of similarities),
# we reduce it to 2 dimensions using PCA for plotting.
pca = PCA(n_components=2)
points_2d = pca.fit_transform(embeddings_df.values)

# -------------------------------
# Plot the Clustering Results with Observation Labels
# -------------------------------
plt.figure(figsize=(60, 40))

# Plot each cluster in a different color
for cluster in np.unique(clusters):
    idx = clusters == cluster
    plt.scatter(points_2d[idx, 0], points_2d[idx, 1],
                label=f'Cluster {cluster}', s=100, alpha=0.7)
    # Annotate each point with its corresponding observation label
    for i in np.where(idx)[0]:
        plt.text(points_2d[i, 0] + 0.01, points_2d[i, 1] + 0.01, labels[i],
                 fontsize=9, fontweight='bold')

plt.title('K-means Clustering on Text Vectors')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.savefig(outfile, dpi=300, bbox_inches='tight')