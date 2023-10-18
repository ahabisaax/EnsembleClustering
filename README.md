# EnsembleClustering
Use Ensemble Clustering to cluster 499 headlines and subheadings. We take a dataframe of shor pieces of text and produces a series of labels where a label is the assigned cluster of a given piece of text. 
This is split into two steps - we produce a latent representation of the concatenated headline and subheading via a pre-trained transformer model all-MiniLM-L6-v2. 
We then cluster these texts using an ensemble of weak learners (MiniBatchKmeans) and use an aggregating clustering model (OPTICS but this could be HDBSCAN or DBSCAN e.t.c.) 
that takes in to the voting similarity matrix from the weak learners and produces a final clustering decision. No evaluation for time but could use one of:
- Silhouette Score
- Davies-Bouldin Index
- Dunn Index
- Calinski-Harabasz Index (Variance Ratio Criterion)
- Within-Cluster Sum of Squares (WCSS)
- Xie-Beni Index
