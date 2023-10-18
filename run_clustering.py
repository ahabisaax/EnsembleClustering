from Clustering_functions import ClusteringPipeline

if __name__ == '__main__':
    df, labels = ClusteringPipeline('data.csv',
                                    num_dims=10,
                                    num_models=128,
                                    k_means_n_clusters=20,
                                    batch_size=200,
                                    min_samples=10)
