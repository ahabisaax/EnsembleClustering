import pandas as pd
import numpy as np 
import sklearn.cluster
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering, HDBSCAN, OPTICS, DBSCAN 
from ensemble_clustering import ClusterSimilarityMatrix, EnsembleCustering
from sklearn.decomposition import PCA 
from typing import List, Union, Tuple
from sklearn.metrics import silhouette_score

#We will concatenate the headline and the short description as the text to embed
def concat_headline_and_subheading(df:pd.DataFrame) -> pd.DataFrame:
    concat  = []
    #make sure we account for np.nan cells
    for idx, row in df.iterrows():
        description = row['short_description'] if not pd.isna(row['short_description']) else ''
        heading = row['headline'] if not pd.isna(row['headline']) else ''
        concat.append(heading + '. ' + description)
    
    df['combined'] = concat
    return df


# embed using MiniLM would rather use ada or better model
def embed_text(df:pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(list(df['combined']), show_progress_bar=True)
    print(f'embedding shape: {embeddings.shape}')
    np.save('embeddings.npy', embeddings)
    print(f'embeddings saved as emeddings.npy')
    return df, embeddings


def dimensionality_reduction(num_dims:int, embeddings:np.ndarray) -> np.ndarray:
    pca = PCA(n_components = num_dims)
    pca.fit(embeddings)
    data_pca = pca.transform(embeddings)
    print(f'reduced dimensionality from {embeddings.shape[1]} to {data_pca.shape[1]}')
    return data_pca


def cluster_text(df: pd.DataFrame,
        embeddings:np.ndarray,
                num_models:int, 
                k_means_n_clusters:int, 
                batch_size:int, 
                min_samples: int) -> Tuple[pd.DataFrame, np.ndarray]:
    

    clustering_models = num_models*[
        # Note: Do not set a random_state, as the variability is crucial
        MiniBatchKMeans(n_clusters=k_means_n_clusters, batch_size=200, n_init=1, max_iter=20)
    ]
    aggregator_clt =  OPTICS(min_samples=min_samples)
    ens_clt = EnsembleCustering(clustering_models, aggregator_clt)
    y_ensemble, cluster_matrix = ens_clt.fit_predict(embeddings)
    
    df['labels'] = y_ensemble
    return df, y_ensemble


def ClusteringPipeline(filename:str,
                    num_dims:int,
                    num_models:int,
                    k_means_n_clusters:int,
                    batch_size:int,
                    min_samples:int) -> Tuple[pd.DataFrame, List]:

    df = pd.read_csv(filename)
    df = concat_headline_and_subheading(df=df)
    df, embeddings = embed_text(df=df)
    data_pca = dimensionality_reduction(num_dims=num_dims, embeddings=embeddings)
    df, labels = cluster_text(df,
                embeddings=data_pca,
                num_models=num_models, 
                k_means_n_clusters=k_means_n_clusters, 
                batch_size=batch_size, 
                min_samples=min_samples)
    return df, labels




