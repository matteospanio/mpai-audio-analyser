"""
.. _equalization_clustering:

Cluster analysis
=========================

In this notebook is described the clusterization process of datasets identified in :numref:`clustering-datasets`. The goal is to find out if there is any evident difference in the audio tapes acquired with different equalizations.

Looking in details, it will be tested the capacity of K-Means and hierarchical clustering to find out if:

- there is an evident difference in audio tapes acquired with different equalizations for tapes recorded and reproduced at a speed of 7.5 ips;
- there is an evident difference in tapes acquired with different equalizations for tapes recorded and reproduced at a speed of 15 ips;
- there is a difference between audio tapes recorded and acquired at different velocities among all possibilities (3.75 ips, 7.5 ips, 15 ips) and among the most common ones (7.5, 15)
"""
# %%
# Since this analysis is gonig to use the K-Means algorithm, which starts pointing to random numbers, for reproducibility purposes, the seed of the random number generator is set at :math:`42`.
SEED = 42

# %%
# To make sure that the best results are obtained the algorithm's parameters are going to *tuned*, but, since there is no need to cross validate the results neither to split the dataset in the training of the model, the :func:`ml.clusters.validate` function will be used. This function takes a model, a dataset, a set of parameters and a scoring function and returns the best parameters and the score obtained with them.
#
# To make a decision over the quality of a clustering result, between all possible metrics, the :py:func:`sklearn.metrics.v_measure_score` is the one that best suits our needs. It is a function that measures the homogeneity of the clusters and the completeness of the clustering (go to :ref:`cluster-scoring` for a more detailed description of *V-measure*).
#
# In the following code snippet is presented a standard workflow to tune the parameters of a clustering algorithm. The dataset used is the **A** from :numref:`clustering-datasets`.
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from ml.clusters import validate
from ml.datasets import load_dataset

# load all possible combinations of 7.5 ips in CCIR and NAB
# with noise type C from Pretto's dataset
# e. g. ['7C_7C', '7C_7N', '7N_7C', '7N_7N']
data, y, n_clust = load_dataset('A', noise_type='C', return_X_y=True)
data.info()
# %%
# Since the dataset contains the noise type that is of type ``object``, it is necessary to drop it from the dataset before applying the clustering algorithm.
data = data.drop('noise_type', axis=1)

# define the parameters to tune
k_params = {
    'algorithm': ['lloyd', 'elkan'],
    'init': ['k-means++', 'random'],
    'max_iter': [10, 100, 300, 400],
}

parameters, score = validate(
    KMeans(n_clusters=n_clust, n_init='auto', random_state=SEED), data, y,
    k_params, v_measure_score)

parameters, score
# %%
# A good idea to automatize the process of execution the previous workflow over all possible datasets indivituated is to wrap it up in a function: ``evaluate_kmeans``. This function takes a dataset, the labels of the dataset, the number of clusters and returns the labels predicted by the model, the score obtained and the clusters centroids.
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from ml.clusters import validate


def evaluate_kmeans(data, y, n_clusters):

    k_params = {
        'algorithm': ['lloyd', 'elkan'],
        'init': ['k-means++', 'random'],
        'max_iter': [10, 100, 300, 400],
    }

    parameters, score = validate(
        KMeans(n_clusters=n_clusters, n_init='auto', random_state=SEED), data,
        y, k_params, v_measure_score)

    k_means = KMeans(n_clusters=n_clusters, n_init='auto', random_state=SEED)
    k_means.set_params(**parameters)
    # since we are clustering fit_predict produces
    # the same result as fit followed by predict in this case we could also
    # use k_means.labels_ instead of y_pred_kmeans, it is just a matter of taste
    y_pred_kmeans = k_means.fit_predict(data)

    return y_pred_kmeans, score, k_means.cluster_centers_, parameters


# %%
# The same workflow is used to tune the parameters of the :class:`sklearn.cluster.AgglomerativeClustering` model. This time the algorithm doesn't need to set a seed because it is deterministic.
from sklearn.cluster import AgglomerativeClustering


def evaluate_h_cluster(data, y, n_clusters):

    h_params = {
        'linkage': ['ward', 'complete', 'average', 'single'],
        'metric': ['manhattan', 'cosine', 'euclidean', 'l1', 'l2'],
    }

    parameters, score = validate(AgglomerativeClustering(n_clusters=n_clusters),
                                 data, y, h_params, v_measure_score)

    h_clust = AgglomerativeClustering(n_clusters=n_clusters)
    h_clust.set_params(**parameters)
    y_pred_hclust = h_clust.fit_predict(data, y)

    return y_pred_hclust, score, parameters


# %%
# Once the tuning procedures have been defined, the next step is to apply them to all the datasets. The following code snippet shows how to do it defining a function that takes a letter and the noise type and returns the labels predicted by the models, the scores obtained and the parameters used.
#
# .. warning::
#
#    the output of the following code cell is visible only in html version of the docs
import pandas as pd
from ml.datasets import load_dataset


def clusterize(dataset, noise_type=None):
    data, n_clusters = load_dataset(dataset, noise_type=noise_type)
    data = data.drop('noise_type', axis=1)

    # evaluate the dataset with kmeans and hierarchical clustering
    y_kmeans, k_score, centers, k_params = evaluate_kmeans(
        data.drop('label', axis=1), data['label'], n_clusters)
    y_hclust, h_score, h_params = evaluate_h_cluster(data.drop('label', axis=1),
                                                     data['label'], n_clusters)

    return y_kmeans, k_score, centers, k_params, y_hclust, h_score, h_params


clustering_results = []

# for each dataset and noise type
for letter in 'ABCDEFG':
    for noise in ['A', 'B', 'C', None]:
        # find the best parameters for kmeans and hierarchical clustering
        y_kmeans, k_score, centers, k_params, y_hclust, h_score, h_params = clusterize(
            letter, noise_type=noise)

        # store the results
        clustering_results.append({
            'dataset': letter,
            'noise_type': noise,
            'k_clusters': y_kmeans,
            'k_score': k_score,
            'k_centroids': centers,
            'k_params': k_params,
            'h_clusters': y_hclust,
            'h_score': h_score,
            'h_params': h_params
        })

dataframe = pd.DataFrame(clustering_results)
dataframe.drop(['k_centroids', 'k_clusters', 'h_clusters'], axis=1)

# %%
# Last step is to plot the results. The following code snippet shows how to do it. The function ``plot_results`` takes a dataset and plots the results of the clustering algorithms. To easily visualize the plots the data is projected in a 2D space using :class:`sklearn.decomposition.PCA`, which finds the optimal projection of the data in a 2D space.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.decomposition import PCA
from ml.visualization import DATASETS_LEGEND

warnings.filterwarnings('ignore')


def plot_results(dataset, noise_type, k_clusters, k_score, k_centroids,
                 k_params, h_clusters, h_score, h_params, axes):
    # load the dataset
    data = load_dataset(dataset, noise_type=noise_type)[0]

    # project the data in a 2D space
    transformer = PCA(n_components=2)
    X_2 = transformer.fit_transform(data.drop(['label', 'noise_type'], axis=1))

    # generate a dataframe with the projected data and the labels
    # (numerical labels of clusters found are replaced with their semantic meaning,
    # e.g. 0 -> 'tapes at 7.5 ips', 1 -> 'tapes at 15 ips', etc.)
    X_2 = pd.DataFrame(X_2, columns=['x', 'y'])
    X_2['label'] = np.array(data['label'])
    X_2 = X_2.replace(DATASETS_LEGEND[dataset]['labels'])

    # plot on the left subplot the real distribution of the data
    sns.scatterplot(x='x',
                    y='y',
                    hue='label',
                    data=X_2,
                    ax=axes[0],
                    palette=sns.color_palette())
    axes[0].set_title(
        f'Real distribution, noise type: {noise_type if noise_type is not None else "All"}'
    )
    axes[0].legend(title=DATASETS_LEGEND[dataset]['title'])

    # plot on the central subplot the clusters found by kmeans
    c = transformer.transform(k_centroids)
    sns.scatterplot(x='x',
                    y='y',
                    hue=k_clusters,
                    data=X_2,
                    ax=axes[1],
                    palette=sns.color_palette())
    axes[1].get_legend().remove()
    axes[1].set_title(f'KMeans score: {k_score:.3f}')
    axes[1].scatter(c[:, 0], c[:, 1], marker='x', s=100, c='black')

    # plot on the right subplot the clusters found by hierarchical clustering
    sns.scatterplot(x='x',
                    y='y',
                    hue=h_clusters,
                    data=X_2,
                    ax=axes[2],
                    palette=sns.color_palette())
    axes[2].get_legend().remove()
    axes[2].set_title(f'H. clustering score: {h_score:.3f}')


# plot the results for each dataset contained in the clustering_results list
for i in range(0, len(clustering_results), 4):
    fig, axes = plt.subplots(4,
                             3,
                             figsize=(12, 16),
                             constrained_layout=True,
                             sharey=True,
                             sharex=True)
    fig.suptitle(
        f"Clustering results for dataset {clustering_results[i]['dataset']}",
        fontsize=16)
    for j, noise in enumerate(['A', 'B', 'C', None]):
        plot_results(**clustering_results[i + j], axes=axes[j])
