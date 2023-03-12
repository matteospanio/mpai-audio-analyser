from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from ml.datasets import load_pretto

warnings.filterwarnings("ignore")

fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharey=True, sharex=True)

dataset_e = load_pretto()
dataset_e = dataset_e[dataset_e["label"].isin(
    ["15C_15C",
     "15N_15N",
     "7C_7C",
     "7N_7N",
     "7C_7N",
     "7N_7C",
     "15C_15N",
     "15N_15C"]
    )]

pca = PCA(n_components=2)
pca.fit(dataset_e.drop(columns=["noise_type", "label"], axis=1))

two_dim = pca.transform(dataset_e.drop(columns=["label", "noise_type"], axis=1))
two_dim = pd.DataFrame(two_dim, columns=["PC1", "PC2"])

speeds = []
dataset_e.apply(
    lambda x: speeds.append("7.5 ips") if "7" in x.label else speeds.append("15 ips"),
    axis=1)
two_dim["Speed W/R"] = speeds

eq = []
mapping = {
    "15C_15C": "correct",
    "15N_15N": "correct",
    "7C_7C": "correct",
    "7N_7N": "correct",
    "15C_15N": "wrong",
    "15N_15C": "wrong",
    "7C_7N": "wrong",
    "7N_7C": "wrong",
}
dataset_e.apply(
    lambda x: eq.append(mapping[x.label]),
    axis=1)
two_dim["Equalization"] = eq

a_2_dim = two_dim[two_dim["Speed W/R"] == '7.5 ips']
b_2_dim = two_dim[two_dim["Speed W/R"] == '15 ips']

fig.suptitle("Equalization distribution in the PCA space and clustering results")

sns.scatterplot(data=a_2_dim, x="PC1", y="PC2", ax=axes[1][0], hue="Equalization")
axes[1][0].set_title("Subset A")
axes[1][0].get_legend().remove()

sns.scatterplot(data=b_2_dim, x="PC1", y="PC2", ax=axes[1][1], hue="Equalization")
axes[1][1].set_title("Subset B")
axes[1][1].get_legend().remove()

h_clust = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")

a_2_dim["Clusters"] = h_clust.fit_predict(a_2_dim.drop(columns=["Equalization", "Speed W/R"], axis=1))

sns.scatterplot(data=a_2_dim, x="PC1", y="PC2", ax=axes[0][0], hue="Clusters")
axes[0][0].set_title("Clustering A")
axes[0][0].get_legend().remove()

h_clust = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")

b_2_dim["Clusters"] = h_clust.fit_predict(b_2_dim.drop(columns=["Equalization", "Speed W/R"], axis=1))

sns.scatterplot(data=b_2_dim, x="PC1", y="PC2", ax=axes[0][1], hue="Clusters")
axes[0][1].set_title("Clustering B")
axes[0][1].get_legend().remove()

two_dim["Clusters"] = h_clust.fit_predict(two_dim.drop(columns=["Equalization", "Speed W/R"], axis=1))

sns.scatterplot(data=two_dim, x="PC1", y="PC2", hue="Clusters", style="Speed W/R", ax=axes[0][2])
axes[0][2].set_title("Clustering E")
axes[0][2].get_legend().remove()

sns.scatterplot(data=two_dim, x="PC1", y="PC2", hue="Equalization", style="Speed W/R", ax=axes[1][2])
axes[1][2].set_title("Subset E")
axes[1][2].get_legend().remove()

plt.legend(bbox_to_anchor=(0.8, 0.5),
           loc='center left',
           borderaxespad=0)
