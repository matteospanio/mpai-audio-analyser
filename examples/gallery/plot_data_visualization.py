"""
.. _data_visualization:

Data Visualization
==================

This notebook shows the distribution of the datasets from :ref:`datasets`. The center of interest is to study the distribution of the labels in the dataset.

"""
# %%
# Firstly we load the dataset and we display some information about it (the number of records and their features).
from ml.datasets import load_pretto

data = load_pretto()
data.info()
# %%
# Displaing the distribution of the labels in a matrix we can see that when the equalization filters are the same, the number of noises found is almost costant across the different combinations (the main diagonal). While tapes recorded at a higher speed and reproduced at a lower speed have a higher number of noises (and viceversa).
import matplotlib.pyplot as plt
from ml.visualization import plot_distribution_matrix

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
labels = ["3N", "7C", "7N", "15C", "15N"]

filters = {'labels': labels, 'combination': True}

for ax, noise in zip(axes.flatten(), [None, "A", "B", "C"]):
    filters["noise_type"] = noise
    dataset = load_pretto(filters=filters)
    plot_distribution_matrix(
        dataset,
        ax=ax,
        labels=labels,
        title=(f"Noise type {noise}" if noise is not None else "Whole dataset"))
# %%
# We can do the same process for the other dataset.
from ml.datasets import load_berio_nono

data = load_berio_nono()
data.info()
# %%
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(13, 10))

for ax, noise in zip(axes.flatten(), [None, "A", "B", "C"]):
    labels = ["7C", "7N", "15C", "15N"]
    dataset = load_berio_nono(filters={
        'labels': labels,
        "noise_type": noise,
        'combination': True
    })
    plot_distribution_matrix(
        dataset,
        ax=ax,
        labels=labels,
        title=(f"Noise type {noise}" if noise is not None else "Whole dataset"))
# %%
# Since in Berio-Nono dataset there are only tapes with correct equalization the distribution matrix is less useful, as we can see it populates only the main diagonal. In this case it's more useful to use a more classic histogram as the one below.
import seaborn as sns

sns.histplot(data=data, x="label", hue="noise_type", multiple="stack")
plt.title("Distribution of the labels in the Berio-Nono dataset")
plt.xlabel("Equalization parameters")

# %%
# Another interesting thing to study is how the two datasets are distributed in the feature space. We can do this by plotting the first two principal components of the dataset.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,
                         2,
                         figsize=(13, 10),
                         sharex=True,
                         sharey=True,
                         constrained_layout=True)

data = load_pretto()
pca = PCA(n_components=2)
pca.fit(data.drop(columns=['noise_type', 'label'], axis=1))

for ax, noise in zip(axes.flatten(), [None, "A", "B", "C"]):

    pretto = load_pretto(filters={'noise_type': noise})
    berio = load_berio_nono(filters={'noise_type': noise})

    pretto = pretto.drop(columns=['noise_type', 'label'], axis=1)
    berio = berio.drop(columns=['noise_type', 'label'], axis=1)

    pretto2d = pca.transform(pretto)
    berio2d = pca.transform(berio)

    sns.scatterplot(x=pretto2d[:, 0], y=pretto2d[:, 1], ax=ax, label="Pretto")
    sns.scatterplot(x=berio2d[:, 0], y=berio2d[:, 1], ax=ax, label="Berio-Nono")
    ax.get_legend().remove()
    ax.set_title(
        f"Noise type {noise}" if noise is not None else "Whole dataset")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.legend(bbox_to_anchor=(1.02, 1),
           loc='upper left',
           borderaxespad=0,
           title="Dataset")
plt.suptitle("Distribution of the datasets in the feature space",
             size=16,
             y=1.05)

# %%
