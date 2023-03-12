from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ml.datasets import load_pretto, load_berio_nono

fig, ax = plt.subplots(figsize=(10, 10))

pretto = load_pretto()
berio = load_berio_nono()

pca = PCA(n_components=2)
pca.fit(pretto.drop(columns=['noise_type', 'label'], axis=1))

pretto2d = pca.transform(pretto.drop(columns=['noise_type', 'label'], axis=1))
berio2d = pca.transform(berio.drop(columns=['noise_type', 'label'], axis=1))

pretto2d = pd.DataFrame(pretto2d, columns=["PC1", "PC2"])
pretto2d["noise type"] = pretto["noise_type"]
pretto2d["label"] = pretto["label"]
pretto2d["dataset"] = "Pretto"

berio2d = pd.DataFrame(berio2d, columns=["PC1", "PC2"])
berio2d["noise type"] = berio["noise_type"]
berio2d["label"] = berio["label"]
berio2d["dataset"] = "Berio-Nono"

data = pd.concat([pretto2d, berio2d])

sns.scatterplot(data=data, x="PC1", y="PC2", ax=ax, hue="noise type", style="dataset")
ax.set_title("Distribution of the datasets in the feature space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
