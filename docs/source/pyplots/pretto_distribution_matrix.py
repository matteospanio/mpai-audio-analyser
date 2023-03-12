import matplotlib.pyplot as plt
from ml.datasets import load_pretto
from ml.visualization import plot_distribution_matrix

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, sharex=True, sharey=True)
labels = ["3N", "7C", "7N", "15C", "15N"]
filters = {'labels': labels, 'combination': True}

fig.suptitle("Pretto dataset distribution matrix")

for ax, noise in zip(axes.flatten(), [None, "A", "B", "C"]):
    filters['noise_type'] = noise
    dataset = load_pretto(filters=filters)
    plot_distribution_matrix(
        dataset,
        ax=ax,
        labels=labels,
        title=(f"Noise type {noise}" if noise is not None else "Whole dataset"))
