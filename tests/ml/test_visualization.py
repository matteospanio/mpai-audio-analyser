import tempfile
import os
from importlib import resources
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ml.datasets import load_pretto
from ml.visualization import plot_distribution_matrix, plot_confusion_matrix

REFERENCE_FOLDER = resources.files('tests.ml.baseline_images')


def test_plot_distribution_matrix():
    _, ax = plt.subplots()
    data = load_pretto()
    plot_distribution_matrix(data, labels=['7C', '7N'], ax=ax)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'plot_distribution_matrix.png')
        ax.figure.savefig(file_path)

        saved_plot = plt.imread(file_path)
        reference = REFERENCE_FOLDER.joinpath('plot_distribution_matrix.png')
        reference = plt.imread(reference)

        saved_plot = np.array(saved_plot, dtype=np.float32)
        reference = np.array(reference, dtype=np.float32)

        assert np.allclose(saved_plot, reference)


def test_plot_confusion_matrix():
    _, ax = plt.subplots()
    X, y = load_pretto(return_X_y=True)
    X = X.drop(columns=['noise_type'], axis=1)

    model = DecisionTreeClassifier(random_state=0)
    model.fit(X, y)
    y_pred = model.predict(X)

    plot_confusion_matrix(y, y_pred, ax=ax)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'plot_confusion_matrix.png')
        ax.figure.savefig(file_path)

        saved_plot = plt.imread(file_path)
        reference = REFERENCE_FOLDER.joinpath('plot_confusion_matrix.png')
        reference = plt.imread(reference)

        saved_plot = np.array(saved_plot, dtype=np.float32)
        reference = np.array(reference, dtype=np.float32)
        assert np.allclose(saved_plot, reference)
