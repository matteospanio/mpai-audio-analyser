import itertools
import json
from importlib import resources
import pandas as pd

_DATA_MODULE = 'ml.datasets'
_CONFIG = resources.files(_DATA_MODULE).joinpath('data/config.json')


def load_config():
    with open(_CONFIG, encoding='utf-8') as f:
        config = json.load(f)
    return config


def load_results():
    return load_config()['results']


def load_dataset(
    letter: str,
    noise_type: str = None,
    return_X_y: bool = False,
    purpose: str = 'clustering'
) -> tuple[pd.DataFrame, pd.Series, int] | tuple[pd.DataFrame, int]:
    """Load a dataset from the list defined in :numref:`clustering-datasets`.

    Parameters
    ----------
    letter : str
        the name of the dataset (a letter between A and G).
    noise_type : str, optional
        If present it applies a filter to the dataframe returning only entries with this kind of silence. Defaults to None.
    return_X_y : bool, optional
        If True splits the dataframe into X and y. Defaults to False.
    purpose : str, optional
        If 'clustering' it replaces the labels with integers, other strings have no effect. Defaults to 'clustering'.

    Raises
    ------
    KeyError
        If the letter is not in the list of available datasets.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, int] | tuple[pd.DataFrame, int]
        The dataset and the number of clusters.

    Examples
    --------
    >>> from ml.datasets import load_dataset
    >>> data, n_clusters = load_dataset('A')
    >>> n_clusters
    2
    >>> data.label.unique()
    array([1, 0])
    """
    mappings = load_config()['datasets'][letter]
    labels = mappings['labels']

    filters = {
        'labels': labels.keys(),
        'noise_type': noise_type,
        'combination': False
    }

    if letter in 'ABCDEHIJKL':
        data = load_pretto(filters=filters)
        data = data.replace(labels)
    elif letter in 'FG':
        data = load_berio_nono(filters=filters)
    else:
        raise ValueError(f'Invalid dataset letter: {letter}')

    if purpose == 'clustering':

        data = data.replace(labels)
        n_clusters = mappings['n_clusters']

        if return_X_y:
            return data.drop(columns=['label']), data.label, n_clusters

        return data, n_clusters

    if return_X_y:
        return data.drop(columns=['label']), data.label

    return data


def _filter_dataset(data: pd.DataFrame,
                    labels: list | None = None,
                    noise_type: str | None = None,
                    combination: bool = False):
    df = data
    if labels is not None:
        if combination:
            df = data[data['label'].isin(
                ['_'.join(l) for l in itertools.product(labels, labels)])]
        else:
            df = data[data['label'].isin(labels)]
    if noise_type is not None:
        df = df[df.noise_type == noise_type]

    return df


def load_pretto(filters: dict = None, return_X_y: bool = False):
    """Load and return the Pretto dataset (classification).

    =================   ============================
    Classes                                       25
    Samples per noise   2075 (A), 5050 (B), 1933 (C)
    Samples total                               9058
    Dimensionality                                15
    Features                           string, float
    =================   ============================

    Read more in the :ref:`Datasets <pretto>`.

    Examples
    --------

    .. doctest::

        >>> from ml.datasets import load_pretto
        >>> data = load_pretto(filters={'labels': ['7C', '7N'], 'noise_type': None, 'combination': True})
        >>> data.noise_type.unique()
        array(['A', 'B', 'C'], dtype=object)
        >>> data.label.unique()
        array(['7C_7C', '7C_7N', '7N_7C', '7N_7N'], dtype=object)

    """
    data = pd.read_csv(resources.files(_DATA_MODULE).joinpath('data/train.csv'))

    if filters is not None:
        data = _filter_dataset(data, filters.get('labels'),
                               filters.get('noise_type'),
                               filters.get('combination'))

    if return_X_y:
        return data.drop("label", axis=1), data["label"]

    return data


def load_berio_nono(filters: dict = None, return_X_y: bool = False):
    """Load and return the Berio-Nono dataset (classification).

    =================   ============================
    Classes                                        4
    Samples per noise   1231 (A), 1796 (B), 9175 (C)
    Samples total                              12202
    Dimensionality                                15
    Features                           string, float
    =================   ============================

    Read more in the :ref:`Datasets <berio-nono>`.

    Examples
    --------

    .. doctest::

        >>> from ml.datasets import load_berio_nono
        >>> data = load_berio_nono(filters={'labels': ['7C', '7N'], 'noise_type': None, 'combination': True})
        >>> data.noise_type.unique()
        array(['A', 'B', 'C'], dtype=object)
        >>> data.label.unique()
        array(['7C_7C', '7N_7N'], dtype=object)

    """
    data = pd.read_csv(resources.files(_DATA_MODULE).joinpath('data/test.csv'))

    if filters is not None:
        data = _filter_dataset(data, filters.get('labels'),
                               filters.get('noise_type'),
                               filters.get('combination'))

    if return_X_y:
        return data.drop("label", axis=1), data["label"]

    return data
