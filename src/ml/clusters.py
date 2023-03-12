import itertools
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator


def _combinations_generator(params: dict):
    """A generator function. It returns the cartesian product of the values of the parameters."""
    keys, values = zip(*params.items())
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def validate(model: BaseEstimator, x, y, params: dict,
             score_function) -> tuple[dict, float]:
    """
    Exhaustive search over specified parameter values for an estimator. It is the corrispodent of GridSearchCV for clustering: it tries as parameters the cartesian product of the parameters given and returns the best one.

    Parameters
    ----------
    model: BaseEstimator
        the estimator to be validated
    x
        the data to be fitted
    y
        the target values, it is used to evaluate the score
    params: dict
        the parameters to be tuned, it should be a dictionary where the keys are the parameters' names and the values are lists of the values to be tried
    score_function
        the function used to evaluate the score (it is assumed that actual labels are known), it should take as parameters the actual labels and the predicted labels, one from :mod:`sklearn.metrics` can be used (adjusted_rand_score, v_measure_score, etc.)

    Returns
    -------
    tuple[dict, float]
        The best parameters and the best score

    Examples
    --------

    Here is an example of usage:

    .. doctest::

        >>> from sklearn.cluster import AgglomerativeClustering
        >>> from sklearn.datasets import make_blobs
        >>> from sklearn.metrics import adjusted_rand_score
        >>> from ml.clusters import validate
        >>> x, y = make_blobs(n_samples=100, n_features=2, centers=3, random_state=0)
        >>> model = AgglomerativeClustering()
        >>> params = {'n_clusters': [2, 3, 4], 'linkage': ['ward', 'complete', 'average']}
        >>> best_params, best_score = validate(model, x, y, params, adjusted_rand_score)
        >>> best_params
        {'n_clusters': 3, 'linkage': 'average'}

    """
    best_score = 0
    best_params = None
    for combination in _combinations_generator(params):
        if isinstance(model, AgglomerativeClustering):
            if combination.get('linkage') == 'ward':
                combination['metric'] = 'euclidean'
        model.set_params(**combination)
        y_pred = model.fit_predict(x, y)
        score = score_function(y, y_pred)
        if score > best_score:
            best_score = score
            best_params = combination

    return best_params, best_score
