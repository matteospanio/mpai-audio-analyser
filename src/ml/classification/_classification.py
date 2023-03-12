import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from ml.datasets import load_berio_nono, load_pretto
from ._data_structures import Classifier
from ._constants import CLASSIFICATION_MAPPING, MODELS_PATH


def evaluate_model(model: BaseEstimator,
                   x,
                   y,
                   params: dict,
                   cv: int = 5) -> GridSearchCV:
    """
    It evaluates a model using GridSearchCV.

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
    cv: int
        the number of folds to be used in the cross-validation, default is 5.

    Returns
    -------
    GridSearchCV
        The GridSearchCV object fitted on the data

    Examples
    --------
    Here is an example of usage:

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from ml.classification import evaluate_model
    >>> x, y = load_iris(return_X_y=True)
    >>> model = DecisionTreeClassifier(random_state=0)
    >>> params = {'max_depth': [2, 3, 4], 'criterion': ['gini', 'entropy']}
    >>> gscv = evaluate_model(model, x, y, params)
    >>> gscv.best_params_
    {'criterion': 'gini', 'max_depth': 4}

    Notes
    -----
    This function is just a wrapper of :class:`sklearn.model_selection.GridSearchCV`. It has been created for comodity.

    """
    gscv = GridSearchCV(estimator=model,
                        param_grid=params,
                        cv=cv,
                        n_jobs=-1,
                        return_train_score=True)
    gscv.fit(x, y)

    return gscv


def load_model(model_name: str) -> Classifier:
    """Load a trained classifier from disk.
    Aviable models are:

    - pretto_and_berio_nono_classifier

    Parameters
    ----------
    model_name: str
        the path of the model to be loaded

    Raises
    ------
    ValueError
        if the model name is not valid

    Returns
    -------
    Classifier
        the classifier loaded from disk

    """

    models = {
        "pretto_classifier":
        MODELS_PATH.joinpath('pretto_classifier.pkl'),
        "pretto_and_berio_nono_classifier":
        MODELS_PATH.joinpath('pretto_and_berio_nono_classifier.pkl')
    }

    try:
        with open(models[model_name], 'rb') as f:
            return Classifier(pickle.load(f))
    except FileNotFoundError:
        generate_classifier(models[model_name])
        return load_model(model_name)


def generate_classifier(dest_path):
    data1 = load_pretto()
    data2 = load_berio_nono()

    data = pd.concat([data1, data2])
    data = data.replace(CLASSIFICATION_MAPPING)

    X = data.drop(columns=['noise_type', 'label'], axis=1)
    y = data.label

    rfc = RandomForestClassifier(n_estimators=111,
                                 criterion="log_loss",
                                 max_features="log2",
                                 min_samples_leaf=1,
                                 n_jobs=-1)

    rfc.fit(X, y)

    with open(dest_path, 'wb') as f:
        pickle.dump(rfc, f)
