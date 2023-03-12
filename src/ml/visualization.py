"""
This module is a collection of functions to plot varoius data analysis and ml algorithms results.
"""
import numbers
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

_TITLES = {
    'eq': 'Equalization',
    'speed': 'Speed W/R',
}

DATASETS_LEGEND = {
    'A': {
        'title': _TITLES['eq'],
        'labels': {
            0: 'Wrong',
            1: 'Correct'
        }
    },
    'B': {
        'title': _TITLES['eq'],
        'labels': {
            0: 'Wrong',
            1: 'Correct'
        }
    },
    'C': {
        'title': _TITLES['speed'],
        'labels': {
            0: '7.5 ips',
            1: '15 ips'
        }
    },
    'D': {
        'title': _TITLES['speed'],
        'labels': {
            0: '3.75 ips',
            1: '7.5 ips',
            2: '15 ips'
        }
    },
    'E': {
        'title': _TITLES['eq'],
        'labels': {
            0: 'Wrong',
            1: 'Correct'
        }
    },
    'F': {
        'title': _TITLES['eq'],
        'labels': {
            0: '7.5 ips, CCIR',
            1: '7.5 ips, NAB',
            2: '15 ips, CCIR',
            3: '15 ips, NAB'
        }
    },
    'G': {
        'title': _TITLES['speed'],
        'labels': {
            0: '7.5 ips',
            1: '15 ips'
        }
    }
}


def plot_random_forest_feature_importance(df: pd.DataFrame,
                                          rf: RandomForestClassifier
                                          | RandomForestRegressor,
                                          ax: plt.Axes = None,
                                          title: str = 'Feature Importances'):
    """Plot the feature importances of a random forest as a bar chart.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the features.
    rf: RandomForestClassifier | RandomForestRegressor
        The random forest model.
    ax: plt.Axes, optional
        The axes to plot on.
    title: str, optional
        The title of the plot.
    """
    feature_names = df.columns.to_list()

    if ax is None:
        _, ax = plt.subplots()

    ax.bar(range(0, df.shape[1]), rf.feature_importances_)
    ax.set_title(title)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(feature_names)
    ax.grid()


def plot_distribution_matrix(df: pd.DataFrame,
                             labels: list,
                             ax: plt.Axes = None,
                             title: str = 'Distribution Matrix'):
    """Plot the distribution matrix of a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to plot.
    ax: plt.Axes, optional
        The axes to plot on, if None a new figure is created.
    labels: list
        The labels to use.
    title: str, optional
        The title of the plot.

    Example
    -------

    The classic use case is to plot the confusion matrix of a classifier:

    .. doctest::

        >>> from ml.datasets import load_pretto
        >>> from ml.visualization import plot_distribution_matrix
        >>> data = load_pretto()
        >>> plot_distribution_matrix(data, labels=['7C', '7N'])

    The code above will produce the following plot:

    .. figure:: ../../../_static/img/plot_distribution_matrix.png
        :align: center

    """

    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(title)

    distribution = []
    combinations = ['_'.join(l) for l in itertools.product(labels, labels)]

    for label in labels:
        tmp = []
        for combination in combinations:
            if combination.startswith(label):
                try:
                    tmp.append(df.label.value_counts()[combination])
                except KeyError:
                    tmp.append(0)
        distribution.append(tmp)

    sns.heatmap(distribution,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)


def _get_indexes_of_best_params(grid, param_ranges, param_to_vary_idx):
    slices = []

    for idx, param in enumerate(grid.best_params_):
        if idx == param_to_vary_idx:
            slices.append(slice(None))
            continue
        best_param_val = grid.best_params_[param]
        idx_of_best_param = 0
        if isinstance(param_ranges[idx], np.ndarray):
            idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
        else:
            idx_of_best_param = param_ranges[idx].index(best_param_val)
        slices.append(idx_of_best_param)

    return slices


def plot_grid_search_validation_curve(grid: GridSearchCV,
                                      param_to_vary: str,
                                      ax: plt.Axes = None,
                                      title: str = 'Validation Curve',
                                      ylim: tuple = None):
    """Plot the validation curve of a grid search.

    Parameters
    ----------
    grid: GridSearchCV
        The grid search to plot.
    param_to_vary: str
        The name of the parameter is going to be in the x-axis.
    ax: plt.Axes, optional
        The axes to plot on.
    title: str, optional
        The title of the plot.
    ylim: tuple, optional
        The y-axis limits.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from ml.classification import evaluate_model
    >>> from ml.visualization import plot_grid_search_validation_curve
    >>> X, y = load_iris(return_X_y=True)
    >>> param_grid = {
    ...     'n_estimators': [10, 100, 1000],
    ...     'max_depth': [1, 10, 100]
    ... }
    >>> grid = evaluate_model(RandomForestClassifier(), X, y, param_grid)
    >>> plot_grid_search_validation_curve(grid, 'n_estimators')

    """

    df_cv_results = pd.DataFrame(grid.cv_results_)

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
    param_ranges_lengths = [len(pr) for pr in param_ranges]

    param_to_vary_idx = param_cols.index(f'param_{param_to_vary}')

    slices = _get_indexes_of_best_params(grid, param_ranges, param_to_vary_idx)

    scores = {
        "train": {
            "mean":
            np.array(df_cv_results['mean_train_score']).reshape(
                *param_ranges_lengths)[tuple(slices)],
            "std":
            np.array(df_cv_results['std_train_score']).reshape(
                *param_ranges_lengths)[tuple(slices)],
        },
        "valid": {
            "mean":
            np.array(df_cv_results['mean_test_score']).reshape(
                *param_ranges_lengths)[tuple(slices)],
            "std":
            np.array(df_cv_results['std_test_score']).reshape(
                *param_ranges_lengths)[tuple(slices)],
        }
    }

    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(param_to_vary)
    ax.set_ylabel("Score")

    ylim = (0.0, 1.1) if ylim is None else ylim
    ax.set_ylim(*ylim)

    lw = 2

    param_range = param_ranges[param_to_vary_idx]
    if not isinstance(param_range[0], numbers.Number):
        param_range = [str(x) for x in param_range]
    ax.plot(param_range,
            scores['train']['mean'],
            label='Training score',
            color='r',
            lw=lw)
    ax.fill_between(param_range,
                    scores['train']['mean'] - scores['train']['std'],
                    scores['train']['mean'] + scores['train']['std'],
                    alpha=0.1,
                    color='r',
                    lw=lw)
    ax.plot(param_range,
            scores['valid']['mean'],
            label='Cross-validation score',
            color='b',
            lw=lw)
    ax.fill_between(param_range,
                    scores['valid']['mean'] - scores['valid']['std'],
                    scores['valid']['mean'] + scores['valid']['std'],
                    alpha=0.1,
                    color='b',
                    lw=lw)

    ax.legend(loc='lower right')


def plot_confusion_matrix(y_true,
                          y_pred,
                          ax: plt.Axes = None,
                          cmap='Blues',
                          labels=None,
                          title='Confusion Matrix'):
    """Plot the confusion matrix of a model.

    Parameters
    ----------
    y_true: array-like
        The true labels.
    y_pred: array-like
        The predicted labels.
    ax: plt.Axes, optional
        The axes to plot on.
    cmap: str, optional
        The color map to use.
    title: str, optional
        The title of the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    ax.set_title(title)
    sns.heatmap(df_cm, annot=True, fmt="", ax=ax, cmap=cmap)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
