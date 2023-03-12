"""
.. _equalization_classification:

Classification analysis
========================

This notebook shows the performance of classification algorithms on the datasets from :ref:`datasets`.

As for clustering analysis, the datasets have been splitted in smaller ones to look at specific aspects of the data, but this time also a global analysis has been made (see :ref:`all_classes` and :ref:`mixin_datasets`).

In synthesis the subsets are elencated in :numref:`classification-datasets`. Since those datasets are used to train classification models, they are all subsets of Pretto dataset which comprehends more cases.

To evaluate the performance of the classification algorithms has been used the accuracy score, which is the ratio of the number of correctly classified samples to the total number of samples.

The classification is always performed tuning the hyperparameters of the model using a grid search with cross validation at 5 folds on 80% of the selected dataset, once the parameters have been tuned the model performance is tested over the remaining 20% of the data. The models taken in consideration are: K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest.

In the first part of the notebook datasets are analyzed separately, trained and validated on the same data splitted 80% for training and 20% for testing. The procedure to find and evaluate the best models are as follow:

1. split the dataset in 80% and 20%
2. tune and train KNN, SVM, DT and RF on the dataset without splitting by noise type
3. for each noise type and for each model train the model on the subset of the dataset with the best parameters found previously
4. test each model trained on a specific validation set

"""
# %%
# Firslty we set the seed for reproducibility at a specific value.
SEED = 42
# %%
# Workflow explanation
# --------------------
# Let's see the classification process over a specific dataset (*H*).
from ml.datasets import load_dataset

X, y = load_dataset('H', return_X_y=True, purpose='classification')
X = X.drop(columns=['noise_type'])

# %%
# We split the dataset in train and test set, reserving 20% of the data as validation set.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=SEED)
print('Number of train Samples:', len(X_train), 'Number of test Samples:',
      len(X_test))
# %%
# We firstly explore the K-Nearest Neighbors algorithm.
from sklearn.neighbors import KNeighborsClassifier
from ml.classification import evaluate_model

knn = evaluate_model(KNeighborsClassifier(),
                     X_train,
                     y_train,
                     params={
                         'n_neighbors': [x for x in range(1, 100)],
                         'weights': ['uniform', 'distance'],
                         'metric':
                         ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
                     })

print("KNN best params:", knn.best_params_)
# %%
# Let's see what happens in function of the parameters passed for tuning
import matplotlib.pyplot as plt
from ml.visualization import plot_grid_search_validation_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for param, ax in zip(['n_neighbors', 'weights', 'metric'], axes.flatten()):
    plot_grid_search_validation_curve(knn,
                                      param,
                                      ax=ax,
                                      title=f"{param}",
                                      ylim=(0.85, 1.01))

# %%
from sklearn.metrics import accuracy_score, classification_report

y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
# %%
# The accuracy is really good, for further analysis we can look at the classification report and the confusion matrix.
from ml.visualization import plot_confusion_matrix

print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test,
                      y_pred,
                      title='KNN Confusion matrix',
                      labels=['7C_7C', '7C_7N', '7N_7C', '7N_7N'])

# %%
# The report still gives excellent results. The next step is to look for better results using SVM, Decision Tree and Random Forest.
#
# Starting from SVM, we approach the problem in a similar way to the previous one:
#
# - we tune the hyperparameters of the model using a grid search with cross validation at 5 folds
# - we plot the validation curve for each hyperparameter
# - we evaluate the model on the validation set
# - we look at the confusion matrix
from sklearn.svm import SVC

svm = evaluate_model(SVC(random_state=SEED),
                     X_train,
                     y_train,
                     params={
                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'degree': [x for x in range(1, 5)],
                         'gamma': ['scale', 'auto'],
                     })

print("SVM best params:")
print(svm.best_params_)
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for param, ax in zip(['kernel', 'degree', 'gamma'], axes.flatten()):
    plot_grid_search_validation_curve(svm,
                                      param,
                                      ax=ax,
                                      title=f"{param}",
                                      ylim=(0.5, 1.05))
# %%
y_pred = svm.predict(X_test)
print('accuracy:', accuracy_score(y_test, y_pred))
plot_confusion_matrix(y_test,
                      y_pred,
                      title='SVM Confusion matrix of dataset H',
                      labels=['7C_7C', '7C_7N', '7N_7C', '7N_7N'])

# %%
# Perfect, the results are the same as in KNN.
#
# Now that a methodology has been defined we can produce a routine to train 4 models for each dataset. The function ``classify`` will return a list with all best parameters obtained from GridSearchCV for each model.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ml.datasets import load_pretto


def classify(dataset=None):

    if dataset is None:
        X, y = load_pretto(return_X_y=True)

    else:
        X, y = load_dataset(dataset, return_X_y=True, purpose='classification')

    labels = y.unique()
    X = X.drop(columns=['noise_type'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=SEED)

    knn = evaluate_model(
        KNeighborsClassifier(),
        X_train,
        y_train,
        params={
            'n_neighbors': [x for x in range(1, 50)],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        })

    svm = evaluate_model(SVC(random_state=SEED),
                         X_train,
                         y_train,
                         params={
                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                             'degree': [x for x in range(1, 5)],
                             'gamma': ['scale', 'auto'],
                         })

    tree = evaluate_model(DecisionTreeClassifier(random_state=SEED),
                          X_train,
                          y_train,
                          params={
                              'max_features': ['log2', 'sqrt', None],
                              'criterion': ['gini', 'log_loss'],
                              'min_samples_leaf': [1, 2, 5, 10, 20],
                              'max_depth': [None, 2, 5, 10, 20, 30],
                              'splitter': ['best'],
                          })

    rfc = evaluate_model(RandomForestClassifier(random_state=SEED),
                         X_train,
                         y_train,
                         params={
                             'n_estimators': [x for x in range(1, 121, 10)],
                             'max_features': ['log2', 'sqrt'],
                             'criterion': ['gini', 'log_loss'],
                             'min_samples_leaf': [1, 2, 5, 10],
                         })

    models_list = [('KNN', knn), ('SVM', svm), ('Decision Tree', tree),
                   ('Random Forest', rfc)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    fig.suptitle(
        f'Confusion matrices on entire dataset {dataset if dataset is not None else ""}',
        fontsize=16)

    for ax, (name, model) in zip(axes.flatten(), models_list):

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        plot_confusion_matrix(y_test,
                              y_pred,
                              ax=ax,
                              title=f"{name} - {acc}",
                              labels=labels)

    return models_list


# %%
# Then ``plot_performance`` will print the confusion matrix for each algorithm and noise type.
def plot_performance(dataset, trained_models):
    fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    fig.suptitle(f'Confusion matrices on noise type, dataset {dataset}',
                 fontsize=16)

    for ax_line, (name, model) in zip(axes, trained_models):

        for ax, noise_type in zip(ax_line.flatten(), ['A', 'B', 'C']):
            if dataset is None:
                X, y = load_pretto(return_X_y=True,
                                   filters={'noise_type': noise_type})
            else:
                X, y = load_dataset(dataset,
                                    noise_type,
                                    return_X_y=True,
                                    purpose='classification')
            X = X.drop(columns=['noise_type'], axis=1)
            labels = y.unique()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            plot_confusion_matrix(
                y_test,
                y_pred,
                ax=ax,
                title=f"{name} - noise {noise_type} - {acc:.3f}",
                labels=labels)


# %%
# All Pretto subsets
# ------------------
#
# Once we defined two functions to evaluate the models performance let's evaluate all the datasets defined above:
#
# - subsets H, I, J, K are going to classify pre and post emphasis equalization curves
# - subset L is going to classify the tape's speed used in both recording and replaying (7.5 or 15 ips)
import pandas as pd
for dataset in 'HIJKL':
    trained_models = classify(dataset)
    print(
        pd.DataFrame([(name, model.best_score_, model.best_params_)
                      for name, model in trained_models],
                     columns=['Model', 'Best Score', 'Trained model']))
    plot_performance(dataset, trained_models)
# %%
# .. _all_classes:
#
# All classes
# ------------
#
# Let's see what happens if we use all the classes of the dataset Pretto.
# Since this dataset is greater than the others, we will exclude the SVM model from the evaluation because it takes too long to train, and the decision tree, which underperforms RF.
X, y = load_pretto(return_X_y=True)
X = X.drop(columns=['noise_type'], axis=1)
labels = y.unique()
labels
# %%
# This time there are 25 classes.
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=SEED)

knn = evaluate_model(KNeighborsClassifier(),
                     X_train,
                     y_train,
                     params={
                         'n_neighbors': [x for x in range(1, 50)],
                         'weights': ['uniform', 'distance'],
                         'metric':
                         ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
                     })

rfc = evaluate_model(RandomForestClassifier(random_state=SEED),
                     X_train,
                     y_train,
                     params={
                         'n_estimators': [x for x in range(1, 121, 10)],
                         'max_features': ['log2', 'sqrt'],
                         'criterion': ['gini', 'log_loss'],
                         'min_samples_leaf': [1, 2, 5, 10],
                     })

models_list = [('KNN', knn), ('Random Forest', rfc)]

pd.DataFrame([(name, model.best_score_, model.best_params_)
              for name, model in models_list],
             columns=['Model', 'Best Score', 'Trained model'])
# %%
# Then we plot the confusion matrices:
fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
fig.suptitle('Confusion matrices on entire dataset', fontsize=20)

for ax, (name, model) in zip(axes, models_list):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plot_confusion_matrix(y_test,
                          y_pred,
                          ax=ax,
                          title=f"{name} - {acc}",
                          labels=labels)

# %%
# Let's see more result stats:
for (name, model) in models_list:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(name)
    print(pd.DataFrame(report).transpose())

# %%
# .. _mixin_datasets:
#
# Mixin datasets
# --------------
# Once the best model has been found, it's interesting to see if the model, trained on the whole dataset, is able to generalize well. To do this we can train the model on the Pretto dataset, which contains more equalization parameters, and is expected to have a better coverage over all possible inputs and test it on the Berio-Nono dataset, which is composed only by features of correctly equalized tapes.
#
# The model selected for this analysis is the Random Forest, which gave the best results in the previous analysis.
from ml.datasets import load_berio_nono, load_pretto
import matplotlib.pyplot as plt

X_train, y_train = load_pretto(return_X_y=True)
X_test, y_test = load_berio_nono(return_X_y=True)

X_train = X_train.drop(columns=['noise_type'])
X_test = X_test.drop(columns=['noise_type'])

labels = y_train.unique()

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
if len(pd.Series(y_pred).unique()) != len(labels):
    missing = set(labels) - set(pd.Series(y_pred).unique())

    for i, m in enumerate(missing):
        y_pred[i] = m

fig, axes = plt.subplots(figsize=(9, 9), constrained_layout=True)

plot_confusion_matrix(
    y_test,
    y_pred,
    ax=axes,
    title=f'Random Forest {accuracy_score(y_test, y_pred):.3f}',
    labels=labels)
# %%
# Since results are not so good, we can try to take in consideration the noise type and see if the results improve.
import warnings

warnings.filterwarnings('ignore')

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle('RF Confusion matrix (varying noise type)')

for ax, noise_type in zip(axes.flatten(), ['A', 'B', 'C']):
    X_train, y_train = load_pretto(return_X_y=True,
                                   filters={'noise_type': noise_type})
    X_test, y_test = load_berio_nono(return_X_y=True,
                                     filters={'noise_type': noise_type})

    labels = y_train.unique()
    X_train = X_train.drop(columns=['noise_type'])
    X_test = X_test.drop(columns=['noise_type'])

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    if len(pd.Series(y_pred).unique()) != len(labels):
        missing = set(labels) - set(pd.Series(y_pred).unique())

        for i, m in enumerate(missing):
            y_pred[i] = m

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=f'{noise_type}, {accuracy_score(y_test, y_pred):.3f}',
        ax=ax)

# %%
# Let's see if the results improve taking in consideration also specific speeds.
from ml.datasets import load_dataset

fig, axes = plt.subplots(2, 2, figsize=(13, 12))
fig.suptitle('SVM Confusion matrix (varying noise type) with speed at 7.5 ips')

svc = SVC(kernel='poly', random_state=SEED, degree=4, gamma='auto')

for ax, noise_type in zip(axes.flatten(), ['A', 'B', 'C', None]):
    X_train, y_train = load_dataset(return_X_y=True,
                                    noise_type=noise_type,
                                    purpose='classification',
                                    letter='H')
    test = load_berio_nono(filters={'noise_type': noise_type})
    test = test[(test.label == '7C_7C') | (test.label == '7N_7N')]

    X_train = X_train.drop(columns=['noise_type'], axis=1)
    X_test = test.drop(columns=['noise_type', 'label'], axis=1)
    y_test = test.label
    y_test = y_test.replace({'7C_7C': 'C-C', '7N_7N': 'N-N'})

    labels = y_train.unique()

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    if len(pd.Series(y_pred).unique()) != len(labels):
        missing = set(labels) - set(pd.Series(y_pred).unique())

        for i, m in enumerate(missing):
            y_pred[i] = m

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=f'{noise_type}, {accuracy_score(y_test, y_pred):.3f}',
        ax=ax)

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
fig.suptitle('SVM Confusion matrix (varying noise type) with speed at 15 ips')

svc = SVC(kernel='poly', random_state=SEED, degree=4, gamma='auto')

for ax, noise_type in zip(axes.flatten(), ['A', 'B', 'C', None]):
    X_train, y_train = load_dataset(return_X_y=True,
                                    noise_type=noise_type,
                                    purpose='classification',
                                    letter='I')
    test = load_berio_nono(filters={'noise_type': noise_type})
    test = test[(test.label == '15C_15C') | (test.label == '15N_15N')]

    X_train = X_train.drop(columns=['noise_type'], axis=1)
    X_test = test.drop(columns=['noise_type', 'label'], axis=1)
    y_test = test.label
    y_test = y_test.replace({'15C_15C': 'C-C', '15N_15N': 'N-N'})

    labels = y_train.unique()

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    if len(pd.Series(y_pred).unique()) != len(labels):
        missing = set(labels) - set(pd.Series(y_pred).unique())

        for i, m in enumerate(missing):
            y_pred[i] = m

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=f'{noise_type}, {accuracy_score(y_test, y_pred):.3f}',
        ax=ax)
# %%
# In the following section we take in consideration only ``7C_7C`` and ``7N_7N`` from Pretto dataset to see how a model trained on these data performs at classifying Berio-Nono istances with this equalization parameters.
from ml.datasets import load_dataset

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'SVM Confusion matrix (varying noise type) with speed at 7.5 ips and correct equalization'
)

for ax, noise_type in zip(axes.flatten(), ['A', 'B', 'C', None]):
    X_train, y_train = load_dataset(noise_type=noise_type,
                                    return_X_y=True,
                                    purpose='classification',
                                    letter='J')
    test = load_dataset(letter='F',
                        noise_type=noise_type,
                        purpose='classification')
    test = test[(test.label == '7C_7C') | (test.label == '7N_7N')]

    X_train = X_train.drop(columns=['noise_type'], axis=1)
    X_test = test.drop(columns=['noise_type', 'label'], axis=1)
    y_test = test.label
    y_test = y_test.replace({'7C_7C': 'CCIR', '7N_7N': 'NAB'})
    labels = y_train.unique()

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=
        f'{"All" if noise_type is None else noise_type}, {accuracy_score(y_test, y_pred):.3f}',
        ax=ax)
# %%
# In the following section we take in consideration only ``15C_15C`` and ``15N_15N`` from Pretto dataset to see how a model trained on these data performs at classifying Berio-Nono istances with this equalization parameters.
from ml.datasets import load_dataset

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    'SVM Confusion matrix (varying noise type) with speed at 15 ips and correct equalization'
)

for ax, noise_type in zip(axes.flatten(), ['A', 'B', 'C', None]):
    X_train, y_train = load_dataset(noise_type=noise_type,
                                    return_X_y=True,
                                    purpose='classification',
                                    letter='K')
    test = load_dataset(letter='F',
                        noise_type=noise_type,
                        purpose='classification')
    test = test[(test.label == '15C_15C') | (test.label == '15N_15N')]

    X_train = X_train.drop(columns=['noise_type'], axis=1)
    X_test = test.drop(columns=['noise_type', 'label'], axis=1)
    y_test = test.label
    y_test = y_test.replace({'15C_15C': 'CCIR', '15N_15N': 'NAB'})
    labels = y_train.unique()

    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=
        f'{"All" if noise_type is None else noise_type}, {accuracy_score(y_test, y_pred):.3f}',
        ax=ax)
# %%
# Even if the results are not so good, it is interesting to note that a model trained an validated on both datasets (Pretto and Berio-Nono) gives better results than a model trained and validated only on Pretto dataset.
from ml.datasets import load_pretto, load_berio_nono
from ml.visualization import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from ml.classification import evaluate_model

SEED = 42
data1 = load_pretto()
data2 = load_berio_nono()

data = pd.concat([data1, data2])
X = data.drop(columns=['noise_type', 'label'], axis=1)
y = data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rfc = evaluate_model(RandomForestClassifier(random_state=SEED),
                     X_train,
                     y_train,
                     params={
                         'n_estimators': [x for x in range(90, 121, 10)],
                         'max_features': ['log2'],
                         'criterion': ['log_loss'],
                         'min_samples_leaf': [1],
                     })
y_pred = rfc.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(y_test,
                      y_pred,
                      labels=y_train.unique(),
                      title=f'All, {accuracy_score(y_test, y_pred):.3f}',
                      ax=ax)

# %%
# Speed
# +++++
#
# Another important classification concerns the speed recognition. Here in the same way as above we train a model over Pretto dataset generatin a new label for speed (7.5 or 15 ips) and see if the model recognizes those speed in Berio-Nono data.
# In particular we filter Pretto dataset to get only instances where:
# - speed was only 7.5 or 15 ips for both writing and reading operations
# - equalization curves are the same for pre and post emphasis
# (so we get only the following list of labels: ``['7C_7C', '7N_7N', '15C_15C', '15N_15N']``, it is a subset of dataset :math:`L`).
from sklearn.metrics import classification_report

fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
fig.suptitle(
    'RF Confusion matrix (varying noise type) for speed classification')

for ax, noise_type in zip(axes.flatten(), ['A', 'B', 'C', None]):
    train = load_pretto(filters={'noise_type': noise_type})
    train = train[(train.label == '15C_15C') | (train.label == '15N_15N') |
                  (train.label == '7C_7C') | (train.label == '7N_7N')]
    test = load_berio_nono(filters={'noise_type': noise_type})
    mapping = {
        '7C_7C': "7.5 ips",
        '7N_7N': "7.5 ips",
        '15C_15C': "15 ips",
        '15N_15N': "15 ips"
    }
    train = train.replace(mapping)
    test = test.replace(mapping)

    X_train = train.drop(columns=['noise_type', 'label'], axis=1)
    y_train = train.label
    X_test = test.drop(columns=['noise_type', 'label'], axis=1)
    y_test = test.label

    rfc = evaluate_model(RandomForestClassifier(random_state=SEED),
                         X_train,
                         y_train,
                         params={
                             'n_estimators': [x for x in range(90, 121, 10)],
                             'max_features': ['log2'],
                             'criterion': ['log_loss'],
                             'min_samples_leaf': [1],
                         })
    y_pred = rfc.predict(X_test)

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=y_train.unique(),
        title=
        f'{"All" if noise_type is None else noise_type}, {accuracy_score(y_test, y_pred):.3f}',
        ax=ax)
    print(f"Classification report for {noise_type or 'All noise types'}")
    print(classification_report(y_test, y_pred, target_names=y_train.unique()))
# %%
