
Classification
--------------

.. note::

   Classification results can be reproduced by executing the ``equalization_classification`` notebook in the ``notebooks`` section of the `thesis' repository <https://github.com/matteospanio/mpai-audio-analyser>`_, where a detailed written and graphic explanation of the code for the process is provided.

As for clustering analysis, the classification task has also been performed on different subsets of the datasets. The classification has been carried out using the K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees (DT) and Random Forest (RF) algorithms. The first three methods have already been used in previous investigations, while in this study, the Random Forest is introduced.

To evaluate the performance of the classification algorithms, the accuracy score has been used, which is the ratio of the number of correctly classified samples to the total number of samples.

.. topic:: About the accuracy scoring

    The disadvantage of using the accuracy score is that it is insensible to heavily unbalanced dataset. For example, if in a classification task, 99% of the data belongs to class :math:`A`, and 1% of the data belongs to class :math:`B`, if the classifier puts all :math:`B` istances in :math:`A`, the accuracy will return a score of 0.99 (almost perfect classification) but, in reality, the classifier put everithing in only one class (so it is completely useless).

The classification has always been performed tuning the hyperparameters of the model using a grid search with cross-validation at 5 folds on 80% of the selected dataset. Once the parameters have been tuned, the model's performance is tested over the remaining 20% of the data.

In summary, the datasets are listed in :numref:`classification-datasets`. Since those datasets are used to train classification models, they are all subsets of the Pretto dataset, which comprehends more cases of wrong and correct equalizations and speeds.

.. _classification-datasets:

.. csv-table:: Classification Datasets
   :header: "Dataset", "Speeds", "Equalization", "Classes"
   :widths: 5, 10, 15, 5

    "H", "7.5 ips", "Mixed and correct", "4"
    "I", "15 ips", "Mixed and correct", "4"
    "J", "7.5 ips", "correct", "2"
    "K", "15 ips", "correct", "2"
    "L", "7.5, 15 ips", "Mixed and correct", "2"

The results obtained from perform classification over the 20% test data of each dataset subset are reported in :numref:`knn-classification-results`, :numref:`svm-classification-results`, :numref:`dt-classification-results` and :numref:`rf-classification-results`.

The classification results show that all the models have very good performance on the test data, with accuracy scores ranging from 0.91 to 1.0 depending on the dataset and the algorithm used. The K-Nearest Neighbors and Support Vector Machines algorithms generally have the best performance, with accuracy scores of 0.98 or higher for all datasets except for Dataset H with KNN, where the accuracy score is 0.94. The Decision Tree and Random Forest algorithms have slightly lower performance than KNN and SVM, but still have accuracy scores above 0.9 for all datasets.

.. _knn-classification-results:

.. csv-table:: KNN Classification results
   :header: "Dataset", "Noise A", "Noise B", "Noise C", "All"
   :widths: 10, 10, 10, 10, 10

   "H", "0.94", "0.91", "1.0", "0.99"
   "I", "1.0", "1.0", "1.0", "0.98"
   "J", "0.88", "0.96", "1.0", "1.0"
   "K", "1.0", "1.0", "1.0", "1.0"
   "L", "1.0", "0.99", "1.0", "1.0"

.. _svm-classification-results:

.. csv-table:: SVM Classification results
   :header: "Dataset", "Noise A", "Noise B", "Noise C", "All"
   :widths: 10, 10, 10, 10, 10

   "H", "0.88", "0.93", "1.0", "0.99"
   "I", "1.0", "1.0", "1.0", "0.99"
   "J", "1.0", "1.0", "1.0", "1.0"
   "K", "1.0", "1.0", "1.0", "1.0"
   "L", "1.0", "1.0", "1.0", "1.0"

.. _dt-classification-results:

.. csv-table:: Decision Tree Classification results
   :header: "Dataset", "Noise A", "Noise B", "Noise C", "All"
   :widths: 10, 10, 10, 10, 10

   "H", "0.79", "0.71", "0.98", "0.91"
   "I", "0.95", "0.92", "0.96", "0.93"
   "J", "1.0", "0.87", "1.0", "0.94"
   "K", "0.95", "1.0", "1.0", "0.95"
   "L", "0.99", "0.93", "0.98", "0.97"

.. _rf-classification-results:

.. csv-table:: Random Forest Classification results
   :header: "Dataset", "Noise A", "Noise B", "Noise C", "All"
   :widths: 10, 10, 10, 10, 10

   "H", "0.88", "0.87", "0.99", "0.97"
   "I", "1.0", "0.98", "0.96", "0.98"
   "J", "0.88", "0.91", "1.0", "1.0"
   "K", "1.0", "0.94", "1.0", "0.99"
   "L", "1.0", "1.0", "0.99", "1.0"

A step further
++++++++++++++

The studies by :cite:`Micheloni2017AST` and :cite:`10.1162/comj_a_00487` have been confirmed, and also the introduction of speed classification (subset :math:`L`) gave an exciting result, allowing us to make a step further and test the classification task on wider datasets:

- study the classification performance on the entire Pretto dataset, giving as input all the possible combinations of speed and equalization curves (25 classes)
- test the effectiveness of the trained classifiers on Berio-Nono dataset

In this case, based on the previous results, it was decided to use only the KNN and RF algorithms since the former provided very similar results to those obtained with SVM and has a much lower training time, while the latter proved to be clearly more effective than a single decision tree and still has a relatively fast training time.

.. plot:: pyplots/classification_25_pretto.py
    :caption: Confusion matrix of the classification of the Pretto dataset for Random Forest algorithm.

The results of the classification of the 25 possible combinations of tape speed and equalization curves are once again very promising. Despite a 4-point difference in the third decimal place in terms of accuracy, both models are capable of correctly classifying most of the samples, even for the less represented classes in the dataset. The confusion matrix of the classification of the Pretto dataset for the Random Forest algorithm is reported in :numref:`classification_25_pretto`. The classification's accuracy has decreased by about 10% compared to the results on the subsets of the dataset, but this is still a very good result, considering that the dataset is much larger and that the classes are not balanced.

The next step was to use the models trained on the entire Pretto dataset to classify the Berio-Nono data. In this case, the process was reversed: first, the model trained on the entire Pretto dataset (using a Random Forest) was evaluated. Then, due to its inefficiency, the possible classes were limited by considering the subsets shown in :numref:`classification-datasets` (in this case, returning to SVMs), and models were evaluated on individual silence classes (only silences :math:`A`, :math:`B`, or :math:`C`). The confusion matrix of the classification of the Berio-Nono dataset using a Random Forest trained on the Pretto dataset is reported in :numref:`classification_25_pretto_berio_nono`. The results were completely wrong. However, it can be seen that the equalization curves are not recognized in any way, while the model seems to roughly recognize the correct speeds (although much less accurately than the models seen previously).

.. plot:: pyplots/classification_25_pretto_berio_nono.py
   :caption: Confusion matrix of the classification of the Berio-Nono dataset using a Random Forest trained on the Pretto dataset.

A better result has been obtained using subsets of the Pretto dataset, but of course, limiting the data also means limiting the possibilities of making mistakes. Classifying 25 categories of data results in :math:`25^2` possibilities between input and output, while simply dealing with the equalizations of a speed (e.g. subset :math:`H``) is equivalent to choosing between :math:`4^2` possible combinations. It goes without saying that limiting the scope of action can increase accuracy, but it also reduces usefulness.

.. csv-table:: accuracy test on Berio-Nono dataset using models trained on Pretto subsets
   :header: "Train set", "Algorithm", "Accuracy Score"
   :widths: 15, 15, 15

   "whole Pretto", "Random Forest", "0.01"
   "H", "SVM", "0.12"
   "I", "SVM", "0.01"
   "J", "SVM", "0.42"
   "K", "SVM", "0.58"
   "L", "SVM", "0.74"

Overall, all algorithms have performed really well on the training/validation set, but testing them on a different dataset did not yield any results. Even though the datasets' structures and the analysis objects are the same, a model trained on one dataset cannot be used to classify the other one. This fact underlines the insufficient amount of data for the analysis. As seen in :numref:`features_space`, the samples belonging to different datasets are very far apart from each other in space, making it very difficult to accurately classify Berio-Nono samples with models trained on Pretto. The result might be surprising as both cases involve magnetic tapes with different equalization and speed curves, but evidently some parameters were not taken into consideration during data acquisition or analysis.

It is important to note that combining the datasets together and training a model on the whole dataset gives really good results, as those given by the algorithm on a single dataset. In :numref:`classification_all`, the confusion matrix of the classification of the tapes from both the Pretto and the Berio-Nono datasets is reported. Even though the data is not balanced, the model is able to classify the tapes with good accuracy, revealing once again that a good direction to follow for further analysis could be to enlarge the dataset.

.. plot:: pyplots/classification_all.py
   :caption: Confusion matrix of the classification of the union of the Pretto and the Berio-Nono datasets.

It would be interesting to conduct an analysis taking into consideration additional tapes and see the performance of this model on them. If the model is able to classify correctly, then it could be thought that the data collected is sufficient. Otherwise, it will be necessary to increase the dataset size until all points in the feature space are considered.