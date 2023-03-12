
.. _cluster-scoring:

Clustering
----------

.. note::

   Clustering results can be reproduced executing the ``equalization_clustering`` notebook in the ``notebooks`` section of the `thesis' repository <https://github.com/matteospanio/mpai-audio-analyser>`_, where a detailed written and graphic explanation of the code for the process is provided.

To evaluate the quality of the data and the feasibility of the classification task, the dataset has been clustered using the K-Means and Hierarchical clustering algorithms as in :cite:`Micheloni2017AST`.

As a measure of the quality of the clustering, the V-measure has been used. The V-measure is a measure of the similarity between two data clusterings, and it is defined as the harmonic mean between the homogeneity and completeness of the clustering. Homogeneity is the ratio between the number of pairs of samples that are in the same cluster in both data clusterings and the total number of pairs of samples that are in the same cluster in the first data clustering. Completeness is the ratio between the number of pairs of samples that are in the same cluster in both data clusterings and the total number of pairs of samples that are in the same cluster in the second data clustering. :cite:`scikit-learn`

.. topic:: About the cluster scoring

   There are many metrics to evaluate the quality of a clustering, anyway not always are all applicable to the specific clustering task. For example, the Davies-Bouldin index is a metric that evaluates the quality of a clustering by measuring the distance between clusters and the distance between points in the same cluster. This metric is not applicable to the equalization classification task since the clusters are often overlayed. In addition is useful to observe that in this specific case the ground truth is know in advance, so all the metrics that take advantage of this information are a good choice. Lastly the V-measure returns a value in the range :math:`[0, 1]` which make really easy to treat the results as if they were the accuracy of a classifier. For a more detailed explanation of the metrics see `this article <https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation>`_.

Since the data classes are overlayed the clustering and algorithm evalutation have been performed on different subsets of the dataset structured as shown in :numref:`clustering-datasets`. For each subset has been specified the original dataset, the clustering task, the speeds used and the number of clusters [#f1]_ , i. e. if the speed is "7.5 ips, 15 ips", it means that all the considered classes have been extracted from recorded and played back tapes at 7.5 ips or 15 ips, regardless of the equalization (i.e., ``7N_7N``, ``7C_7C``, ``7C_7N``, ``7N_7C``, ``15N_15N``, ``15C_15C``, ``15C_15N``, ``15N_15C``).

The subsets A and B were constructed based on the datasets built in :cite:`Micheloni2017AST`, where classification and clustering were only performed on data with the recording speed equal to the playback speed. The study demonstrated that, in general, machine learning algorithms achieve similar performance in identifying equalization curves at a single speed. This thesis aimed to go beyond that by introducing the E dataset, which combines samples from both subsets A and B, yielding encouraging results. Furthermore, the clustering of recording speeds is addressed by subsets C, D, and G, a problem not tackled in either :cite:`Micheloni2017AST` or :cite:`10.1162/comj_a_00487`.

There isn't a subset of Berio-Nono dataset to evaluate the clustering of equalization classes (wrong/right equalization) because the tapes have been recorded and played back at the same speed.

.. _clustering-datasets:

.. csv-table:: Clustering Datasets
   :header: "Dataset", "Original dataset", "Clusters", "Speeds", "# Clusters"

   "A", "Pretto", "Right/Wrong equalization", "7.5 ips", "2"
   "B", "Pretto", "Right/Wrong equalization", "15 ips", "2"
   "C", "Pretto", "Speed", "7.5 ips, 15 ips", "2"
   "D", "Pretto", "Speed", "3.75, 7.5 ips, 15 ips", "3"
   "E", "Pretto", "Right/Wrong equalization", "7.5 ips, 15 ips", "2"
   "F", "Berio-Nono", "Speed and equalization", "7.5 ips, 15 ips", "4"
   "G", "Berio-Nono", "Speed", "7.5 ips, 15 ips", "2"

The results of K-Means and Hierarchical clustering are shown respectively in :numref:`kmeans_results` and :numref:`hierarchical_results`. The scores higher than 0.5 are highlighted in bold.

.. _kmeans_results:

.. csv-table:: K-means Clustering results
   :header: "Dataset", "Noise A", "Noise B", "Noise C", "All"
   :widths: 10, 10, 10, 10, 10

   "A", "0.2", "0.086", "**0.589**", "0.166"
   "B", "**0.747**", "0.413", "0.046", "0.301"
   "C", "0.09", "0.015", "0.145", "0.008"
   "D", "0.172", "0.177", "0.123", "0.084"
   "E", "0.377", "0.215", "**0.527**", "0.194"
   "F", "0.176", "0.108", "0.22", "0.156"
   "G", "0.056", "0.008", "0.043", "0.024"

.. _hierarchical_results:

.. csv-table:: Hierarchical Clustering results
   :header: "Dataset", "Noise A", "Noise B", "Noise C", "All"
   :widths: 10, 10, 10, 10, 10

   "A", "0.079", "0.178", "**0.972**", "0.338"
   "B", "**0.948**", "**0.567**", "0.304", "**0.514**"
   "C", "0.27", "0.119", "0.116", "0.05"
   "D", "0.243", "**0.535**", "0.222", "0.094"
   "E", "0.453", "0.395", "**0.760**", "0.268"
   "F", "0.233", "0.145", "0.233", "0.163"
   "G", "0.099", "0.03", "0.043", "0.025"

Subsets A, B, E have been used to evaluate the clustering of equalization classes (:numref:`clusters_distribution`). Overall the clusters resulting by analyzing only tapes recorded and played at the same speed (dataset A and B) gave better results, from this it can be inferred that MFCCs describe in a quite good manner the difference between correctly equalized and non-correctly equalized tapes. The situation changes when clustering of wrong or correct equalization is performed on different speeds (dataset E [#f2]_), in this case the results are not so good. This latter fact is much more evident in the clusterization of subset F, where the clusters should highlight both the equalization and the speed of the tapes.

.. plot:: pyplots/clusters_distribution.py
    :caption: Clusters distribution from dataset A and B. While considering only one speed at a time (7.5 ips or 15) the clustering performance is better, the dataset union generates instead more confusion.

The clustering results are not homogeneous over the different datasets, but, looking for trends, it can be said that, in general, Hierarchical clustering overperforms K-Means and when one algorithm gives an acceptable result also the other one does. Another fact is that good results came across different kind of noises, but the same noise also gave really bad results for other datasets, e. g. Noise C for dataset A and B: in the first one the score was 0.972 with Hierarchical clustering, while in the second one it was 0.304 with the same algorithm. From this fact can be inferred that for a effective classification of the tapes it can be helpful to use different kind of noise, and when an algorithm gives a bad result, it can be useful to evaluate it over another noise.

Another fact is that the results were bad where speed clustering was involved, this is a confirmation over previous studies, but should make us think about the possibility to use other features to cluster the tape's speed. MFCCs are a good choice for equalization clustering, but they are not so good for speed clustering, so it would be interesting to find other features that can be used to cluster the tapes based on their speed.

.. rubric:: Footnotes

.. [#f1] Both K-Means and Hierarchical clustering algorithms require the number of clusters as input, in this case the number of classes to be clustered.

.. [#f2] Before the actual subset E, composed only by tapes recorded and played back at the same speed (7.5 and 15 ips), a preliminary study has been performed on the dataset union, i. e. all the tapes recorded and played back at different speeds, where the results were much worse than the ones obtained in dataset E. Due to drastically unuseful results the dataset E has then been resized to its actual shape.
