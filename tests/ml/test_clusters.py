import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import v_measure_score
from ml.clusters import validate, _combinations_generator


def test_combinations_generator():
    combos = list(_combinations_generator({'a': [1, 2], 'b': [3, 4]}))

    assert combos == [{
        'a': 1,
        'b': 3
    }, {
        'a': 1,
        'b': 4
    }, {
        'a': 2,
        'b': 3
    }, {
        'a': 2,
        'b': 4
    }]


class TestValidation():

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    def test_kmeans(self):
        model = KMeans(n_clusters=2, n_init='auto', random_state=5)
        params = {'algorithm': ['lloyd', 'elkan'], 'max_iter': [10, 20]}
        parameters, score = validate(model, self.X, self.y, params,
                                     v_measure_score)

        assert score == 1.0
        assert parameters == {'algorithm': 'lloyd', 'max_iter': 10}

    def test_hclust(self):
        model = AgglomerativeClustering(n_clusters=2)
        params = {
            'linkage': ['ward', 'complete', 'average', 'single'],
            'metric': ['manhattan', 'cosine', 'euclidean', 'l1', 'l2'],
        }
        parameters, score = validate(model, self.X, self.y, params,
                                     v_measure_score)

        assert score == 1.0
        assert parameters == {'linkage': 'ward', 'metric': 'euclidean'}
