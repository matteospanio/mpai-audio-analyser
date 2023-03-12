import pytest
import pandas as pd
from ml.datasets import load_pretto, load_berio_nono, _filter_dataset, load_config, load_dataset, load_results


def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert 'datasets' in config.keys()
    assert 'results' in config.keys()


def test_load_dataset():
    A, n_clust = load_dataset('A')
    X, y, number = load_dataset('B', noise_type='A', return_X_y=True)

    with pytest.raises(KeyError):
        load_dataset('Z')

    assert n_clust == 2
    assert number == 2
    assert isinstance(A, pd.DataFrame)
    assert len(X) == 319
    assert isinstance(y, pd.Series)


def test_load_results():
    results = load_results()
    assert isinstance(results, dict)
    assert 'datasets' in results.keys()
    assert 'seed' in results.keys()


def test_load_pretto():
    data = load_pretto()
    assert data.shape == (9058, 15)
    assert data.label.nunique() == 25
    assert data.noise_type.nunique() == 3


def test_filter_pretto():
    filters = {'labels': ['7C', '7N'], 'noise_type': 'A', 'combination': True}
    data = load_pretto(filters=filters)
    assert data.shape == (168, 15)
    assert data.label.nunique() == 4
    assert data.noise_type.nunique() == 1


def test_load_berio_nono():
    data = load_berio_nono()
    assert data.shape == (12202, 15)
    assert data.label.nunique() == 4
    assert data.noise_type.nunique() == 3


def test_filter_dataset():
    data = load_pretto()
    filtered_data = _filter_dataset(data, labels=['7C_7C'], noise_type='A')
    assert filtered_data.shape == (42, 15)
    assert filtered_data.label.nunique() == 1
    assert filtered_data.noise_type.nunique() == 1
