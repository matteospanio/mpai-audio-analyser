import numpy as np
from src.audiohandler import utils


def test_db_to_pcm():
    assert utils.db_to_pcm(0, 16) == 32768
    assert utils.db_to_pcm(0, 8) == 128
    assert utils.db_to_pcm(0, 4) == 8


def test_pcm_to_db():
    assert utils.pcm_to_db(32768, 16) == 0
    assert utils.pcm_to_db(128, 8) == 0
    assert utils.pcm_to_db(8, 4) == 0
    assert utils.pcm_to_db(0, 16) == -98
    assert utils.pcm_to_db(0, 24) == -146
    assert utils.pcm_to_db(0, 32) == -194


def test_rms():
    assert utils.rms(np.array([1, 2, 3, 4])) == np.sqrt(30 / 4)
    assert utils.rms(np.array([1, 1, 1, 1])) == 1
    assert utils.rms(np.array([0])) == 0


def test_get_last_index():
    assert utils.get_last_index(np.array([1, 2, 3, 4]), 4) == 3
    assert utils.get_last_index(np.array([[1, 2, 3, 4], [1, 2, 3, 5]]),
                                5) == (1, 3)
    assert utils.get_last_index(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
                                5) is None
    assert utils.get_last_index(np.array([]), 5) is None
