"""
.. _data_extraction:

Data extraction
===============

.. admonition:: About data generation

    The audio chunks are the output of the software ``noise-extractor`` on the audio files provided by CSC. For a more detailed description of the dataset, see :ref:`datasets`.

In this notebook, data will be extracted from the raw data files and stored in a csv file. In particular we want to extract the first 13 MFCCs from the audio files provided by the CSC. The MFCCs will be stored in a csv file with the following columns:

* `mfcc_1` to `mfcc_13`: the 13 MFCCs
* `label`: a number between 0 and 3 that represents all possible combinations of reading and acquisition conditions
* `noise_type`: the type of noise present in the audio file

The type of noises that are present in the audio files are:

* `A`: for noise between -50 and -63 dB
* `B`: for noise between -63 and -69 dB
* `C`: for noise between -69 and -72 dB (teoretically the limit is -72 dB, but we will take in account also noise below this limit)

Firstly, as illustrated in the pipeline of :numref:`pipeline`, we get the list of audio files that are present in the `data` folder. Those files have been generated using the `noise-extractor` tool. The following analysis takes in consideration files generated via:

.. code-block:: console

    noise-extractor -i data/audio_samples -d data/dataset/pretto -l 500
    noise-extractor -i data/audio_samples -d data/dataset/berio_nono -l 500
"""
# %%
# Then the files are collected in two lists:
import os


def collect_audio_files(path):
    return [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
        for f in filenames if f.endswith('.wav')
    ]


audio_pretto_list = collect_audio_files("../data/dataset/pretto")
audio_berio_list = collect_audio_files("../data/dataset/berio_nono")
# %%
# The audio files are then loaded and the MFCCs are extracted. Then all audio features are collected in a list.
import numpy as np
import librosa
import json
import warnings

warnings.filterwarnings('ignore')

with open('../src/ml/datasets/data/config.json') as f:
    test_labels = json.load(f)['targets']['test']['labels']


def extract_mfcc(filepath: str) -> list[float]:
    """Get the mean of the first 13 mfccs of a given audio file."""
    audio, sr = librosa.load(filepath, sr=96000)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    mean_mfccs = []
    for e in mfccs:
        mean_mfccs.append(np.mean(e))
    return mean_mfccs


def collect_features(audio_list: list[str], dset: str = 'berio'):
    """Collects the features of all audio files in the given list."""
    labels = []
    noise_type = []
    mfccs = []
    for audiofile in audio_list:
        # extract mfccs
        mfccs.append(extract_mfcc(audiofile))

        # extract labels
        folder_name = audiofile.split('/')[4]
        if dset == 'pretto':
            folder_name = folder_name.split('_')
            write = folder_name[0][1::]
            read = folder_name[1][1::]

        elif dset == 'berio':
            write = read = test_labels[folder_name]
        labels.append(f'{write}_{read}')

        # extract silence type
        noise_type.append(os.path.basename(audiofile).split('_')[0])
    return labels, mfccs, noise_type


pretto_features = collect_features(audio_pretto_list, "pretto")
berio_features = collect_features(audio_berio_list, "berio")
# %%
# The features are then stored in a csv file.
import pandas as pd


def create_dataframe(features, name):
    labels, mfccs, noise_type = features
    data = []
    for label, noise_type, mfccs in zip(labels, noise_type, mfccs):
        data.append([label, noise_type, *mfccs])

    headers = [
        'label', 'noise_type', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
        'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
        'mfcc13'
    ]
    dataframe = pd.DataFrame(data, columns=headers)
    dataframe.to_csv(f'../data/{name}', index=False)


create_dataframe(pretto_features, 'pretto.csv')
create_dataframe(berio_features, 'berio.csv')
