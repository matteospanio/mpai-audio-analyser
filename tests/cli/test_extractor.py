import tempfile
import os
import json
import numpy as np
from src.audiohandler import AudioWave, Noise
from cli.noise_extractor import extractor


class TestExtractor:

    audio = AudioWave(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), 16, 2, 44100)
    filename = 'test'
    results = {
        "0": {
            "A": [(0, 1)],
            "B": [],
            "C": [],
        },
        "1": {
            "A": [],
            "B": [],
            "C": [],
        }
    }

    def test_create_indexes_repr(self):

        idxs = extractor.create_indexes_representation(self.results,
                                                       [self.audio],
                                                       self.filename)

        assert isinstance(idxs, list)

        chunk = idxs[0]

        assert chunk['Path'] == 'test/AudioBlocks/A_0_0_1.wav'
        assert chunk['StartTime'] == 0
        assert chunk['EndTime'] == 1 / 44100
        assert chunk['Channel'] == 0

    def test_save_chunks(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor.save_chunks([self.audio], self.results,
                                  f'{tmpdir}/{self.filename}')

            assert os.path.isfile(
                f"{tmpdir}/{self.filename}/AudioBlocks/A_0_0_1.wav")
            assert AudioWave.from_file(
                f"{tmpdir}/{self.filename}/AudioBlocks/A_0_0_1.wav"
            ) == self.audio[0:1]

    def test_extract_noise(self):

        noise_list = [
            Noise("A", -50, -63),
            Noise("B", -63, -69),
            Noise("C", -69, -72)
        ]
        silence_len = 500

        with open('tests/test_sound/test.json', mode='r',
                  encoding='utf-8') as f:
            expected = json.load(f)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, results = extractor.extract_noise('tests/test_sound/test.wav',
                                                 noise_list, silence_len,
                                                 self.filename, tmpdir)

            results = results[0]
            expected = expected[0]

            assert results['StartTime'] == expected['StartTime']
            assert results['EndTime'] == expected['EndTime']
            assert results['NoiseType'] == expected['NoiseType']
            assert results['Channel'] == expected['Channel']
            assert results['RMSdB'] == expected['RMSdB']
