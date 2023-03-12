import os
import json
import tempfile
from src.cli import files


def test_get_file_list():
    file_list = files.get_files_list("tests/test_sound")
    assert len(file_list) == 2
    assert file_list[1][0] == "tests/test_sound/test.wav"
    assert file_list[1][1] == "test"


def test_get_file_list_file():
    file_list = files.get_files_list("tests/test_sound/test.wav")
    assert len(file_list) == 1
    assert file_list[0][0] == "tests/test_sound/test.wav"
    assert file_list[0][1] == "test"


def test_get_file_list_wrong_path():
    file_list = files.get_files_list("tests/test_sound/wrong")
    assert len(file_list) == 0


def test_save_as_json():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test.json")
        files.save_as_json({"test": 1}, path)

        with open(path, 'r', encoding='utf-8') as fp:
            assert json.load(fp) == {"test": 1}
