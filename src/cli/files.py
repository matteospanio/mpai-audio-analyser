import os
import json
import logging


def save_as_json(obj, path: str) -> None:
    """Save the given object as a json file."""
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(obj, fp, indent=4)


def get_files_list(path: str) -> list[tuple[str, str]]:
    """Get a list of files from the given path"""
    log = logging.getLogger()

    file_list = []

    if os.path.isfile(path):
        filename = os.path.basename(path).split('.')[0]
        file_list.append((path, filename))
    elif os.path.isdir(path):
        for file_path in os.listdir(path):
            complete_filepath = os.path.join(path, file_path)
            if os.path.isfile(complete_filepath):
                filename = os.path.basename(file_path).split('.')[0]
                file_list.append((complete_filepath, filename))
    else:
        log.warning("Check input: cannot read the given path (%s)", path)

    return file_list


def get_json_audio_blocks(path: str) -> list:
    """Get a list of audio blocks from a json file"""

    json_list = []

    for directory in os.listdir(path):
        with open(os.path.join(path, directory, "log.json"),
                  'r',
                  encoding='utf-8') as json_audio_blocks:
            json_list += json.load(json_audio_blocks)

    return json_list
