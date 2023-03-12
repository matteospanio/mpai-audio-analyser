# noise-extractor runner

import os
import sys
import math
import logging
import concurrent.futures as cf
from argparse import ArgumentParser, Namespace

from audiohandler import Noise
from ..files import save_as_json, get_files_list
from .extractor import extract_noise


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        help="Specify a single file or a folder containing audio files as input",
        required=True)
    parser.add_argument("-d",
                        "--destination-dir",
                        help="Where to put file chunks",
                        required=True)
    parser.add_argument(
        "-l",
        "--silence-len",
        help="The duration in milliseconds of silence to detect",
        required=True,
        type=int)
    parser.add_argument("-q",
                        "--quiet",
                        help="Display less output (errors only)",
                        action="store_true")


def main(args: Namespace):
    log = logging.getLogger()

    destination: str = args.destination_dir

    if not os.path.isdir(destination):
        os.makedirs(destination)

    if os.listdir(destination) != []:
        log.warning(
            "Destination directory is not empty, files may be overwritten")

    log.info("Reading files")
    file_list = get_files_list(args.input)

    noise_list = [
        Noise("A", -50, -63),
        Noise("B", -63, -69),
        Noise("C", -69, -72)
    ]

    with cf.ThreadPoolExecutor(max_workers=math.ceil(os.cpu_count() /
                                                     2)) as executor:
        futures = [
            executor.submit(extract_noise, filepath, noise_list,
                            args.silence_len, filename, destination)
            for filepath, filename in file_list
        ]
        results = [f.result() for f in futures]

    log.info("Creating json logfiles")
    for destination_dir, result in results:
        save_as_json(result, f"{destination_dir}/log.json")


def run(args=None):
    parser = ArgumentParser(
        prog="noise-extractor",
        description="A tool to extract noise from audio files")
    add_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                        level=logging.ERROR if args.quiet else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    main(args)


if __name__ == "__main__":
    run()
