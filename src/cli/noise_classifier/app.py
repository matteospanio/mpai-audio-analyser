# noise-classifier runner

import os
import sys
import json
import logging
# import concurrent.futures as cf
from argparse import ArgumentParser, Namespace

from .classifier import classify, extract_features, write_irregularity_file


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        help="Specify a single file or a folder containing audio files as input",
        required=True)
    parser.add_argument("-d",
                        "--destination-dir",
                        help="Where to put classification results",
                        required=True)
    parser.add_argument("-q",
                        "--quiet",
                        help="Display less output (errors only)",
                        action="store_true")


def main(args: Namespace):
    log = logging.getLogger()

    destination: str = args.destination_dir

    if not os.path.isdir(destination):
        os.makedirs(destination)

    for folder in os.listdir(args.input):
        if not os.path.isdir(os.path.join(args.input, folder)):
            log.error("Input folder contains files, please remove them")

        destination = os.path.join(args.destination_dir, folder)

        log.info("Reading files")
        with open(os.path.join(args.input, folder, "log.json"),
                  'r',
                  encoding='utf-8') as json_audio_blocks:
            json_audio_blocks = json.load(json_audio_blocks)

        os.remove(os.path.join(args.input, folder, "log.json"))

        log.info("Extracting features for %s", folder)
        dataframe = extract_features(json_audio_blocks)

        log.info("Classifying %s", folder)
        classification_results = classify(dataframe)

        log.info("Writing irregularity file for %s", folder)
        write_irregularity_file(json_audio_blocks, classification_results,
                                destination)


def run(args=None):
    parser = ArgumentParser(
        prog="noise-classifier",
        description=
        "A tool to classify audio files equalization and speed based on MFCCs")
    add_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                        level=logging.ERROR if args.quiet else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    main(args)


if __name__ == "__main__":
    run()
