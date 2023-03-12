import os
import sys
import logging
import subprocess
from argparse import ArgumentParser, Namespace


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        help="Specify a single file or a folder containing audio files as input",
        required=True)
    parser.add_argument("-d",
                        "--destination-dir",
                        help="Where to put irregularity files",
                        required=True)
    parser.add_argument("-q",
                        "--quiet",
                        help="Display less output (errors only)",
                        action="store_true")


def main(args: Namespace):
    log = logging.getLogger()

    destination: str = args.destination_dir + "/temp"

    if not os.path.isdir(destination):
        os.makedirs(destination)

    if os.listdir(destination) != []:
        log.warning(
            "Destination directory is not empty, files may be overwritten")

    log.info("Generate Audio Blocks")

    noise_extractor = [
        'noise-extractor', '-i', args.input, '-d', destination, '-l', '500'
    ]
    noise_classifier = [
        'noise-classifier', '-i', destination, '-d', destination
    ]

    if args.quiet:
        noise_extractor.append('-q')
        noise_classifier.append('-q')

    subprocess.run(noise_extractor, check=True)

    log.info("Generate Irregularity Files")
    subprocess.run(noise_classifier, check=True)

    log.info("Audio Analysis Complete")


def run(args=None):
    parser = ArgumentParser(
        prog="audio-analyser",
        description="A tool to generate irregularity files from audio files")
    add_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                        level=logging.ERROR if args.quiet else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    main(args)


if __name__ == "__main__":
    run()
