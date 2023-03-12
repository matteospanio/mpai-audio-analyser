#!/usr/bin/env python
import argparse
import subprocess


def parse_args(parser):
    parser.add_argument("-c",
                        "--container-name",
                        help="docker container name",
                        required=True)
    parser.add_argument("-d",
                        "--data-folder",
                        help="data folder",
                        required=True)
    parser.add_argument(
        "-i",
        "--input",
        help="input file (must be relative path from data folder)",
        required=True)
    parser.add_argument(
        "-o",
        "--output-dir",
        help="output directory (will be created in data folder)",
        required=True)
    parser.add_argument("-l",
                        "--length",
                        help="length of the noise to extract",
                        required=True)
    return parser.parse_args()


def main(args):
    subprocess.run([
        "docker", "run", "-v", f"{args.data_folder}:/data", args.container_name,
        "poetry", "run", "noise-extractor", "-i", f"/data/{args.input}", "-d",
        f"/data/{args.output_dir}", "-l", args.length
    ],
                   check=True)


if __name__ == "__main__":
    myparser = argparse.ArgumentParser(prog="docker-noise-extractor")
    arguments = parse_args(parser=myparser)
    main(args=arguments)
