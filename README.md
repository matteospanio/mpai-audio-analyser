# Eq-Detection

# Installation

The package uses poetry as a dependency manager, so you need to install it first. You can find the installation instructions [here](https://python-poetry.org/docs/#installation).

Then install the package dependencies and executables using make:
```bash
make install
```

> Note that project dependencies are splitted in `docs`, `dev`, `notebooks` and the required ones for the package to work, by default `make install` will install everithing except the `notebooks` dependencies, if you want to install all in once you can use `poetry install`.

## Docker

To avoid any issue with setup and installations a docker container is provided launch the command to create a local container called mpai:
```bash
docker build -t mpai .
```

# Usage

The package come with three executables:

- `noise-extractor`: Extracts noise from audio files and saves it in a folder
- `noise-classifier`: Classifies noise files using a pre-trained model and generates a report in form of Irregularity files
- `audio-analyser`: executes the whole audio analysis pipeline, extracting noise, classifying it and generating the irregularity files

To run the commands from the container use the command:
```bash
docker run -it mpai /bin/bash
```
and then you can run the commands as described below.

Since the container main use is to execute CLI commands the `scripts` folder contains some wrapper scripts to execute the CLI commands from outside the container, to execute the `noise-extractor` command type the command:
```bash
./scripts/docker-noise-extractor.py -h
```

## noise-extractor

Once installed you can run the package using the command from the `poetry shell`[^1]
```bash
noise-extractor -h
# or poetry run noise-extractor -h if you are not in the poetry shell
```
This will output the help message:
```bash
usage: noise-extractor [-h] -i INPUT -d DESTINATION_DIR -l MIN_SILENCE_LEN [-q]

A tool to extract noise from sound files

options:
-h, --help            show this help message and exit
-i INPUT, --input INPUT
                        Specify a single file or a folder containing audio files as input
-d DESTINATION_DIR, --destination-dir DESTINATION_DIR
                        Where to put file chunks
-l SILENCE_LEN, --silence-len SILENCE_LEN
                        The duration in milliseconds of silence to detect
-q, --quiet           Display less output (errors only)
```

## noise-classifier

```bash
noise-classifier -h
# or poetry run noise-classifier -h if you are not in the poetry shell
```

## audio-analyser

```bash
audio-analyser -h
# or poetry run audio-analyser -h if you are not in the poetry shell
```

# Docs
A more detailed documentation can be found in the [docs](docs) folder. This docs can be rendered to HTML or PDF using `sphinx`. To read the docs in HTML format type the command `make docs` from the root folder, it will open the system browser on the generated documentation, while `make docs-pdf` will do the same as `make docs` but generates a pdf otput. Otherwise you can run the Makefile in the docs folder and type `make html` or `make latexpdf` to generate the documentation in PDF format (`make help` will output a list of all possible formats), the generated files will be in the `docs/build` folder.

The first time you run the `make docs` command it will take a while to generate the documentation, because it generates the images and code examples by executing some python scripts, but after the first rendering it will be much faster.

## Notebooks
To execute the notebooks you can open them in your favourite way (using conda or similar environments) or install the `notebooks` dependencies and run them using `jupyter` from the poetry virtual environment. To install the `notebooks` dependencies type the command:
```bash
make install-notebooks
```
Then you can run the notebooks using the command:
```bash
poetry run jupyter notebook ./notebooks
```

[^1]: If you are not familiar with poetry, you can find more information [here](https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment).

## TO BE IMPLEMENTED 

1. The offset calculation between audio and video is not implemented yet.
2. The audio-analyser CLI command is generating only one irregularity file for each input file, but it should generate two (one for the Tape Irregularity Classifier and one for the Video Analyser)
3. At the moment Irregularities from the Video Analyser are not being considered.
4. Some tests are missing, be careful when using the package.