UNAME := $(shell uname)
POETRY := poetry run
DOCS := cd docs && $(POETRY) make clean

ifeq ($(UNAME), Linux)
	OPEN = xdg-open
endif
ifeq ($(UNAME), Darwin)
	OPEN = open
endif
ifeq ($(UNAME), Windows)
	OPEN = start
endif

.PHONY: help clean docs

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

install:
	poetry install --without notebooks

install-notebooks:
	poetry install --only notebooks

install-all:
	poetry install

test:
	$(POETRY) pytest
	cd docs && $(POETRY) make doctest

test-coverage:
	$(POETRY) pytest --cov-config .coveragerc --cov-report term-missing --cov-report html --cov=src

format:
	$(POETRY) yapf --in-place --recursive ./src ./tests ./examples/gallery ./scripts

lint:
	$(POETRY) pylint ./src ./tests ./scripts

docs:
	$(DOCS) && $(POETRY) make html && $(OPEN) build/html/index.html

docs-pdf:
	$(DOCS) && $(POETRY) make latexpdf && $(OPEN) build/latex/eq-detection.pdf
	
docs-live:
	$(DOCS) && make livehtml
