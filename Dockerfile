FROM python:3.10.0 as python-base

LABEL maintainer="Matteo Spanio" version="0.3.7a"

# https://python-poetry.org/docs#ci-recommendations
ENV POETRY_VERSION=1.3.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv

# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

# Create stage for Poetry installation
FROM python-base as poetry-base

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Create a new stage from the base python image
FROM python-base as example-app

# Copy Poetry to app image
COPY --from=poetry-base ${POETRY_VENV} ${POETRY_VENV}

# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY ./src ./Makefile ./poetry.lock ./pyproject.toml ./README.md ./

RUN poetry install --no-cache --only main

VOLUME [ "/data" ]

CMD ["poetry", "run", "noise-extractor", "-h"]
