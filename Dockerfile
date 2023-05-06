FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

ARG YOUR_ENV
ARG DEBIAN_FRONTEND=noninteractive

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.3.2 \
  POETRY_HOME=/opt/poetry \
  POETRY_VENV=/opt/poetry-venv \
  POETRY_CACHE_DIR=/opt/.cache

# Install python and pip
RUN set -xe \
    && apt-get update \
    && apt-get -y install python3-pip \
    && apt-get -y install python3.8-venv \
    && apt-get -y install ffmpeg libsm6 libxext6 wget vim \
    && apt-get -y install libqt5gui5 \
    && rm -rf /var/lib/apt/lists/*
ENV QT_DEBUG_PLUGINS=1

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Creating folders, and files for a project:
COPY . /code

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") \
  --no-interaction --no-ansi

# python jumpstreet/controller.py --config controller/camera.yml