NAME := jumpstreet
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)
.DEFAULT_GOAL := help

.PHONY: help
help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  install     install packages and prepare environment"
		@echo "  clean       remove all temporary files"
		@echo "  lint        run the code linters"
		@echo "  format      reformat code"
		@echo "  test        run all the tests"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

install: $(INSTALL_STAMP)
$(INSTALL_STAMP): pyproject.toml poetry.lock
		@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
		$(POETRY) install
		touch $(INSTALL_STAMP)

.PHONY: clean
clean:
		find . -type d -name "__pycache__" | xargs rm -rf {};
		rm -rf $(INSTALL_STAMP) .coverage .mypy_cache

.PHONY: lint
lint: $(INSTALL_STAMP)
		$(POETRY) run isort --profile=black --lines-after-imports=2 --check-only ./tests/ $(NAME)
		$(POETRY) run black --check ./tests/ $(NAME) --diff
		$(POETRY) run flake8 --ignore=W503,E501 ./tests/ $(NAME)
		$(POETRY) run mypy ./tests/ $(NAME) --ignore-missing-imports
		$(POETRY) run bandit -r $(NAME) -s B608

.PHONY: format
format: $(INSTALL_STAMP)
		$(POETRY) run isort --profile=black --lines-after-imports=2 ./tests/ $(NAME)
		$(POETRY) run black ./tests/ $(NAME)

.PHONY: test
test: $(INSTALL_STAMP)
		$(POETRY) run pytest ./tests/ --cov-report term-missing --cov-fail-under 0 --cov $(NAME)

.PHONY: demo_platform
demo_platform: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/controllers/demo_platform.py

.PHONY: replay
replay: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/sensors/run.py \
			--config sensors/MOT15-replay.yml \
			--sensor_id camera_1

.PHONY: flir
flir: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/sensors/run.py \
			--config sensors/FLIR-BFS-50S50C.yml \
			--sensor_id camera_1

.PHONY: radar
radar: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/sensor.py \
			--sensor_type ti-radar \
			--config radar_1 \
			--host 127.0.0.1 \
			--backend 5550 \
			--verbose \