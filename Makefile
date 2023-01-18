NAME := jumpstreet
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)
PORT := 5556
RATE := 10
# DATA := /mnt/data/KITTI/raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data
DATA := /home/spencer/Documents/PercepTech/Technical/jumpstreet/data/TUD-Campus/img1
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
		@echo "  run         run the whole shebang"
		@ECHO "  runreplay   run with replay testing"
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
		$(POETRY) run pytest ./tests/ --cov-report term-missing --cov-fail-under 100 --cov $(NAME)

.PHONY: run_image_broker
run_image_broker: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/broker.py loadbalancing --frontend=5555 --backend=5556

.PHONY: run_replay
run_replay: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/sensor_replay.py -n 1 --host localhost --port 5555

.PHONY: run_detection
run_detection: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/object_detection.py -n 3 \
			--in_host localhost --in_port 5556 --out_host localhost --out_port 5557

.PHONY: run_tracking
run_tracking: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/object_tracking.py \
			--in_host localhost --in_port 5557 --in_bind \
			--out_host localhost --out_port 5558 --out_bind