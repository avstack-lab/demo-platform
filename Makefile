NAME := jumpstreet
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)
PORT := 5556
RATE := 30
DATA := ./data/ADL-Rundle-6/img1
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
		@echo "  runreplay   run with replay testing"
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

.PHONY: replay
replay: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/replay.py -n 1 --send_dir $(DATA) --send_rate $(RATE)\
		 	--host 127.0.0.1 --port 5550

.PHONY: data_broker
data_broker: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/broker.py \
			lb_with_xsub_extra_xpub --frontend 5550 \
			--backend 5551 --backend_other 5552

.PHONY: detection_workers
detection_workers: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/object_detection.py -n 1 \
			--in_host localhost --in_port 5551 \
			--out_host localhost --out_port 5553 \
			--model fasterrcnn

.PHONY: tracking_worker
tracking_worker: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/object_tracking.py \
			--in_host localhost --in_port 5553 --in_bind \
			--out_host localhost --out_port 5554 --out_bind

.PHONY: frontend
frontend: $(INSTALL_STAMP)
		$(POETRY) run python jumpstreet/frontend/simple.py \
			--host localhost --port_tracks 5554 --port_images=5552