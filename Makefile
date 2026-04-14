PY ?= python
PORT ?= 8000
IMAGE ?= runhaoli-creator/latent-agent-team:local

.PHONY: help install lint serve docker async-bench train eval clean

help:
	@echo "make install     - install package"
	@echo "make lint        - ruff check"
	@echo "make serve       - run FastAPI team API on :$(PORT)"
	@echo "make docker      - build local Docker image"
	@echo "make async-bench - run async concurrent benchmark runner"
	@echo "make train       - run full 2-stage training pipeline (SFT + DPO)"
	@echo "make eval        - run evaluation on all benchmarks"
	@echo "make clean       - remove caches and build artifacts"

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

lint:
	ruff check src serve scripts

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port $(PORT) --reload

docker:
	docker build -t $(IMAGE) .

async-bench:
	$(PY) scripts/async_bench.py \
		--benchmark mind2web \
		--split test --limit 200 \
		--comm-mode vq --concurrency 8

train:
	bash scripts/run_train.sh configs/phi3.yaml outputs/phi3_run

eval:
	bash scripts/run_eval.sh configs/phi3.yaml outputs/phi3_run/stage2 results/phi3

clean:
	rm -rf .ruff_cache .pytest_cache **/__pycache__ *.egg-info dist build
