MODEL_SIZE        ?= tiny
DEVICE            ?= cpu
MODEL_CACHE_DIR   ?= $(HOME)/.cache/huggingface/hub

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

run:
	MODEL_SIZE=$(MODEL_SIZE) DEVICE=$(DEVICE) MODEL_CACHE_DIR=$(MODEL_CACHE_DIR) \
		uvicorn main:app --host 0.0.0.0 --port 9090 --app-dir $(ROOT)server

test:
	python -m pytest tests/ -v
