# Makefile for Warp Bubble QFT

.PHONY: help install test lint docs clean demo

help:
	@echo "Available targets:"
	@echo "  install    - Install package in development mode"
	@echo "  test       - Run test suite"
	@echo "  lint       - Run code linting"
	@echo "  docs       - Build documentation"
	@echo "  clean      - Clean build artifacts"
	@echo "  demo       - Run demonstration script"

install:
	pip install -e .

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=warp_qft --cov-report=html

lint:
	flake8 src/warp_qft/
	black --check src/warp_qft/

format:
	black src/warp_qft/

docs:
	cd docs && pdflatex polymer_field_algebra.tex
	cd docs && pdflatex warp_bubble_proof.tex

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo:
	cd examples && python demo_warp_bubble_sim.py

.PHONY: all
all: install test lint
