.PHONY: setup install clean data analyze notebook help

help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make install    - Install package in development mode"
	@echo "  make clean      - Remove build artifacts and cache files"
	@echo "  make data       - Download data files"
	@echo "  make analyze    - Run analysis and generate report"
	@echo "  make notebook   - Start Jupyter notebook server"

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

install:
	pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

data:
	python download_data.py

analyze:
	python analyze_shopify.py

notebook:
	jupyter notebook notebooks/ 