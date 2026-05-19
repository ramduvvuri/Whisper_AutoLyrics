install:
	pip install -e .

train:
	python src/autolyrics/training/train.py

test:
	pytest tests/

lint:
	ruff check src/
