install:
	pip install -e ".[dev]"

generate:
	python -m src.cli generate

eda:
	python -m src.cli eda

train:
	python -m src.cli train

run:
	python -m src.cli run-all

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	black --check src/ tests/
