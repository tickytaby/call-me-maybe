install:
	uv sync

run:
	uv run python -m src $(ARGS)

regex:
	uv run python3 -m src --functions_definition ./data/input/functions_definition.json --input ./data/input/regex.json --output ./data/output/regex_output.json

debug:
	uv run python -m pdb -m src $(ARGS)

test:
	uv run python test/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src/
	mypy src/ --strict

.PHONY: install run regex debug test clean lint lint-strict
