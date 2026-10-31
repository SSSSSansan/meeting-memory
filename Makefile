.PHONY: setup validate test budget
setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
validate:
	python -m scripts.validate_extraction
test:
	python -m pytest -q
budget:
	python -m scripts.cost_guard
