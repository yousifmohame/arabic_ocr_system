.PHONY: check test run

check:
	python scripts/check_environment.py

test:
	python -m unittest discover -s tests -v

run:
	python -m arabic_doc_ai.cli $(INPUT)
