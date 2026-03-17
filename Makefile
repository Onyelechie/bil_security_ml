.PHONY: test

PYTHON ?= python

test:
	$(PYTHON) scripts/run_tests.py
