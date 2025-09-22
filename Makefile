# Makefile for managing the Python project environment, dependencies, and common tasks.
#
# This Makefile provides a set of commands to help developers set up the project,
# run tests, lint the code, and clean up temporary files. It uses Python's built-in
# `venv` module to create a virtual environment and `pip` to manage dependencies.
#
# Usage:
#   make install   - Set up the virtual environment and install all dependencies (including dev dependencies).
#   make test      - Run the test suite using pytest and generate a coverage report.
#   make lint      - Run code linting and style checks using flake8.
#   make clean     - Remove the virtual environment and temporary files.
#
# Note: Ensure you have Python 3 installed on your system before using this Makefile.
.PHONY: install test lint clean
install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt
test:
	. .venv/bin/activate && pytest --cov
lint:
	. .venv/bin/activate && flake8
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -r {} +
