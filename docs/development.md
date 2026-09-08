# Development Guide

## Setup Environment
1. Clone repository.
2. Setup virtual environment: `python -m venv venv`
3. Activate environment.
4. Install dev dependencies: `pip install -e .[dev]` or `pip install -r requirements-dev.txt`

## Running Locally
CLI Mode:
```bash
privacycam cli --config configs/development.yaml
```

GUI Mode:
```bash
privacycam gui --config configs/development.yaml
```

## Testing
Run the test suite using pytest:
```bash
pytest
```
Test fixtures (like sample images) should be placed in `tests/fixtures/`.

## Code Style
We use `ruff` for linting and formatting, and `mypy` for static type checking.
Run checks before submitting PRs:
```bash
ruff check src/
mypy src/
```
