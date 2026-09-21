# Repository Guidelines

## Project Structure

- `abraia/` contains the Python SDK, organized into inference, runtime, training, editing, and utility modules.
- `tests/` contains the pytest suite, with files named `test_<feature>.py`.
- `images/` stores sample assets used by examples and tests; `notebooks/` contains exploratory training material.
- `setup.py` and `requirements.txt` define packaging and runtime dependencies. CI configuration is in `.github/workflows/build.yml`.

## Build, Test, and Development Commands

Create an isolated environment and install dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the full test suite with coverage, matching CI:

```sh
pytest -v tests/ --cov=abraia
```

Run a focused test file with `pytest -q tests/test_inference.py`. Build source and wheel distributions with `python setup.py sdist bdist_wheel`.

## Coding Style and Naming

Use four-space indentation and standard Python naming: `snake_case` for functions, methods, and modules; `CamelCase` for classes; and `UPPER_CASE` for constants. Keep imports explicit and preserve the package’s public exports in the relevant `__init__.py`. No repository-wide formatter or linter is configured, so follow the surrounding code and keep changes focused.

## Testing Guidelines

Add tests under `tests/` using `test_*.py` filenames and `test_*` functions. Prefer deterministic unit tests with mocked network, model, camera, and accelerator boundaries. Run the smallest relevant test first, then the full suite before submitting. CI runs on Python 3.8 and supplies `ABRAIA_ID` and `ABRAIA_KEY` for tests that require service access.

## Commit and Pull Requests

Recent commits use short, imperative subjects such as `Fix ...`, `Refactor ...`, and `Merge ...`. Follow that style and keep each commit focused. Pull requests should describe the behavior changed, list validation commands and results, call out dependency or API changes, and link a related issue when applicable. Include screenshots only when a visual or notebook-facing change requires them.
