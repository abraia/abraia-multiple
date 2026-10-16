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

Prefer the smallest change that solves the request. Keep implementation direct; avoid adding helpers, classes, modules, or compatibility paths unless they are needed. Prefer functions and existing data structures over new classes unless a class meaningfully owns state or behavior. Inline one-use logic when that makes the code simpler, and remove obsolete paths instead of preserving them without a concrete requirement.

Before adding a helper or another code path, search the SDK, client, and `abraia/utils` for an existing function that already performs the operation. Reuse the existing API when its behavior fits; do not duplicate loading, path resolution, metadata lookup, normalization, or validation in a caller. Use the canonical filename or SDK operation directly when it is known instead of listing resources and probing candidates. Keep CLI code as orchestration and let the client, runtime, or utility that owns the behavior handle it. Add a new abstraction only when it removes real duplication or owns behavior used in multiple places.

## Testing Guidelines

Add tests under `tests/` using `test_*.py` filenames and `test_*` functions. Prefer deterministic unit tests with mocked network, model, camera, and accelerator boundaries. Run the smallest relevant test first, then the full suite before submitting. CI runs on Python 3.8 and supplies `ABRAIA_ID` and `ABRAIA_KEY` for tests that require service access.

## Commit and Pull Requests

Recent commits use short, imperative subjects such as `Fix ...`, `Refactor ...`, and `Merge ...`. Follow that style and keep each commit focused. Pull requests should describe the behavior changed, list validation commands and results, call out dependency or API changes, and link a related issue when applicable. Include screenshots only when a visual or notebook-facing change requires them.
