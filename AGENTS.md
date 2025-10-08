# dspy-code-agent overview

- dspy-code-agent implements an dspy AI software engineer agent that solves github issues and similar programming challenges. 
- We forked a open-source AI-software engineering agent repo which is benchmarked with SWE-bench

The project is structured as

```bash
minisweagent/__init__  # Protocols/interfaces for all base classes
minisweagent/agents  # Agent control flow & loop & 
minisweagent/environments  # Executing agent actions
minisweagent/models  # LM interfaces
minisweagent/run  # Run scripts that serve as an entry point
minisweagent/tools # Tools for the DSPy coding agent
```

# Repository Guidelines

- The project embraces polymorphism: Every individual class should be simple, but we offer alternatives
- Every use case should start with a run script, that picks one agent, environment, and model class to run

  ## Project Structure & Module Organization
  - `src/minisweagent/` holds the agent runtime, with subpackages for `agents`, `config`, `environments`, `models`, `run`, `tools`, and shared
  `utils`.
  - Tests mirror the source layout under `tests/`, using `tests/test_data/` for fixtures and golden files.
  - Reusable scripts live in `scripts/`, documentation in `docs/`, examples in `examples/`, and configuration defaults in `src/minisweagent/
  config/`.
  - CLI entry points (e.g., `mini-dspy`, `mini-swe-agent`) resolve through `pyproject.toml` and expect modules in `src/minisweagent/run/`.

  ## Build, Test, and Development Commands
  - `python -m pip install -e .[dev]` sets up the project with development dependencies.
  - `python -m minisweagent.run.dspy_agent -t "Implement feature"` runs the DSPy agent with the default config.
  - `pytest` runs the full test suite; add `-k name` to focus tests or `-m "not slow"` to skip slow markers.
  - `ruff check .` or `ruff check . --fix` enforces formatting and lint rules defined in `pyproject.toml`.

  ## Coding Style & Naming Conventions
  - Use 4-space indentation, 120-character line limit, and double-quoted strings to match the Ruff formatter profile.
  - Module, package, and file names use `snake_case`; class names are `PascalCase`; constants are `UPPER_SNAKE_CASE`.
  - Type hints are required for public APIs. Prefer pathlib over `os.path` (PTH rules) and avoid mutable defaults (B006).

  ## Testing Guidelines
  - Author tests with `pytest`, keeping source and test file names aligned (e.g., `tests/agents/test_dspy.py` for `src/minisweagent/agents/
  dspy.py`).
  - Mark slow scenarios with `@pytest.mark.slow`; default runs should stay fast.
  - Use fixtures in `tests/conftest.py` or `tests/test_data/` to avoid duplicating setup, and keep coverage stable when modifying critical paths.

  ## Commit & Pull Request Guidelines
  - Follow the existing history pattern: descriptive prefixes such as `[feat]:`, `Fix:`, or `Doc:` paired with a concise summary (e.g., `[feat]:
  add DSPy planner toolchain`).
  - Branches and PRs should map to a focused change. Provide a clear description, link related issues, and include repro or CLI output when
  relevant.
  - Ensure lint and tests pass locally before opening a PR, and highlight config or dependency changes so reviewers can verify their environment.

  ## Agent Configuration Tips
  - Default agent settings live in `src/minisweagent/config/dspy.yaml`; override with `-c path/to/custom.yaml` in CLI runs.
  - Models are resolved via LiteLLM adapters (see `minisweagent/models/`); verify API keys via `.env` or environment variables before invoking
  remote endpoints.

# Style guide

1. Target python 3.10 or higher
2. Use python with type annotations. Use `list` instead of `List`.
3. Use `pathlib` instead of `os.path`. Use `Path.read_text()` over `with ...open()` constructs.
4. Use `typer` to add interfaces
5. Keep code comments to a minimum and only highlight particularly logically challenging things
6. Do not append to the README unless specifically requested
7. Use `jinja` for formatting templates
8. Use `dataclass` for keeping track config
9. Do not catch exceptions unless explicitly told to.
10. Write concise, short, minimal code.
11. In most cases, avoid initializing variables just to pass them to a function. Instead just pass the expression to the function directly.
12. Not every exception has to be caught. Exceptions are a good way to show problems to a user.
13. This repository rewards minimal code. Try to be as concise as possible.

Here's an example for rule 11:

```python
# bad
a = func()
Class(a)

# good
Class(func())
```

## Test style

1. Use `pytest`, not `unittest`.
2. <IMPORTANT>Do not mock/patch anything that you're not explicitly asked to do</IMPORTANT>
3. Avoid writing trivial tests. Every test should test for at least one, preferably multiple points of failure
4. Avoid splitting up code in multiple lines like this: `a=func()\n assert a=b`. Instead, just do `assert func() == b`
5. The first argument to `pytest.mark.parametrize` should be a tuple (not a string! not a list!), the second argument must be a list (not a tuple!).

Here's an example for rule 4:

```python
# bad
result = func()
assert result == b

# good
assert func() == b
```
