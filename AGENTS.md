# AGENTS.md (Python Project)

## Purpose

This document defines **rules and guidelines for AI coding agents** working in this Python repository. Follow these instructions before suggesting or writing code.

---

## General Development Guidelines

* **Code Style (Identifiers)**

  * Follow [PEP 8](https://peps.python.org/pep-0008/) for code style.
  * Use **clear, descriptive English names** for all variables, functions, and classes.

    * ❌ Do not use abbreviations or short forms.
    * ❌ Do not use single-letter names (`i`, `j`, `n`, etc.), even in loops.
    * ✅ Example: use `soldier_index` instead of `i`, `total_soldiers` instead of `n`.
  * Use `snake_case` for variables and functions, `PascalCase` for classes, and `UPPER_CASE` for constants.
  * Keep functions short and focused; one function should ideally do one thing.
  * **Logic must be straightforward and easy to follow.**

    * Code should read top-to-bottom without hidden tricks.
    * Avoid deeply nested structures, inline one-liners, or overly clever patterns.
    * Always prioritize clarity over brevity.

* **Structure**

  * Organize code into modules and packages logically.
  * Place reusable scripts in `src/` or `lib/` directories, tests in `tests/`.
  * Keep configuration files (e.g., `.env`, `config.yaml`) separate from code.
  * Avoid leaving unused functions, variables, or imports.

* **Documentation**

  * Write **docstrings** for all non-trivial functions, classes, and modules.

    * Use [PEP 257](https://peps.python.org/pep-0257/) conventions.
    * For public APIs, include input/output descriptions and expected types.
  * Add comments to explain *why* code is written a certain way, not just *what* it does.
  * Clearly mark AI-generated code sections with `# TODO: review`.

* **Safety & Reliability**

  * Do not delete existing code unless explicitly requested.
  * Do not introduce breaking API changes without documentation.
  * Use **type hints** (`typing`) for all new code.
  * Handle exceptions explicitly; avoid bare `except`.
  * Use `logging` instead of `print` for runtime messages.

---

## Environment Setup

- Use [uv](https://docs.astral.sh/uv/) for dependency management; avoid invoking `pip` commands directly.
- Run `uv sync` whenever `pyproject.toml` or `uv.lock` changes before executing tests or scripts.
- Activate the project environment with `source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`) for terminal sessions, including VS Code.
- When activation is impractical, prefer `uv run <command>` (`uv run pytest`, `uv run python -m stock_indicator.manage`, etc.).
- Record new dependencies in `pyproject.toml`, regenerate the lock file with `uv lock --upgrade-package <name>` if needed, and commit both files together.

---

## Python-Specific Notes

* **Imports**

  * Use absolute imports where possible.
  * Group imports by: standard library → third-party → local modules.

* **Error Handling**

  * Catch specific exceptions, not broad ones.
  * Log errors with enough context for debugging.

* **Testing**

  * Write unit tests for new functionality.
  * Place all tests under the `tests/` directory.
  * Prefer [pytest](https://docs.pytest.org/) style with clear test function names.

* **Performance**

  * Avoid unnecessary loops; prefer list comprehensions and built-ins when readable.
  * Be cautious with large memory usage; consider generators for big data.

---

## UI & Localization (If Applicable)

If this project has **user-facing text** (CLI, GUI, web responses):

* **Source of Truth**

  * Store user-facing strings in a **separate resource file** (e.g., JSON, YAML, CSV), not hard-coded in scripts.
  * Reference strings via keys or constants instead of literals.

* **Keys vs. Values**

  * **Keys (IDs):** stable, English, `snake_case` or `dot.separated`.
  * **Values (Displayed Text):** natural language in the target locale.
  * Never mix code-style identifiers with user-facing text.

* **Placeholders**

  * Use named placeholders (`{value}`, `{level}`), not positional (`%s`).
  * Preserve formatting tags (e.g., `<b>`, `<color>`) if used in UI.

---

## Agent Behavior

* Always use complete, meaningful English names for **code** identifiers.
* For **user-facing text**, follow natural writing conventions, not code style.
* Write code with **logic that reads clearly and directly** — no clever hacks.
* When multiple solutions are possible, explain trade-offs and recommend the most maintainable one.
* When unsure, ask for clarification instead of making assumptions.
* Always align with this repository’s existing patterns before introducing new ones.
