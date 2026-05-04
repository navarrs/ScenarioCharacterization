# Development Workflow

- **Always use `uv run`, not python**. For example:
```sh
# Run
uv run python ...

# Format and lint before committing
uv run ruff format
uv run ruff check --fix
uv run pre-commit run --all-files
```

- Always run `uv run pre-commit run --all-files` before committing and creating a PR. This runs formatting, linting, and type checking. Do not commit code that fails type checking.

- When making user-facing changes, review the documentation inside `docs` and ensure it is up to date.

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in the title.
- PR body should be plain, concise prose. No section headers, checklists, or structured templates. Describe the problem, what the change does, and any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 88 columns. Let each sentence flow on one line.

# Code Guidelines

- Line length limit is 120 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- When fixing pre-commit errors, prioritize fixing the root cause over adding suppression comments (like # noqa, # pyright: ignore, etc.)
- Always use specific type hints instead of `typing.Any`.
- Don't duplicate logic: Before writing new code, check if similar logic already exists in the codebase. Reuse existing functions, especially those in `src/safeair/utils/`, even if it means importing across modules.
- Extract shared logic into `src/safeair/utils/` if you encounter duplicated code.
- Before creating any new utilities, search `src/safeair/utils/` for existing library code.
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.
