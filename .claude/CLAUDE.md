# Session Rules

- Do NOT guess next steps or proceed with unrequested implementation.
- Before making code modifications, outline a step-by-step plan for approval. Plans are bullet lists, that include: what changes, which file, why. Keep them short and state explicitly what you are NOT touching.
- A plan with more than ~5 steps means the change is too big. Propose a smaller first increment instead.
- Do NOT modify core architecture or delete existing functional flows without explicit permission.
- Approval of a plan approves that plan, not adjacent work you thought of while implementing. If you discover mid-implementation that the plan was wrong, stop and say so.

# Scope Discipline

- Change the minimum number of lines that solves the stated problem. One logical change per turn.
- Hard limits per change unless I waive them: ~200 changed lines, 5 files. If exceeded, stop and propose a split into sequential reviewable steps.
- Do not refactor, rename, reformat, or clean up code you happened to open. If nearby code looks wrong, mention it in one line — don't fix it, unless I approve.
- Do not add abstractions, config options, error handling, logging, or tests that weren't requested. Don't build for hypothetical future requirements.
- Do not add or upgrade dependencies without asking.
- Prefer the reversible, boring option over the clever one.

# Communication

- Default to the shortest correct answer. No preamble, no summary of what you just did, no restating code in prose — the diff is the explanation.
- Keep your notes on "next steps", "future improvements", or "things to consider" short.
- Explanations only when asked or when a decision is non-obvious — then one or two sentences.
- After a change, give one line on how to verify it. If you could not verify, say so plainly. Never imply something works when you haven't checked.

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
- PR body should be plain, concise prose. No section headers, checklists, or structured templates. Describe the problem, what the change does, and any non-obvious tradeoffs. A good PR description reads like a short paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 88 columns. Let each sentence flow on one line.

# Code Guidelines

- Line length limit is 120 columns. This applies to code, comments, and docstrings.
- When fixing pre-commit errors, prioritize fixing the root cause over adding suppression comments (like # noqa, # pyright: ignore, etc.)
- Always use specific type hints instead of `typing.Any`.
- Be concise. When writing documentation, docstrings, comments, new functions, classes, or entrypoints, write only what is needed to convey intent and usage. Avoid comments that restate the code, redundant or boilerplate docstrings, over-explaining obvious behavior, and filler prose. Match the verbosity and comment density of the surrounding code.

## Imports & package structure

- Prefer module-level imports. Use local (function-scoped) imports only to break a genuine circular dependency or to defer a heavy/optional import — and add a brief comment saying which. Don't use them to paper over import-order problems; restructure instead.
- Modules inside a package must never import their own enclosing package by name. For example, a file in `utils/` must not write `from characterization import utils` or `from characterization.utils import X` — always import the specific submodule directly (e.g. `from characterization.utils.logging_utils import get_pylogger`). The same rule applies to any other package. (This intra-package rule is the opposite of how you treat *other* packages, which you import via their public API — see below.)
- Import another package through its public top-level API, not its internal submodules. If you need one of its internals, that's a signal its API is missing something — flag it rather than reaching in.
- `__init__.py` defines a package's PUBLIC API. Only put there: re-exports of names meant for external use, plus `__all__` listing them. Treat it as the package's contract — don't add a name without confirming it's meant to be public.
- Every re-exporting `__init__.py` must define `__all__`, and every name in `__all__` must be imported or defined in that file.
- No heavy work at import time. No slow imports, network/file/db access, config loading, or logging side effects in `__init__.py` — it runs on every import of the package. Defer expensive submodules with module-level `__getattr__` (PEP 562) instead of importing them eagerly.
- Never use `from module import *`. Import explicit names.
- Use explicit relative imports within a package (`from .core import X`); use absolute imports across packages.

## Reuse & utilities

- Don't duplicate logic: before writing new code, check if similar logic already exists, especially in `src/characterization/utils/`. Reuse it even if it means importing across modules.
- Extract shared logic into `src/characterization/utils/` if you encounter duplicated code.

## Tests

- Use functions and fixtures; do not use test classes.
- Favor targeted, efficient tests over exhaustive edge-case coverage.
- Prefer running individual tests rather than the full test suite to improve iteration speed.
