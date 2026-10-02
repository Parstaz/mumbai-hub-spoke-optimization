"""Static import-closure walking, shared by the two tests that forbid a dependency.

Two of CLAUDE.md §1.1's rules are about what a module is *allowed to import*, and neither has a
behavioural symptom:

* ``src/baseline`` must not reach the optimized stages, or tuning the treatment would move the
  control's column with it (``tests/test_assignment.py``).
* the pipeline must not reach the OR-Tools reference, or the quality benchmark would become a
  component of the thing it benchmarks (``tests/test_closure.py``).

Both are the same walk over the same graph, so the walk lives here once. It reads the import
statements with :mod:`ast` rather than importing anything, because importing a module to inspect
its imports would make the test pass or fail on import side effects.
"""

from __future__ import annotations

import ast
import pathlib

SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"


def imports_of(path: pathlib.Path) -> set[str]:
    """Every module name ``path`` imports, as written."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def module_path(module: str) -> pathlib.Path | None:
    """Resolve a ``src.`` module name to its file, or ``None`` if it is not ours."""
    if not module.startswith("src."):
        return None
    relative = pathlib.Path(*module.split(".")[1:])
    for candidate in (SRC_ROOT / f"{relative}.py", SRC_ROOT / relative / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def reachable_modules(roots: tuple[pathlib.Path, ...]) -> set[str]:
    """Every ``src.`` module reachable from ``roots`` by following imports transitively.

    Walked as a closure rather than checked one file deep because the coupling these tests forbid
    would arrive indirectly — someone adds an import to a shared helper, not to the file where it
    would be obvious.

    Args:
        roots: Files to start from.

    Returns:
        The names of every ``src.`` module reachable from any root, the roots' own imports
        included.
    """
    frontier = list(roots)
    seen: set[pathlib.Path] = set()
    reached: set[str] = set()
    while frontier:
        path = frontier.pop()
        if path in seen:
            continue
        seen.add(path)
        for module in imports_of(path):
            if module.startswith("src."):
                reached.add(module)
            resolved = module_path(module)
            if resolved is not None:
                frontier.append(resolved)
    return reached
