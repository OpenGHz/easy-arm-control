#!/usr/bin/env python3
"""Check that two git revisions contain behaviourally identical Python code.

Compares the abstract syntax trees (ASTs) of every ``.py`` file and of every
code cell in ``.ipynb`` notebooks between two revisions. Comments, blank lines,
quoting style, line wrapping and other layout changes do not appear in the AST,
so a pure formatting change passes while any change to logic is reported.

Two normalisations are applied to both sides before comparing:

* Docstrings are compared after ``inspect.cleandoc(...).strip()``, because
  formatters re-indent docstrings and trim their surrounding whitespace. This
  is only safe while no code reads ``__doc__``; the check fails if it does.
* ``--rename OLD=NEW`` maps the module/package name ``OLD`` to ``NEW`` in
  import statements, bare names and file paths of the *first* revision, so a
  pure rename passes as well.

Usage::

    python tools/check_ast_equivalence.py REV_A REV_B [--rename OLD=NEW ...]

Exit status is 0 when every file present in both revisions is equivalent and
no file was removed; files only present in ``REV_B`` are listed as added.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import subprocess
import sys

DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def list_sources(rev: str) -> list[str]:
    files = git("ls-tree", "-r", "--name-only", rev).splitlines()
    return [f for f in files if f.endswith((".py", ".ipynb"))]


def read_sources(rev: str, path: str) -> list[str]:
    """Return the Python sources of a file: one for .py, one per code cell."""
    raw = git("show", f"{rev}:{path}")
    if path.endswith(".py"):
        # Python reads source files with universal newlines.
        return [raw.replace("\r\n", "\n").replace("\r", "\n")]
    notebook = json.loads(raw)
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        # IPython magics / shell escapes are not Python syntax.
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        cells.append("\n".join(lines))
    return cells


class Rename(ast.NodeTransformer):
    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def _module(self, name: str | None) -> str | None:
        if name is None:
            return None
        head, sep, tail = name.partition(".")
        return self.renames.get(head, head) + sep + tail

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            alias.name = self._module(alias.name)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.level == 0:
            node.module = self._module(node.module)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.renames.get(node.id, node.id)
        return node


def normalise(source: str, renames: dict[str, str] | None) -> str:
    tree = ast.parse(source)
    if renames:
        tree = Rename(renames).visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, DOC_NODES) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                first.value.value = inspect.cleandoc(first.value.value).strip()
    return ast.dump(tree)


def reads_docstrings(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__doc__":
            return True
        if isinstance(node, ast.Name) and node.id == "__doc__":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "getdoc":
            return True
    return False


def map_path(path: str, renames: dict[str, str]) -> str:
    return "/".join(renames.get(part, part) for part in path.split("/"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that two git revisions contain identical Python code."
    )
    parser.add_argument("rev_a")
    parser.add_argument("rev_b")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="module/package rename applied to REV_A (repeatable)",
    )
    args = parser.parse_args()
    renames = dict(item.split("=", 1) for item in args.rename)

    files_a = {map_path(p, renames): p for p in list_sources(args.rev_a)}
    files_b = set(list_sources(args.rev_b))
    failures = 0

    for new_path in sorted(files_a.keys() | files_b):
        old_path = files_a.get(new_path)
        if old_path is None:
            print(f"ADDED    {new_path}")
            continue
        if new_path not in files_b:
            print(f"REMOVED  {old_path}")
            failures += 1
            continue
        sources_a = read_sources(args.rev_a, old_path)
        sources_b = read_sources(args.rev_b, new_path)
        same = len(sources_a) == len(sources_b) and all(
            normalise(a, renames) == normalise(b, None)
            for a, b in zip(sources_a, sources_b)
        )
        if same and any(reads_docstrings(b) for b in sources_b):
            print(f"DOCREAD  {new_path} (reads __doc__; docstring check unsafe)")
            failures += 1
            continue
        label = new_path if old_path == new_path else f"{old_path} -> {new_path}"
        print(f"{'OK      ' if same else 'DIFF    '} {label}")
        failures += not same

    print("RESULT:", "EQUIVALENT" if not failures else f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
