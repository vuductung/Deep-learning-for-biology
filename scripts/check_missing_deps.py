#!/usr/bin/env python3
"""Check for third-party imports missing from pyproject dependencies."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import tomlkit

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_PACKAGE = "dl_biology"


MODULE_TO_PACKAGE = {
    "bio": "biopython",
    "sklearn": "scikit-learn",
}

PACKAGE_SUGGESTIONS = {
    "biopython": "biopython>=1.83",
    "deeplc": "deeplc>=2.4",
    "h5py": "h5py>=3.9",
    "jax": "jax>=0.4",
    "matplotlib": "matplotlib>=3.8",
    "numpy": "numpy>=2.0",
    "pandas": "pandas>=2.0",
    "psm_utils": "psm_utils>=0.3",
    "scikit-learn": "scikit-learn>=1.0",
    "scipy": "scipy>=1.10",
    "seaborn": "seaborn>=0.13",
    "torch": "torch>=2.0",
    "tqdm": "tqdm>=4.66",
    "transformers": "transformers>=4.40",
}

EXTRA_STDLIB_MODULES = {
    "builtins",
    "dataclasses",
    "typing_extensions",  # available in some stdlib distributions
}


def normalize_dep(dep: str) -> str | None:
    dep = dep.strip()
    if not dep:
        return None
    for delimiter in ("[", " ", "=", "<", ">", "!", "~"):
        idx = dep.find(delimiter)
        if idx != -1:
            dep = dep[:idx]
    return dep.lower() or None


def load_declared_packages() -> set[str]:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    project = data.get("project", {})
    deps = list(project.get("dependencies", []))

    optional = project.get("optional-dependencies", {})
    for extra_deps in optional.values():
        deps.extend(extra_deps)

    declared: set[str] = set()
    for dep in deps:
        normalized = normalize_dep(dep)
        if normalized:
            declared.add(normalized)
    return declared


def gather_python_files() -> list[Path]:
    ignore_dirs = {
        ".git",
        ".github",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "__pycache__",
        "build",
        "dist",
        "env",
        "venv",
        ".venv",
        "data",
        "kaggle-submission",
        "pdf",
    }
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in ignore_dirs for part in path.parts):
            continue
        files.append(path)
    return files


def is_stdlib(module: str) -> bool:
    return module in sys.stdlib_module_names or module in EXTRA_STDLIB_MODULES


def determine_missing_packages(declared_packages: set[str]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for py_file in gather_python_files():
        with py_file.open("r", encoding="utf-8") as handle:
            try:
                tree = ast.parse(handle.read(), filename=str(py_file))
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    check_module(module, declared_packages, missing)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.module is None:
                    continue
                if node.level:
                    continue
                if not node.module:
                    continue
                module = node.module.split(".")[0]
                check_module(module, declared_packages, missing)
    return missing


def check_module(module: str, declared_packages: set[str], missing: dict[str, str]) -> None:
    module_lower = module.lower()
    if module_lower == PROJECT_PACKAGE:
        return
    if module_lower.startswith(f"{PROJECT_PACKAGE}."):
        return

    if is_stdlib(module_lower):
        return

    package = MODULE_TO_PACKAGE.get(module_lower, module_lower)
    if package in declared_packages:
        return

    missing[module] = package


def main() -> int:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    declared_packages = load_declared_packages()

    missing = determine_missing_packages(declared_packages)
    if not missing:
        return 0

    new_packages = sorted(set(missing.values()) - declared_packages)
    if not new_packages:
        return 1

    document = tomlkit.parse(pyproject_text)
    project_table = document.setdefault("project", tomlkit.table())

    deps_array = project_table.get("dependencies")
    if deps_array is None:
        deps_array = tomlkit.array()
        deps_array.multiline(True)
        project_table["dependencies"] = deps_array
    elif isinstance(deps_array, list):
        new_array = tomlkit.array()
        new_array.multiline(True)
        for value in deps_array:
            new_array.append(value)
        deps_array = new_array
        project_table["dependencies"] = deps_array
    else:
        deps_array.multiline(True)

    existing_dep_names = {
        normalize_dep(item.value if hasattr(item, "value") else str(item))
        for item in deps_array  # type: ignore[arg-type]
    }
    existing_dep_names.discard(None)

    for package in new_packages:
        if package in existing_dep_names:
            continue
        dependency_str = PACKAGE_SUGGESTIONS.get(package, package)
        deps_array.append(tomlkit.string(dependency_str))

    pyproject_path.write_text(tomlkit.dumps(document), encoding="utf-8")

    print(
        "Added missing packages to pyproject.toml: " + ", ".join(new_packages),
        file=sys.stderr,
    )
    print(
        "Please review the dependency constraints and re-run pre-commit.",
        file=sys.stderr,
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
