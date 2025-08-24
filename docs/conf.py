"""Sphinx configuration for the PSD project."""

from __future__ import annotations

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

project = "PSD"
author = "PSD Authors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Enable Markdown features in MyST
myst_enable_extensions = ["colon_fence"]

# Avoid importing heavy dependencies during doc builds
autodoc_mock_imports = ["numpy", "torch", "torchvision"]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"

