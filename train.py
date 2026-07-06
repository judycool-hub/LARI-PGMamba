#!/usr/bin/env python
"""Compatibility wrapper for the LARI-Mamba training entry point."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().with_name("main_DsDTW_length_adaptive.py")
    runpy.run_path(str(script), run_name="__main__")
