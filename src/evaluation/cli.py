"""Evaluation CLI entrypoint.

This module keeps ``knee-oa-evaluate`` stable by forwarding to the
``evaluate`` command implemented in ``src.training.cli``.
"""

from __future__ import annotations

from src.training.cli import evaluate as main


if __name__ == "__main__":
    main()
