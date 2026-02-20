"""Experiments CLI that proxies to training utilities."""

from __future__ import annotations

import click

from src.training.cli import evaluate as eval_command
from src.training.cli import train as train_command


@click.group()
def main() -> None:
    """High-level experiment runner."""


@main.command(name="train")
@click.pass_context
def train(ctx):
    ctx.forward(train_command)


@main.command(name="evaluate")
@click.pass_context
def evaluate(ctx):
    ctx.forward(eval_command)


if __name__ == "__main__":
    main()
