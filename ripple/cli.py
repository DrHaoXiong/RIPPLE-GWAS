"""RIPPLE V1 private CLI skeleton."""

from __future__ import annotations

import typer

app = typer.Typer(help="Private RIPPLE-GWAS V1 prototype CLI.")


@app.command()
def run(config: str) -> None:
    """Run the full RIPPLE pipeline from a configuration file."""

    raise NotImplementedError("Full pipeline execution is not implemented yet.")


@app.command()
def diagnose(config: str) -> None:
    """Run configuration and environment diagnostics."""

    raise NotImplementedError("Diagnostics are not implemented yet.")


if __name__ == "__main__":
    app()
