#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyse scattering in molecular dynamics simulation."""

from maicos.core import AnalysisBase
from mdacli import cli

from scatterkit import __version__


def main():
    """Execute main CLI entry point."""
    cli(
        name="scatterkit",
        module_list=["scatterkit"],
        base_class=AnalysisBase,
        version=__version__,
        description=__doc__,
        ignore_warnings=True,
    )


if __name__ == "__main__":
    main()
