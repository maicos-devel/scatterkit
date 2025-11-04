#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""

import sys
from pathlib import Path

import pytest

import scatterkit.lib.util

sys.path.append(str(Path(__file__).parents[1]))


def test_banner():
    """Test banner string by checking some necesarry features.

    The banner is not tested for exact string equality. We just check the necessary
    features. Everything else is up to the developers to get creative.
    """
    # Test the character replacement
    assert scatterkit.lib.util.scatterkit_banner(frame_char="%")[1] == "%"
    # Check for correct number of lines as a sanity check
    assert scatterkit.lib.util.scatterkit_banner().count("\n") == 4
    # Check that newlines are added top and bottom
    assert scatterkit.lib.util.scatterkit_banner().startswith("\n")
    assert scatterkit.lib.util.scatterkit_banner().endswith("\n")
    # Check for correct length of lines (80 characters excluding top and bottom)
    # Also add in a long version string to check that it doesn't overflow
    for line in scatterkit.lib.util.scatterkit_banner(version="v1.10.11").split("\n")[
        1:-1
    ]:
        assert len(line) == 80
    # Check that the version is correctly inserted
    assert "v0.0.1" in scatterkit.lib.util.scatterkit_banner(version="v0.0.1")


@pytest.mark.parametrize(
    ("doc", "new_doc"),
    [
        ("${TEST}", "test"),
        (None, None),
        ("", ""),
        ("foo", "foo"),
        ("${TEST} ${BLA}", "test blu"),
        ("${OUTER}", "desc with inner"),
    ],
)
def test_render_docs(doc, new_doc):
    """Test decorator for replacing patterns in docstrings."""

    def func():
        pass

    DOC_DICT = dict(
        TEST="test",
        BLA="blu",
        INNER="inner",
        OUTER="desc with ${INNER}",
    )

    func.__doc__ = doc
    func_decorated = scatterkit.lib.util._render_docs(func, doc_dict=DOC_DICT)
    assert func_decorated.__doc__ == new_doc
