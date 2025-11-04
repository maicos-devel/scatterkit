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
