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

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import scatterkit.lib.weights
from maicos.lib.util import unit_vectors_planar

sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP # noqa: E402
from util import line_of_water_molecules  # noqa: E402

@pytest.fixture
def ag_spce():
    """Atomgroup containing a single water molecule poiting in z-direction."""
    return mda.Universe(SPCE_ITP, SPCE_GRO).atoms

def test_diporder_pair_weights_single(ag_spce):
    """Test that the weight of the same molecules is equal to one 1."""
    weights = scatterkit.lib.weights.diporder_pair_weights(
        ag_spce, ag_spce, compound="residues"
    )
    assert_allclose(weights, 1)

def test_diporder_pair_weights_line():
    """Test that the weight of the same molecules is equal to one 1."""
    ag = line_of_water_molecules(n_molecules=4, angle_deg=[0.0, 45.0, 90.0, 180.0])
    weights = scatterkit.lib.weights.diporder_pair_weights(ag, ag, compound="residues")

    weights_expected = np.array(
        [
            [1.00, 0.71, 0.00, -1.00],
            [0.71, 1.00, 0.71, -0.71],
            [0.00, 0.71, 1.00, 0.00],
            [-1.00, -0.71, 0.00, 1.00],
        ]
    )
    assert_equal(weights.round(2), weights_expected)
