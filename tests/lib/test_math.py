#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test for lib."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

import scatterkit.lib.math
import scatterkit.lib.util

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_GRO_NPT, WATER_TPR_NPT  # noqa: E402


class Test_sfactor:
    """Tests for the sfactor."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms

    @pytest.fixture
    def qS(self):
        """Define q and S."""
        q = np.array(
            [
                0.25,
                0.25,
                0.25,
                0.36,
                0.36,
                0.36,
                0.44,
                0.51,
                0.51,
                0.51,
                0.56,
                0.56,
                0.56,
                0.56,
                0.56,
                0.56,
                0.62,
                0.62,
                0.62,
                0.71,
                0.71,
                0.71,
                0.76,
                0.76,
                0.76,
                0.76,
                0.76,
                0.76,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.84,
                0.84,
                0.84,
                0.88,
                0.91,
                0.91,
                0.91,
                0.91,
                0.91,
                0.91,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
            ]
        )

        S = np.array(
            [
                1.86430e02,
                6.91000e00,
                8.35300e02,
                5.06760e02,
                1.92540e02,
                1.57790e02,
                9.96500e01,
                5.87470e02,
                7.88630e02,
                5.18170e02,
                4.58650e02,
                1.69000e00,
                3.99910e02,
                6.10340e02,
                1.21359e03,
                4.11800e01,
                9.31980e02,
                6.29120e02,
                9.88500e01,
                3.15220e02,
                1.00840e02,
                1.19420e02,
                2.13180e02,
                4.61770e02,
                3.99640e02,
                8.03880e02,
                1.74830e02,
                3.20900e01,
                1.99190e02,
                4.24690e02,
                1.73552e03,
                1.37732e03,
                1.25050e02,
                2.61750e02,
                4.29610e02,
                2.09000e01,
                2.71450e02,
                4.22340e02,
                1.07590e02,
                3.79520e02,
                6.69000e00,
                5.35330e02,
                1.09210e02,
                6.69970e02,
                1.25354e03,
                3.94200e02,
                1.96100e02,
                1.39890e02,
                8.79600e01,
                4.17020e02,
            ]
        )

        return q, S

    @pytest.mark.parametrize("qmin", [0, 0.05])
    @pytest.mark.parametrize("qmax", [0.075, 0.1])
    def test_sfactor(self, ag, qS, qmin, qmax):
        """Test sfactor."""
        q, S = scatterkit.lib.math.compute_structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            qmin,
            qmax,
            0,  # mimtheta
            np.pi,  # thetamax
            np.ones(len(ag.positions)),
        )

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        # Get indices to slice qS array
        sel_indices = np.logical_and(qmin < qS[0], qS[0] < qmax)

        assert_allclose(q, qS[0][sel_indices], rtol=1e-2)

        # Only check S for full q width
        if qmin == 0 and qmax == 1:
            assert_allclose(S, qS[1], rtol=1e-2)

    def test_sfactor_angle(self, ag):
        """Test sfactor angle."""
        q, S = scatterkit.lib.math.compute_structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            0,  # qmin
            0.5,  # qmax
            np.pi / 4,  # thetamin
            np.pi / 2,  # thetamax
            np.ones(len(ag.positions)),
        )

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        assert_allclose(q, np.array([0.25, 0.25, 0.36, 0.36, 0.36, 0.44]), rtol=1e-1)
        assert_allclose(
            S, np.array([6.91, 835.3, 192.54, 157.79, 506.76, 99.65]), rtol=1e-1
        )


def test_form_factor():
    """Regression test for the atomic form factor as function q.

    Reference values for hydrogen are taken from Table 6.1.1.1 in
    https://it.iucr.org/Cb/ch6o1v0001/
    """
    reference_values = np.array(
        [
            [0.00, 1.000],
            [0.01, 0.998],
            [0.02, 0.991],
            [0.03, 0.980],
            [0.04, 0.966],
            [0.05, 0.947],
            [0.06, 0.925],
            [0.07, 0.900],
            [0.08, 0.872],
            [0.09, 0.842],
            [0.10, 0.811],
            [0.22, 0.424],
            [0.46, 0.090],
        ]
    )

    sin_theta = reference_values[:, 0]
    q = 4 * np.pi * sin_theta
    desired = reference_values[:, 1]

    assert_allclose(
        actual=scatterkit.lib.math.atomic_form_factor(q, "H"),
        desired=desired,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    ("atom_type", "n_electrons"),
    [
        ("C", 6),
        ("Cval", 6),
        ("CVAL", 6),  # upper case elements should also work
        ("O", 8),
        ("CH1", 7),
        ("CH2", 8),
        ("CH3", 9),
        ("CH4", 10),
        ("NH1", 8),
        ("NH2", 9),
        ("NH3", 10),
    ],
)
def test_form_factor_zero(atom_type, n_electrons):
    """Test that the atomic form factor for q=0 is same as the number of electrons."""
    assert_allclose(
        actual=scatterkit.lib.math.atomic_form_factor(0.0, atom_type),
        desired=n_electrons,
        rtol=1e-3,
    )


def test_form_factor_unknown_element():
    """Test that an unknown elements raise an error."""
    match = (
        "Element 'foo' not found. Known elements are listed in the "
        "`scatterkit.lib.tables.elements` set."
    )
    with pytest.raises(ValueError, match=match):
        scatterkit.lib.math.atomic_form_factor(0.0, "foo")


def test_rdf_structure_factor_unequal_spacing():
    """Test that a ValueError is raised if the input is not equally spaced.

    Additional tests for the functionality are located in `test_saxs.py` and
    `test_diorderstructurefactor.py`.
    """
    r = np.linspace(0, 10, 10)
    r[0] += 1e-5

    rdf = np.ones(len(r))
    density = 1.0
    match = "Distance array `r` is not equally spaced!"
    with pytest.raises(ValueError, match=match):
        scatterkit.lib.math.compute_rdf_structure_factor(rdf=rdf, r=r, density=density)
