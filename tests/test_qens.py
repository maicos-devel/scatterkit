#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the QENS module.

Multi-tau correlator behaviour is covered by MAiCoS's own test suite
(``tests/lib/test_correlator.py``); this file exercises only the scatterkit
``Qens`` wrapper.
"""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

sys.path.append(str(Path(__file__).parents[1]))

from scatterkit import Qens


class TestQens:
    """Tests for the Qens analysis class using trajectory data."""

    @pytest.fixture
    def ag_water_npt(self):
        """Multi-frame water trajectory."""
        from data import WATER_TPR_NPT, WATER_TRR_NPT

        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.select_atoms("name OW")

    @pytest.fixture
    def ag_water_2f(self):
        """Two-frame water trajectory."""
        from data import WATER_2F_TRR_NPT, WATER_TPR_NPT

        u = mda.Universe(WATER_TPR_NPT, WATER_2F_TRR_NPT)
        return u.select_atoms("name OW")

    def test_smoke(self, ag_water_npt, monkeypatch, tmp_path):
        """Smoke test: Qens runs without errors and produces output."""
        monkeypatch.chdir(tmp_path)
        qens = Qens(
            ag_water_npt,
            qmin=0.5,
            qmax=2.0,
            dq=0.5,
            correlator_num_levels=4,
            correlator_channels_per_level=8,
            output="fs.dat",
        ).run()

        assert hasattr(qens.results, "lag_times")
        assert hasattr(qens.results, "q_values")
        assert hasattr(qens.results, "F_s")
        assert len(qens.results.lag_times) > 0
        assert len(qens.results.q_values) > 0
        assert qens.results.F_s.shape[0] == len(qens.results.q_values)
        assert qens.results.F_s.shape[1] == len(qens.results.lag_times)

    def test_save(self, ag_water_npt, monkeypatch, tmp_path):
        """Test that save produces a loadable file."""
        monkeypatch.chdir(tmp_path)
        qens = Qens(
            ag_water_npt,
            qmin=0.5,
            qmax=2.0,
            dq=0.5,
            correlator_num_levels=4,
            correlator_channels_per_level=8,
            output="fs",
        ).run()
        qens.save()

        data = np.loadtxt("fs.dat")
        assert data.shape[0] == len(qens.results.lag_times)

    def test_f_s_at_t0_near_one(self, ag_water_npt):
        """F_s(q, t=0) should be close to 1 (self-correlation at zero lag)."""
        qens = Qens(
            ag_water_npt,
            qmin=0.5,
            qmax=1.5,
            dq=0.5,
            correlator_num_levels=4,
            correlator_channels_per_level=8,
        ).run()

        # The smallest lag is dt (not exactly 0), so F_s won't be exactly 1,
        # but for small q it should be close. Use a generous tolerance since
        # the trajectory is short and the smallest lag may already show decay.
        assert_allclose(qens.results.F_s[:, 0], 1.0, atol=0.35)

    def test_deterministic_reproducibility(self, ag_water_npt):
        """Same parameters should give identical results (deterministic q-vectors)."""
        kwargs = dict(
            qmin=0.5,
            qmax=1.5,
            dq=0.5,
            correlator_num_levels=4,
            correlator_channels_per_level=8,
        )
        qens1 = Qens(ag_water_npt, **kwargs).run()
        qens2 = Qens(ag_water_npt, **kwargs).run()

        assert_allclose(qens1.results.F_s, qens2.results.F_s)
        assert_allclose(qens1.results.q_values, qens2.results.q_values)
