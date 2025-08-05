#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""

import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.core.util import UnWrapUniverse
from numpy.testing import assert_allclose, assert_equal

import scatterkit.lib.util
from maicos.core.base import AnalysisBase

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_GRO_NPT, WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


@pytest.mark.parametrize(
    ("u", "compound"),
    [
        (UnWrapUniverse(), "molecules"),
        (
            UnWrapUniverse(have_molnums=False, have_bonds=True),
            "fragments",
        ),
        (
            UnWrapUniverse(have_molnums=False, have_bonds=False),
            "residues",
        ),
    ],
)
def test_get_compound(u, compound):
    """Tests check compound."""
    comp = scatterkit.lib.util.get_compound(u.atoms)
    assert compound == comp


def test_get_cli_input():
    """Tests get cli input."""
    testargs = ["scatterkit", "foo", "foo bar"]
    with patch.object(sys, "argv", testargs):
        assert scatterkit.lib.util.get_cli_input() == 'scatterkit foo "foo bar"'


def test_banner():
    """Test banner string by checking some necesarry features.

    The banner is not tested for exact string equality. We just check the necessary
    features. Everything else is up to the developers to get creative.
    """
    # Test the character replacement
    assert scatterkit.lib.util.maicos_banner(frame_char="%")[1] == "%"
    # Check for correct number of lines as a sanity check
    assert scatterkit.lib.util.maicos_banner().count("\n") == 10
    # Check that newlines are added top and bottom
    assert scatterkit.lib.util.maicos_banner().startswith("\n")
    assert scatterkit.lib.util.maicos_banner().endswith("\n")
    # Check for correct length of lines (80 characters excluding top and bottom)
    # Also add in a long version string to check that it doesn't overflow
    for line in scatterkit.lib.util.maicos_banner(version="v1.10.11").split("\n")[1:-1]:
        assert len(line) == 80
    # Check that the version is correctly inserted
    assert "v0.0.1" in scatterkit.lib.util.maicos_banner(version="v0.0.1")


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


def single_class(atomgroup, filter):
    """Single class."""

    @scatterkit.lib.util.charge_neutral(filter)
    class SingleCharged(AnalysisBase):
        def __init__(self, atomgroup):
            self.atomgroup = atomgroup
            self.filter = filter

        def _prepare(self):
            def inner_func(self):
                pass

            inner_func(self)

    return SingleCharged(atomgroup)


def multi_class(atomgroup, filter):
    """Multi class."""

    @scatterkit.lib.util.charge_neutral(filter)
    class MultiCharged(AnalysisBase):
        def __init__(self, atomgroup):
            self.atomgroup = atomgroup
            self.filter = filter

        def _prepare(self):
            def inner_func(self):
                pass

            inner_func(self)

    return MultiCharged(atomgroup)


class ModuleInput(AnalysisBase):
    """Class creating an output file to check the module input reporting."""

    def _single_frame(self):
        # Do nothing, but the run() methods needs to be called
        pass

    def __init__(self, atomgroup, test_input="some_default", refgroup=None):
        self._locals = locals()
        super().__init__(
            atomgroup=atomgroup,
            unwrap=False,
            pack=True,
            refgroup=refgroup,
            jitter=0.0,
            wrap_compound="atoms",
            concfreq=0,
        )


@pytest.fixture
def ag():
    """Import MDA universe."""
    u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
    return u.atoms


def test_get_module_input_str(ag):
    """Test to get the module input."""
    # Test if the module name is written correctly
    ana = ModuleInput(ag)
    ana.run()
    module_input = scatterkit.lib.util.get_module_input_str(ana)

    assert "ModuleInput(" in module_input

    # Test if the refgroup name is written correctly
    ana = ModuleInput(ag, refgroup=ag)
    ana.run()
    module_input = scatterkit.lib.util.get_module_input_str(ana)

    assert "refgroup=<AtomGroup>" in module_input
    assert "atomgroup=<AtomGroup>" in module_input

    # Test if the default value of the test_input parameter is written
    ana = ModuleInput(ag)
    ana.run()
    module_input = scatterkit.lib.util.get_module_input_str(ana)

    assert "test_input='some_default'" in module_input

    assert "refgroup=None" in module_input

    assert (
        ".run(start=None, stop=None, step=None, frames=None, verbose=None, "
        "progressbar_kwargs=None)" in module_input
    )

    # Test if the set test_input parameter is written correctly
    ana = ModuleInput(ag, test_input="some_other_value")
    ana.run()
    module_input = scatterkit.lib.util.get_module_input_str(ana)

    assert "test_input='some_other_value'" in module_input

    ana.run(step=2, stop=7, start=5, verbose=True)
    module_input = scatterkit.lib.util.get_module_input_str(ana)
    assert (
        ".run(start=5, stop=7, step=2, frames=None, verbose=True, "
        "progressbar_kwargs=None)" in module_input
    )


class TestChargedDecorator:
    """Test charged decorator."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms

    def test_charged_single(self, ag):
        """Test charged single."""
        with pytest.raises(UserWarning, match="At least one AtomGroup has free"):
            single_class(ag.select_atoms("name OW*"), filter="error")._prepare()

    def test_charged_single_warn(self, ag):
        """Test charged single warn."""
        with pytest.warns(UserWarning, match="At least one AtomGroup has free"):
            single_class(ag.select_atoms("name OW*"), filter="default")._prepare()

    def test_universe_charged_single(self, ag):
        """Test universe charged single."""
        ag[0].charge += 1
        with pytest.raises(UserWarning, match="At least one AtomGroup has free"):
            single_class(ag.select_atoms("name OW*"), filter="error")._prepare()

    def test_universe_slightly_charged_single(self, ag):
        """Test universe slightly charged single."""
        ag[0].charge += 1e-5
        single_class(ag, filter="error")._prepare()


def unwrap_refgroup_class(**kwargs):
    """Simple class setting keyword arguments as attributes."""

    @scatterkit.lib.util.unwrap_refgroup
    class UnwrapRefgroup(AnalysisBase):
        def __init__(self, **kwargs):
            for key, val in kwargs.items():
                setattr(self, key, val)

        def _prepare(self):
            def inner_func(self):
                pass

            inner_func(self)

    return UnwrapRefgroup(**kwargs)


class TestWrapRefgroup:
    """Test the `unwrap_refgroup` decorator."""

    def test_unwrap_refgroup(self):
        """Test to raise an error if unwrap and refgroup."""
        with pytest.raises(ValueError, match="`unwrap=False` and `refgroup"):
            unwrap_refgroup_class(unwrap=False, refgroup="foo")._prepare()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"unwrap": True, "refgroup": None},
            {"unwrap": False, "refgroup": None},
            {"unwrap": True, "refgroup": "foo"},
        ],
    )
    def test_noerror(self, kwargs):
        """Decorator should raise an error otherwise."""
        unwrap_refgroup_class(**kwargs)._prepare()


class TestTrajectoryPrecision:
    """Test the detection of the trajectory precision."""

    @pytest.fixture
    def trj(self):
        """Import MDA universe trajectory."""
        return mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT).trajectory

    def test_gro_trajectory(self, trj):
        """Test detect gro traj."""
        assert_equal(scatterkit.lib.util.trajectory_precision(trj), np.float32(0.01))


class TestCitationReminder:
    """Test the detection of the trajectory precision."""

    def test_single_citation(self):
        """Test if a single citation will get printed correctly."""
        doi = "10.1103/PhysRevE.92.032718"

        assert doi in scatterkit.lib.util.citation_reminder(doi)
        assert "please read" in scatterkit.lib.util.citation_reminder(doi)
        assert "Schaaf" in scatterkit.lib.util.citation_reminder(doi)

    def test_mutliple_citation(self):
        """Test if a two citations will get printed at the same time."""
        dois = ["10.1103/PhysRevE.92.032718", "10.1103/PhysRevLett.117.048001"]

        assert "Schlaich" in scatterkit.lib.util.citation_reminder(*dois)
        assert "Schaaf" in scatterkit.lib.util.citation_reminder(*dois)
        assert dois[0] in scatterkit.lib.util.citation_reminder(*dois)
        assert dois[1] in scatterkit.lib.util.citation_reminder(*dois)


class TestCorrelationAnalysis:
    """Test the calculation of the correlation of the data."""

    def test_short_data(self):
        """Test if a warning is issued if the data is too short."""
        warning = "Your trajectory is too short to estimate a correlation "
        with pytest.warns(match=warning):
            corrtime = scatterkit.lib.util.correlation_analysis(np.arange(4))
        assert corrtime == -1

    def test_insufficient_data(self, mocker):
        """Test if a warning is issued if the data is insufficient."""
        warning = "Your trajectory does not provide sufficient statistics to "
        mocker.patch("scatterkit.lib.util.correlation_time", return_value=-1)
        with pytest.warns(match=warning):
            corrtime = scatterkit.lib.util.correlation_analysis(np.arange(10))
        assert corrtime == -1

    def test_correlated_data(self, mocker):
        """Test if a warning is issued if the data is correlated."""
        corrtime = 10
        warnings = (
            "Your data seems to be correlated with a ",
            f"correlation time which is {corrtime + 1:.2f} ",
            f"of {int(np.ceil(2 * corrtime + 1)):d} to get a ",
        )
        mocker.patch("scatterkit.lib.util.correlation_time", return_value=corrtime)
        for warning in warnings:
            with pytest.warns(match=warning):
                returned_corrtime = scatterkit.lib.util.correlation_analysis(np.arange(10))
        assert returned_corrtime == corrtime

    def test_uncorrelated_data(self, mocker):
        """Test that no warning is issued if the data is uncorrelated."""
        corrtime = 0.25
        mocker.patch("scatterkit.lib.util.correlation_time", return_value=corrtime)
        with warnings.catch_warnings():  # no warning should be issued
            warnings.simplefilter("error")
            returned_corrtime = scatterkit.lib.util.correlation_analysis(np.arange(10))

        assert returned_corrtime == corrtime

    def test_no_data(self):
        """Test that no warning is issued if no data exists."""
        with warnings.catch_warnings():  # no warning should be issued
            warnings.simplefilter("error")
            returned_corrtime = scatterkit.lib.util.correlation_analysis(
                np.nan * np.arange(10)
            )
        assert returned_corrtime == -1


class Testget_center:
    """Test the `get_center` function."""

    compounds = ["group", "segments", "residues", "molecules", "fragments"]

    @pytest.fixture
    def ag(self):
        """An AtomGroup made from water molecules."""
        return mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)

    @pytest.mark.parametrize("compound", compounds)
    def cog(self, ag, compound):
        """Test same center of geometry."""
        assert_equal(
            scatterkit.lib.util.get_center(
                atomgroup=ag, bin_method="cog", compound=compound
            ),
            ag.center_of_geometry(compound=compound),
        )

    @pytest.mark.parametrize("compound", compounds)
    def com(self, ag, compound):
        """Test same center of mass."""
        assert_equal(
            scatterkit.lib.util.get_center(
                atomgroup=ag, bin_method="com", compound=compound
            ),
            ag.center_of_mass(compound=compound),
        )

    @pytest.mark.parametrize("compound", compounds)
    def coc(self, ag, compound):
        """Test same center of charge."""
        assert_equal(
            scatterkit.lib.util.get_center(
                atomgroup=ag, bin_method="cog", compound=compound
            ),
            ag.center_of_charge(compound=compound),
        )

    def test_get_center_unknown(self):
        """Test a wrong bin_method."""
        with pytest.raises(ValueError, match="'foo' is an unknown binning"):
            scatterkit.lib.util.get_center(atomgroup=None, bin_method="foo", compound=None)


class TestUnitVectors:
    """Test the `unit_vectors` functions."""

    @pytest.mark.parametrize("pdim", [0, 1, 2])
    def test_unit_vectors_planar(self, pdim):
        """Test calculation of planar unit vectors."""
        unit_vectors = np.zeros(3)
        unit_vectors[pdim] += 1

        assert_equal(
            scatterkit.lib.util.unit_vectors_planar(
                atomgroup=None, grouping=None, pdim=pdim
            ),
            unit_vectors,
        )
