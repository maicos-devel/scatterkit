# -*-
#
# Copyright (c) 2025 Authors and contributors (see the AUTHORS.rst file for the full
# list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helper functions for mathematical and physical operations."""

import numpy as np
from scipy.fftpack import dst

from . import tables
from ._cmath import compute_structure_factor  # noqa: F401

# Max spacing variation in series that is allowed
dt_dk_tolerance = 1e-8  # (~1e-10 suggested)
dr_tolerance = 1e-6


def atomic_form_factor(q: float, element: str) -> float:
    r"""Calculate atomic form factor :math:`f(q)` for X-ray scattering.

    The atomic form factor :math:`f(q)` is a measure of the scattering
    amplitude of a wave by an **isolated** atom

    .. attention::

        The atomic form factor should not be confused with the atomic scattering factor
        or intensity (often anonymously called form factor). The scattering intensity
        depends strongly on the distribution of atoms and can be computed using
        :class:`scatterkit.Saxs`.

    Here, :math:`f(q)` is computed in terms of the scattering vector as

    .. math::
        f(q) = \sum_{i=1}^4 a_i e^{-b_i q^2/(4\pi)^2} + c \,.

    The coefficients :math:`a_{1,\dots,4}`, :math:`b_{1,\dots,4}` and :math:`c` are also
    known as Cromer-Mann X-ray scattering factors and are documented in
    :footcite:t:`princeInternationalTablesCrystallography2004` and taken from the `TU
    Graz
    <https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php>`_
    and stored in :obj:`scatterkit.lib.tables.CM_parameters`.

    Parameters
    ----------
    q : float
        The magnitude of the scattering vector in reciprocal angstroms (1/Å).
    element : str
        The element for which the atomic form factor is calculated. Known elements are
        listed in the :attr:`scatterkit.lib.tables.elements` set. United-atom models
        such as ``"CH1"``, ``"CH2"``, ``"CH3"``, ``"CH4"``, ``"NH1"``, ``"NH2"``,
        and ``"NH3"`` are also supported.

        .. note::

            ``element`` is converted to title case to avoid most common issues with
            MDAnalysis which uses upper case elements by default. For example ``"MG"``
            will be converted to ``"Mg"``.

    Returns
    -------
    float
        The calculated atomic form factor for the specified element and q in units of
        electrons.

    """
    if element == "CH1":
        return atomic_form_factor(q, "C") + atomic_form_factor(q, "H")
    if element == "CH2":
        return atomic_form_factor(q, "C") + 2 * atomic_form_factor(q, "H")
    if element == "CH3":
        return atomic_form_factor(q, "C") + 3 * atomic_form_factor(q, "H")
    if element == "CH4":
        return atomic_form_factor(q, "C") + 4 * atomic_form_factor(q, "H")
    if element == "NH1":
        return atomic_form_factor(q, "N") + atomic_form_factor(q, "H")
    if element == "NH2":
        return atomic_form_factor(q, "N") + 2 * atomic_form_factor(q, "H")
    if element == "NH3":
        return atomic_form_factor(q, "N") + 3 * atomic_form_factor(q, "H")

    if element.title() not in tables.CM_parameters:
        raise ValueError(
            f"Element '{element}' not found. Known elements are listed in the "
            "`scatterkit.lib.tables.elements` set."
        )
    # q / (4 * pi) = sin(theta) / lambda
    q2 = np.asarray((q / (4 * np.pi)) ** 2)

    CM_parameter = tables.CM_parameters[element.title()]

    q2_flat = q2.flatten()
    form_factor = (
        np.sum(CM_parameter.a * np.exp(-CM_parameter.b * q2_flat[:, None]), axis=1)
        + CM_parameter.c
    )

    return form_factor.reshape(q2.shape)


def rdf_structure_factor(
    rdf: np.ndarray, r: np.ndarray, density: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes the structure factor based on the radial distribution function (RDF).

    The structure factor :math:`S(q)` based on an RDF :math:`g(r)` is given by

    .. math::
        S(q) = 1 + 4 \pi \rho \int_0^\infty \mathrm{d}r r
                         \frac{\sin(qr)}{q} (g(r) - 1)\,

    where :math:`q` is the magnitude of the scattering vector. The calculation is
    performed via a discrete sine transform as implemented in :func:`scipy.fftpack.dst`.

    For an `example` take a look at :ref:`howto-saxs`.

    Parameters
    ----------
    rdf : numpy.ndarray
        radial distribution function
    r : numpy.ndarray
        equally spaced distance array on which rdf is defined
    density : float
        number density of particles

    Returns
    -------
    q : numpy.ndarray
        array of q points
    struct_factor : numpy.ndarray
        structure factor

    Raises
    ------
    ValueError
        If the distance array ``r`` is not equally spaced.
    """
    dr = (r[-1] - r[0]) / float(len(r) - 1)

    if (abs(np.diff(r) - dr) > dr_tolerance).any():
        raise ValueError("Distance array `r` is not equally spaced!")

    q = np.pi / r[-1] * np.arange(1, len(r) + 1)
    struct_factor = 1 + 4 * np.pi * density * 0.5 * dst((rdf - 1) * r) / q * dr

    return q, struct_factor


def compute_rdf_structure_factor(
    rdf: np.ndarray, r: np.ndarray, density: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes the structure factor based on the radial distribution function (RDF).

    The structure factor :math:`S(q)` based on the RDF :math:`g(r)` is given by

    .. math::
        S(q) = 1 + 4 \pi \rho \int_0^\infty \mathrm{d}r r
                         \frac{\sin(qr)}{q} (g(r) - 1)\,

    where :math:`q` is the magnitude of the scattering vector. The calculation is
    performed via a discrete sine transform as implemented in :func:`scipy.fftpack.dst`.

    For an `example` take a look at :ref:`howto-saxs`.

    Parameters
    ----------
    rdf : numpy.ndarray
        radial distribution function
    r : numpy.ndarray
        equally spaced distance array on which rdf is defined
    density : float
        number density of particles

    Returns
    -------
    q : numpy.ndarray
        array of q points
    struct_factor : numpy.ndarray
        structure factor

    """
    drs = r[1:] - r[:-1]
    diff = drs - np.mean(drs)
    if not np.all(diff < 1e-6):
        raise ValueError("Distance array `r` is not equally spaced!")

    dr = r[1] - r[0]
    q = np.pi / r[-1] * np.arange(1, len(r) + 1)
    struct_factor = 1 + 4 * np.pi * density * 0.5 * dst((rdf - 1) * r) / q * dr

    return q, struct_factor
