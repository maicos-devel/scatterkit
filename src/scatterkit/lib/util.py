#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helper and utilities functions that don't fit anywhere else."""

import re
from collections.abc import Callable

DOC_REGEX_PATTERN = re.compile(r"\$\{([^\}]+)\}")

DOC_DICT = dict(
    #####################
    # DESCRIPTION SECTION
    #####################
    SAVE_METHOD_DESCRIPTION="Save results of analysis to file specified by ``output``.",
    RUN_METHOD_DESCRIPTION="""Iterate over the trajectory.

Parameters
----------
start : int
    start frame of analysis
stop : int
    stop frame of analysis
step : int
    number of frames to skip between each analysed frame
frames : array_like
    array of integers or booleans to slice trajectory; ``frames`` can only be
    used *instead* of ``start``, ``stop``, and ``step``. Setting *both*
    ``frames`` and at least one of ``start``, ``stop``, ``step`` to a
    non-default value will raise a :exc:`ValueError`.
verbose : bool
    Turn on verbosity
progressbar_kwargs : dict
    ProgressBar keywords with custom parameters regarding progress bar position,
    etc; see :class:`MDAnalysis.lib.log.ProgressBar` for full list.

Returns
-------
self : object
    analysis object
""",
    ##########################
    # SINGLE PARAMETER SECTION
    ##########################
    ATOMGROUP_PARAMETER="""atomgroup : MDAnalysis.core.groups.AtomGroup
    A :class:`~MDAnalysis.core.groups.AtomGroup` for which the calculations are
    performed.""",
    OUTPUT_PARAMETER="""output : str
    Output filename.""",
    ###################################
    # MULTI/COMBINES PARAMETERS SECTION
    ###################################
    BASE_CLASS_PARAMETERS="""refgroup : MDAnalysis.core.groups.AtomGroup
    Reference :class:`~MDAnalysis.core.groups.AtomGroup` used for the calculation. If
    ``refgroup`` is provided, the calculation is performed relative to the center of
    mass of the AtomGroup. If ``refgroup`` is :obj:`None` the calculations are performed
    with respect to the center of the (changing) box.
unwrap : bool
    When :obj:`True`, molecules that are broken due to the periodic boundary conditions
    are made whole.

    If the input contains molecules that are already whole, speed up the calculation by
    disabling unwrap. To do so, use the flag ``-no-unwrap`` when using MAICoS from the
    command line, or use ``unwrap=False`` when using MAICoS from the Python interpreter.

    Note: Molecules containing virtual sites (e.g. TIP4P water models) are not currently
    supported in MDAnalysis. In this case, you need to provide unwrapped trajectory
    files directly, and disable unwrap. Trajectories can be unwrapped, for example,
    using the ``trjconv`` command of GROMACS.
pack : bool
    When :obj:`True`, molecules are put back into the unit cell. This is required
    because MAICoS only takes into account molecules that are inside the unit cell.

    If the input contains molecules that are already packed, speed up the calculation by
    disabling packing with ``pack=False``.
jitter : float
    Magnitude of the random noise to add to the atomic positions.

    A jitter can be used to stabilize the aliasing effects sometimes appearing when
    histogramming data. The jitter value should be about the precision of the
    trajectory. In that case, using jitter will not alter the results of the histogram.
    If ``jitter = 0.0`` (default), the original atomic positions are kept unchanged.

    You can estimate the precision of the positions in your trajectory with
    :func:`maicos.lib.util.trajectory_precision`. Note that if the precision is not the
    same for all frames, the smallest precision should be used.
concfreq : int
    When concfreq (for conclude frequency) is larger than ``0``, the conclude function
    is called and the output files are written every ``concfreq`` frames.""",
    Q_SPACE_PARAMETERS="""qmin : float
    Starting q (1/Å)
qmax : float
    Ending q (1/Å)
dq : float
    bin_width (1/Å)""",
)
"""Dictionary containing the keys and the actual docstring used by :func:`scatterkit.lib.util.render_docs`.

    :meta hide-value:
"""  # noqa: E501


def _render_docs(func: Callable, doc_dict: dict = DOC_DICT) -> Callable:
    if func.__doc__ is not None:
        while True:
            keys = DOC_REGEX_PATTERN.findall(func.__doc__)
            if not keys:
                break  # Exit the loop if no more patterns are found
            for key in keys:
                func.__doc__ = func.__doc__.replace(f"${{{key}}}", doc_dict[key])
    return func


def render_docs(func: Callable) -> Callable:
    """Replace all template phrases in the functions docstring.

    Keys for the replacement are taken from in :attr:`scatterkit.lib.util.DOC_DICT`.

    Parameters
    ----------
    func : callable
        The callable (function, class) where the phrase old should be replaced.

    Returns
    -------
    Callable
        callable with replaced phrase

    """
    return _render_docs(func, doc_dict=DOC_DICT)


def scatterkit_banner(version: str = "", frame_char: str = "-") -> str:
    """Prints ASCII banner resembling the MAICoS Logo with 80 chars width.

    Parameters
    ----------
    version : str
        Version string to add to the banner.
    frame_char : str
        Character used to as framing around the banner.

    Returns
    -------
    banner : str
        formatted banner

    """
    banner = rf"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@                           SCATTERKIT {version:^8}                                @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
    return banner.replace("@", frame_char)
