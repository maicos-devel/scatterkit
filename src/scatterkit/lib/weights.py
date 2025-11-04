#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weight functions used for the RDFDiporder."""

import MDAnalysis as mda
import numpy as np
from scipy import constants

def diporder_pair_weights(
    g1: mda.AtomGroup, g2: mda.AtomGroup, compound: str
) -> np.ndarray:
    """Normalized dipole moments as weights for general diporder RDF calculations."""
    dipoles_1 = g1.dipole_vector(compound=compound)
    dipoles_2 = g2.dipole_vector(compound=compound)

    dipoles_1 /= np.linalg.norm(dipoles_1, axis=1)[:, np.newaxis]
    dipoles_2 /= np.linalg.norm(dipoles_2, axis=1)[:, np.newaxis]

    return dipoles_1 @ dipoles_2.T

