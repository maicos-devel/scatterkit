#!/usr/bin/env python3
"""init file for datafiles."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

DIR_PATH = Path(__file__).parent
EXAMPLES = DIR_PATH / ".." / ".." / "examples"


# bulk water (NpT)
WATER_GRO_NPT = DIR_PATH / "water/water.gro"
WATER_TRR_NPT = DIR_PATH / "water/water.trr"
WATER_2F_TRR_NPT = DIR_PATH / "water/water_two_frames.trr"
WATER_TPR_NPT = DIR_PATH / "water/water.tpr"

# bulk water (NVT)
WATER_TPR_NVT = EXAMPLES / "water_nvt.tpr"
WATER_XTC_NVT = EXAMPLES / "water_nvt.xtc"


# An SPC/E water molecule pointing in z-direction
SPCE_ITP = DIR_PATH / "spce.itp"
SPCE_GRO = DIR_PATH / "spce.gro"


