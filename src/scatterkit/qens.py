#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Module for computing the incoherent intermediate scattering function.

Computes the self-part of the intermediate scattering function
:math:`F_s(q, t)` via streaming multi-tau correlation provided by
:class:`maicos.lib.correlator.MultiTauCorrelator` through the
``_corr`` slot of :class:`maicos.core.AnalysisBase`.
"""

import logging
import warnings

import MDAnalysis as mda
import numpy as np
from maicos.core import AnalysisBase

from .lib.util import render_docs


def _generate_q_vectors(
    box_lengths: np.ndarray,
    qmin: float,
    qmax: float,
    dq: float,
    max_q_vectors: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate q-vectors grouped into non-empty shells by magnitude.

    All q-vectors compatible with the simulation box that fall within the
    requested q-range are generated deterministically from Miller indices.
    An optional cap ``max_q_vectors`` can limit the number per shell for
    performance; when set, vectors are selected by taking every n-th vector
    (stride sampling) to preserve orientational coverage.

    Parameters
    ----------
    box_lengths : numpy.ndarray
        Diagonal box lengths (Lx, Ly, Lz) in Angstrom.
    qmin : float
        Minimum q magnitude (1/Angstrom).
    qmax : float
        Maximum q magnitude (1/Angstrom).
    dq : float
        Width of q-shells for binning (1/Angstrom).
    max_q_vectors : int or None
        Maximum number of q-vectors per shell. If ``None`` (default), all
        vectors in each shell are used. When set, vectors are stride-sampled
        to maintain uniform orientational coverage.

    Returns
    -------
    q_bin_values : numpy.ndarray
        Mean :math:`|q|` of the vectors actually used in each non-empty shell.
    q_vectors_per_shell : list of numpy.ndarray
        For each non-empty shell, an ``(n_selected, 3)`` array of q-vectors.

    """
    # Under PBC, only q = 2π n / L for integer Miller indices n = (h, k, l)
    # are allowed. Enumerate every combination whose |q| can reach qmax.
    q_factor = 2 * np.pi / box_lengths  # (3,)
    max_n = np.ceil(qmax / q_factor).astype(int)

    hh = np.arange(-max_n[0], max_n[0] + 1)
    kk = np.arange(-max_n[1], max_n[1] + 1)
    ll = np.arange(-max_n[2], max_n[2] + 1)
    miller = np.array(np.meshgrid(hh, kk, ll, indexing="ij")).reshape(3, -1).T
    miller = miller[np.any(miller != 0, axis=1)]  # drop (0, 0, 0)

    q_vecs = miller * q_factor[np.newaxis, :]  # (N, 3)
    q_mags = np.linalg.norm(q_vecs, axis=1)

    mask = (q_mags >= qmin) & (q_mags <= qmax)
    q_vecs = q_vecs[mask]
    q_mags = q_mags[mask]

    n_shells = int(np.ceil((qmax - qmin) / dq))
    q_bin_edges = np.linspace(qmin, qmin + n_shells * dq, n_shells + 1)
    shell_indices = np.digitize(q_mags, q_bin_edges) - 1

    q_vectors_per_shell: list[np.ndarray] = []
    q_bin_values: list[float] = []
    for i_shell in range(n_shells):
        in_shell = shell_indices == i_shell
        if not in_shell.any():
            continue
        vecs = q_vecs[in_shell]
        mags = q_mags[in_shell]

        if max_q_vectors is not None and len(vecs) > max_q_vectors:
            stride = len(vecs) // max_q_vectors
            sel = np.arange(0, len(vecs), stride)[:max_q_vectors]
            vecs = vecs[sel]
            mags = mags[sel]

        q_vectors_per_shell.append(vecs)
        q_bin_values.append(float(np.mean(mags)))

    return np.asarray(q_bin_values), q_vectors_per_shell


@render_docs
class Qens(AnalysisBase):
    r"""Incoherent intermediate scattering function :math:`F_s(q, t)` via multi-tau.

    Computes the self-part of the intermediate scattering function

    .. math::

        F_s(q, t) = \frac{1}{N} \left\langle \sum_{i=1}^{N}
        \exp\!\big(i\,\boldsymbol{q} \cdot
        [\boldsymbol{r}_i(t_0 + t) - \boldsymbol{r}_i(t_0)]\big)
        \right\rangle_{t_0}

    by streaming the per-atom phase factors
    :math:`e^{i\boldsymbol{q}\cdot\boldsymbol{r}_i(t)}` into the multi-tau
    correlator provided by :class:`maicos.core.AnalysisBase` via its
    ``_corr`` slot. The trajectory is processed in a single pass with bounded
    memory; lag times are logarithmically spaced.

    For each q-shell, all q-vector orientations compatible with the simulation
    box are used and averaged (powder averaging). Requires an orthorhombic
    simulation cell.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${Q_SPACE_PARAMETERS}
    max_q_vectors : int or None
        Maximum number of q-vectors per shell. If :py:obj:`None`, all
        lattice-compatible vectors in each shell are used. Default is 6:
        the powder-average variance budget is dominated by the per-atom
        average ($\propto 1/N$), so once N is large adding more
        orientations gives rapidly diminishing returns while the
        correlator memory scales linearly with the total q-vector count.
    mem_warn_gb : float
        Threshold (in GB) for emitting a :class:`UserWarning` about the
        estimated peak memory of the streaming correlator. Defaults to
        :py:obj:`8.0`. Set to :py:obj:`numpy.inf` to silence the warning.
    ${BASE_CLASS_PARAMETERS}
    ${OUTPUT_PARAMETER}

    Attributes
    ----------
    results.lag_times : numpy.ndarray
        Lag times in the same time unit as the trajectory.
    results.q_values : numpy.ndarray
        Scattering vector magnitudes for each non-empty q-shell.
    results.F_s : numpy.ndarray
        Incoherent intermediate scattering function, shape ``(n_q, n_tau)``.

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        qmin: float = 0.2,
        qmax: float = 2.0,
        dq: float = 0.1,
        max_q_vectors: int | None = 6,
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = False,
        jitter: float = 0.0,
        concfreq: int = 0,
        correlator_num_levels: int = 20,
        correlator_channels_per_level: int = 16,
        mem_warn_gb: float = 8.0,
        output: str = "fs_qt.dat",
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            wrap_compound="atoms",
            correlator_num_levels=correlator_num_levels,
            correlator_channels_per_level=correlator_channels_per_level,
        )
        self.qmin = qmin
        self.qmax = qmax
        self.dq = dq
        self.max_q_vectors = max_q_vectors
        self.mem_warn_gb = mem_warn_gb
        self.output = output

    def _prepare(self) -> None:
        logging.info(
            "Analysis of the incoherent intermediate scattering function F_s(q, t)."
        )

        # The Miller-index q-grid in `_generate_q_vectors` assumes axis-aligned
        # cell vectors; a triclinic cell would need the full reciprocal basis.
        cell = mda.lib.mdamath.triclinic_vectors(self._universe.dimensions)
        if not np.allclose(cell - np.diag(np.diag(cell)), 0):
            raise NotImplementedError(
                "Qens currently supports only orthorhombic simulation cells."
            )
        box = np.diag(cell)

        self._q_bin_values, self._q_vectors_per_shell = _generate_q_vectors(
            box_lengths=box,
            qmin=self.qmin,
            qmax=self.qmax,
            dq=self.dq,
            max_q_vectors=self.max_q_vectors,
        )

        # `_all_q_vecs` is the flat (total_q, 3) view we feed into the
        # correlator; `_shell_of_q[i]` records which shell row i came from so
        # we can powder-average per-q correlations within each shell in
        # `_conclude`.
        if self._q_vectors_per_shell:
            self._all_q_vecs = np.vstack(self._q_vectors_per_shell)
            self._shell_of_q = np.concatenate([
                np.full(len(qv), i)
                for i, qv in enumerate(self._q_vectors_per_shell)
            ])
        else:
            self._all_q_vecs = np.empty((0, 3))
            self._shell_of_q = np.empty(0, dtype=int)

        logging.info(
            f"Streaming {len(self._all_q_vecs)} q-vectors across "
            f"{len(self._q_vectors_per_shell)} q-shells."
        )

        self._check_memory_budget()

    def _check_memory_budget(self) -> None:
        """Estimate streaming-correlator memory and warn if above threshold.

        The :class:`maicos.lib.correlator.MultiTauCorrelator` holds a buffer and
        an accumulator of shape ``(p, m, n_atoms, total_q)`` in complex128 (16
        bytes/element). On read, ``self.correlation.phases`` materialises an
        additional ``(n_lags, n_atoms, total_q)`` array of the same dtype.
        """
        n_atoms = self.atomgroup.n_atoms
        total_q = len(self._all_q_vecs)
        if total_q == 0:
            return

        p = self.correlator_num_levels
        m = self.correlator_channels_per_level
        n_lags = m + (p - 1) * (m // 2)
        bytes_per_elem = 16  # complex128

        buf_bytes = p * m * n_atoms * total_q * bytes_per_elem
        out_bytes = n_lags * n_atoms * total_q * bytes_per_elem
        peak_bytes = 2 * buf_bytes + out_bytes
        peak_gb = peak_bytes / 1e9

        logging.info(
            "Estimated peak memory for streaming correlator: "
            f"{peak_gb:.2f} GB "
            f"(buffer+accumulator {2 * buf_bytes / 1e9:.2f} GB, "
            f"correlation output {out_bytes / 1e9:.2f} GB) "
            f"[n_atoms={n_atoms}, total_q={total_q}, p={p}, m={m}]."
        )

        if peak_gb > self.mem_warn_gb:
            warnings.warn(
                f"Qens streaming correlator estimated peak memory "
                f"~{peak_gb:.1f} GB exceeds threshold "
                f"({self.mem_warn_gb:.1f} GB). To reduce memory: lower "
                f"`max_q_vectors` (currently "
                f"{self.max_q_vectors!r}), coarsen `dq`, reduce "
                f"`correlator_num_levels`, or raise `mem_warn_gb` to silence.",
                UserWarning,
                stacklevel=3,
            )

    def _single_frame(self) -> float:
        if len(self._all_q_vecs) == 0:
            return 0.0

        positions = self.atomgroup.positions  # (n_atoms, 3)
        # phases[j, k] = exp(i q_k . r_j(t)). The base class correlates this
        # element-wise, so every (atom, q) pair gets its own autocorrelation.
        self._corr.phases = np.exp(1j * (positions @ self._all_q_vecs.T))
        return 0.0

    def _conclude(self) -> None:
        if (
            not self._correlators
            or "phases" not in self.correlation
            or len(self.times) < 2
        ):
            logging.warning("No correlation data collected.")
            self.results.lag_times = np.array([])
            self.results.q_values = np.array([])
            self.results.F_s = np.array([])
            return

        # correlation.phases has shape (n_lags, n_atoms, total_q), complex.
        # Mean over atoms then real part gives the per-q-vector F_s; we then
        # powder-average within each shell.
        per_q = self.correlation.phases.mean(axis=1).real
        n_shells = len(self._q_vectors_per_shell)
        F_s = np.zeros((n_shells, per_q.shape[0]))
        for i_shell in range(n_shells):
            F_s[i_shell] = per_q[:, self._shell_of_q == i_shell].mean(axis=1)

        # self.lags arrives from MAiCoS in units of frame intervals; convert
        # to physical time via the actually-sampled dt (which already reflects
        # any user-supplied stride).
        dt = float(self.times[1] - self.times[0])
        self.results.lag_times = self.lags * dt
        self.results.q_values = self._q_bin_values
        self.results.F_s = F_s

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        lag_times = self.results.lag_times
        F_s = self.results.F_s  # (n_q, n_tau)

        data = np.column_stack([lag_times, F_s.T])
        columns = ["t"] + [f"F_s(q={q:.3f})" for q in self.results.q_values]

        self.savetxt(self.output, data, columns=columns)
