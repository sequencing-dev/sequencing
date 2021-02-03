# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import qutip
import numpy as np
import matplotlib.pyplot as plt
from .sequencing import ket2dm, Sequence, PulseSequence, CompiledPulseSequence


class Benchmark(object):
    """A class for comparing the performance of a given
    control sequence to a target unitary.

    Args:
        sequence (PulseSequence, CompiledPulseSequence, or Sequence):
            The sequence to benchmark.
        init_state (qutip.Qobj): The initial state of the system.
        target_unitary (qutip.Qobj): The unitary to which to compare the
            sequence.
        run_sequence (optional, bool): Whether to run the sequence immediately
            upon creating the Benchmark. Default: True.
    """

    def __init__(self, sequence, init_state, target_unitary, run_sequence=True):
        allowed_sequence_types = (PulseSequence, CompiledPulseSequence, Sequence)
        if not isinstance(sequence, allowed_sequence_types):
            raise TypeError(
                f"Expected sequence to be one of the following types: "
                f'({", ".join(allowed_sequence_types)}), '
                f"but got {type(sequence)}."
            )
        self.seq = sequence
        self.init_state = init_state
        self.target_unitary = target_unitary
        self.target_state = (self.target_unitary * self.init_state).unit()
        self.mesolve_state = None
        if run_sequence:
            self.run_sequence()

    def run_sequence(self, **kwargs):
        """Simulate the sequence and save the final state.

        Keyword arguments are passed to ``self.seq.run()``.
        """
        result = self.seq.run(self.init_state, **kwargs)
        self.mesolve_state = result.states[-1]

    def fidelity(self, target_state=None):
        """Returns the fidelty of the state resulting from the sequence
        to the target state.

        Args:
            target_state (optional, qutip.Qobj): State to which to compare the
                final state. Defaults to``target_unitary * init_state``.

        Returns:
            float or None
        """
        if self.mesolve_state is None:
            return
        target_state = target_state or self.target_state
        return qutip.fidelity(self.mesolve_state, target_state) ** 2

    def tracedist(self, target_state=None, sparse=False, tol=0):
        """Returns the trace distance from the state resulting
        from the sequence to the target state.

        Args:
            target_state (optional, qutip.Qobj): State to which to compare the
                final state. Defaults to``target_unitary * init_state``.
            sparse, tol: See ``qutip.tracedist``.

        Returns:
            float or None
        """
        if self.mesolve_state is None:
            return
        target_state = target_state or self.target_state
        state = self.mesolve_state
        return qutip.tracedist(state, target_state, sparse=sparse, tol=tol)

    def purity(self):
        """Returns the purity of the final state.

        Returns:
            float or None
        """
        if self.mesolve_state is None:
            return
        return np.trace(ket2dm(self.mesolve_state) ** 2).real

    def plot_wigners(
        self,
        target_state=None,
        actual_state=None,
        sel=None,
        cmap="RdBu",
        disp_range=(-5, 5, 201),
    ):
        """Plots the Wigner function of the final state and the target state.

        Args:
            target_state (optional, qutip.Qobj): State to which to compare the
                final state. Defaults to``target_unitary * init_state``.
            actual_state (optional, qutip.Qobj): State to compare the
                target_state. Defaults to ``self.mesolve_state``.
            sel (optional, int or list[int]): Indices of modes to keep when
                taking the partial trace of the target and actual states.
                If None, no partial trace is taken. Default: None.
            cmap (optional, str): Name of the matplotlib colormap to use.
                Default: 'RdBu'
            disp_range (tuple[float, float, int]): Range of displacements to
                use when compouting the Wigner function, specified as
                (start, stop, num_steps). Default: (-5, 5, 201).

        Returns:
            tuple: matplotlib Figure and Axis.
        """
        if self.mesolve_state is None:
            return
        target_state = target_state or self.target_state
        actual_state = actual_state or self.mesolve_state
        if sel is not None:
            target_state = target_state.ptrace(sel)
            actual_state = actual_state.ptrace(sel)
        xs = ys = np.linspace(*disp_range)
        w = qutip.wigner(actual_state, xs, ys)
        w0 = qutip.wigner(target_state, xs, ys)
        clim = max(abs(w.min()), abs(w.max()), abs(w0.min()), abs(w0.max()))
        norm = plt.Normalize(-clim, clim)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        labels = ["target state", "mesolve state"]
        for ax, wigner, title in zip(axes, [w0, w], labels):
            im = ax.pcolormesh(xs, ys, wigner, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel(r"Re[$\alpha$]")
            ax.set_ylabel(r"Im[$\alpha$]")
        fig.colorbar(im, ax=axes, orientation="horizontal")
        fig.suptitle(f"{self.seq.system.name} Wigner")
        return fig, axes

    def plot_fock_distribution(
        self,
        target_state=None,
        actual_state=None,
        sel=None,
        offset=0,
        ax=None,
        unit_y_range=True,
    ):
        """Plots the photon number distribution of the
        target and actual states.

        Args:
            target_state (optional, qutip.Qobj): State to which to compare the
                final state. Defaults to``target_unitary * init_state``.
            actual_state (optional, qutip.Qobj): State to compare the
                target_state. Defaults to ``self.mesolve_state``.
            sel (optional, int or list[int]): Indices of modes to keep when
                taking the partial trace of the target and actual states.
                If None, no partial trace is taken. Default: None.
            offset (optional, int): Minimum photon number to plot. Default: 0.
            ax (optional, Axis): matplotlib axis on which to plot. If None,
                one is automatically created. Default: None.
            unit_y_range (optional, bool): Whether to force the y axis limits
                to (0, 1). Default: True.

        Returns:
            tuple: matplotlib Figure and Axis.
        """
        if self.mesolve_state is None:
            return
        if ax is not None:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1, 1)

        target_state = target_state or self.target_state
        actual_state = actual_state or self.mesolve_state
        if sel is not None:
            target_state = target_state.ptrace(sel)
            actual_state = actual_state.ptrace(sel)
        labels = ["target state", "mesolve state"]
        for rho, label in zip([target_state, actual_state], labels):
            rho = ket2dm(rho)
            N = rho.shape[0]
            xs = np.arange(offset, offset + N)
            ys = rho.diag().real
            ax.bar(xs, ys, alpha=0.5, width=0.8, label=label)
        if unit_y_range:
            ax.set_ylim(0, 1)
        ax.set_xlim(-0.5 + offset, N + offset)
        ax.set_xlabel("Fock number", fontsize=12)
        ax.set_ylabel("Occupation probability", fontsize=12)
        ax.legend(loc=0)
        ax.set_title(f"{self.seq.system.name} Fock distribution")
        return fig, ax
