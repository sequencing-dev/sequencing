# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import copy

import numpy as np
from tqdm import tqdm
import qutip

from .basic import CompiledPulseSequence
from .common import (
    Operation,
    SyncOperation,
    DelayOperation,
    DelayChannelsOperation,
    ValidatedList,
    delay_channels,
    get_sequence,
)


class PulseSequence(ValidatedList):
    """A list-like container of Operation, SyncOperation,
    DelayOperation, and DelayChannelOperation objects,
    which can be compiled into a CompiledPulseSequence at a later time.

    Args:
        system (optional, System): The System associated with
            the PulseSequence. Default: None.
        t0 (optional, int): The start time of the PulseSequence.
            Default: 0.
        items (optional, iterable): Iterable containing initial
            values with which to populate the PulseSequence. Default: None.
    """

    VALID_TYPES = (
        Operation,
        SyncOperation,
        DelayOperation,
        DelayChannelsOperation,
    )

    def __init__(self, system=None, t0=0, items=None):
        self.system = None
        self.t0 = None
        super().__init__(items)
        self.set_system(system=system, t0=t0)

    def set_system(self, system=None, t0=0):
        """Sets the System and start time for the PulseSequence.

        Args:
            system (optional, System): The System associated with
                the PulseSequence. Default: None.
            t0 (optional, int): The start time of the PulseSequence.
                Default: 0.
        """
        self.system = system
        self.t0 = t0
        self.clear()

    def compile(self):
        """Compiles the PulseSequence into a CompiledPulseSequence.

        Returns:
            CompiledPulseSequence: A new CompiledPulseSequence.
        """
        seq = CompiledPulseSequence(system=self.system, t0=self.t0)
        for item in self:
            item = self._validate(item)
            if isinstance(item, Operation):
                seq.add_operation(item)
            elif isinstance(item, SyncOperation):
                seq.sync()
            elif isinstance(item, DelayOperation):
                seq.delay(
                    item.length,
                    sync_before=item.sync_before,
                    sync_after=item.sync_after,
                )
            else:
                # item is a DelayChannelsOperation
                delay_channels(item.channels, item.length, seq=seq)
        return seq

    @property
    def times(self):
        """Array of times."""
        return self.compile().times

    @property
    def channels(self):
        """Dict of Hamiltonian channels."""
        return self.compile().channels

    def run(
        self,
        init_state,
        c_ops=None,
        e_ops=None,
        options=None,
        progress_bar=None,
    ):
        """Simulate the sequence using qutip.mesolve.

        Args:
            init_state (qutip.Qobj): Inital state of the system.
            c_ops (optional, list): List of additional collapse operators.
                Default: None.
            e_ops (optional, list): List of expectation operators.
                Default: None.
            options (optional, qutip.Options): qutip solver options.
                Note: defaults to max_step = 1.
            progress_bar (optional, None): Whether to use qutip's progress bar.
                Default: None (no progress bar).

        Returns:
            ``qutip.solver.Result``: qutip.solver.Result instance.
        """
        return self.compile().run(
            init_state,
            c_ops=c_ops,
            e_ops=e_ops,
            options=options,
            progress_bar=progress_bar,
        )

    def propagator(
        self,
        c_ops=None,
        options=None,
        unitary_mode="batch",
        parallel=False,
        progress_bar=None,
    ):
        """Calculate the propagator using ``qutip.propagator()``.

        Args:
            c_ops (list[qutip.Qobj]): List of collapse operators.
            options (optional, qutip.Options): qutip solver options.
                Note: defaults to max_step = 1.
            progress_bar (optional, None): Whether to use qutip's progress bar.
                Default: None (no progress bar).
            unitary_mode (optional, "batch" or "single"): Solve all basis vectors
                simulaneously ("batch") or individually ("single").
                Default: "batch".
            parallel (optional, bool): Run the propagator in parallel mode.
                This will override the  unitary_mode settings if set to True.
                Default: False.

        Returns:
            np.ndarray[qutip.Qobj]: Array of Qobjs representing the propagator U(t).
        """
        return self.compile().propagator(
            c_ops=c_ops,
            options=options,
            unitary_mode=unitary_mode,
            parallel=parallel,
            progress_bar=progress_bar,
        )

    def plot_coefficients(self, subplots=True, plot_imag=False, step=False):
        """Plot the Hamiltionian coefficients for all channels.

        Args:
            subplots (optional, bool): If True, plot each channel
                on a different axis. Default: True.
            plot_imag (optional, bool): If True, plot both real and imag.
                Default: False.
            step (optional, bool): It True, use axis.step()
                instead of axis.plot(). Default: False.

        Returns:
            tuple: (fig, ax): matplotlib Figure and axes.
        """
        return self.compile().plot_coefficients(
            subplots=subplots, plot_imag=plot_imag, step=step
        )


class Sequence(ValidatedList):
    """A list-like container of PulseSequence, Operation, and unitary objects,
    which can be used to evolve an initial state.

    Args:
        system (System): System upon which the
            sequence will act.
        operations (optional, list[qutip.Qobj, PulseSequence, Operation]):
            Initial list of Operations or unitaries. Default: None.
    """

    VALID_TYPES = (
        qutip.Qobj,
        Operation,
        PulseSequence,
    )

    def __init__(self, system, operations=None):
        self.system = system
        self.pulse_sequence = get_sequence(system)
        self._t = 0
        super().__init__(operations)

    def _validate(self, item):
        """Enforces that item is an instance of
        one of the types in VALID_TYPES.

        If item is the global PulseSequence, this function will
        make a deepcopy and then reset the PulseSequence.

        Returns:
            `qutip.Qobj``, Operation, or PulseSequence:
            Returns the item if it is a valid type.
        """
        item = super()._validate(item)
        if item is self.pulse_sequence:
            # If we are appending pulse_sequence,
            # make a deepcopy and then reset it.
            item = copy.deepcopy(item)
            self.reset_pulse_sequence()
        elif len(self.pulse_sequence):
            # Otherwise, if pulse_sequence is not empty,
            # capture its contents and reset it before returning
            # the current item.
            self.capture()
        return item

    def reset_pulse_sequence(self):
        """Reset the current pulse sequence."""
        self.pulse_sequence = get_sequence(self.system)

    def capture(self):
        """Appends the current pulse sequence
        if it is not empty.
        """
        if len(self.pulse_sequence):
            self.append(self.pulse_sequence)

    def run(
        self,
        init_state,
        e_ops=None,
        options=None,
        full_evolution=True,
        progress_bar=False,
    ):
        """Evolves init_state using each PulseSequence, Operation,
        or unitary applied in series.

        Args:
            init_state (qutip.Qobj): Initial state to evolve.
            options (optional, qutip.Options): qutip solver options.
                Default: None.
            full_evolution (optional, bool): Whether to store the states
                for every time point in the included Sequences. If False,
                only the final state will be stored. Default: True.
            progress_bar (optional, bool): If True, displays a progress bar
                when iterating through the Sequence. Default:True.

        Returns:
            SequenceResult: SequenceResult containing the
            time evolution of the system.
        """
        self.capture()
        progbar = tqdm if progress_bar else lambda x, **kw: x
        e_ops = e_ops or []
        self._t = 0
        times = [self._t]
        states = [init_state]
        for item in progbar(self):
            item = self._validate(item)
            if isinstance(item, PulseSequence):
                if item.system != self.system:
                    raise ValueError("All operations must have the same system.")
                # The CompiledPulseSequence created by PulseSequence.run()
                # should start at the current Sequence time.
                item.t0 = self._t
                seq = item.compile()
                seq.sync()
                result = seq.run(states[-1], options=options)
                if full_evolution:
                    new_states = result.states
                    new_times = result.times
                else:
                    new_states = result.states[-1:]
                    new_times = result.times[-1:]
                self._t = seq._t
            elif isinstance(item, Operation):
                seq = CompiledPulseSequence(self.system, t0=self._t)
                seq.add_operation(item)
                seq.sync()
                result = seq.run(states[-1], options=options)
                if full_evolution:
                    new_states = result.states
                    new_times = result.times
                else:
                    new_states = result.states[-1:]
                    new_times = result.times[-1:]
                self._t = seq._t
            else:
                # item is a Qobj
                state = states[-1]
                if state.isket:
                    state = item * state
                else:
                    # state is a density matrix
                    state = item * state * item.dag()
                new_states = [state]
                new_times = [times[-1]]
                # Unitaries take zero time, so self._t
                # should be the latest sequence time.
            states.extend(new_states)
            times.extend(new_times)
        times = np.array(times)
        ix = np.argsort(times)
        times = times[ix]
        states = [states[i] for i in ix]
        expect = []
        for op in e_ops:
            expect.append(qutip.expect(op, states))
        num_collapse = len(self.system.c_ops(clean=True))
        result = SequenceResult(
            times=times, states=states, expect=expect, num_collapse=num_collapse
        )
        return result

    def plot_coefficients(self, subplots=True, sharex=True, sharey=True):
        """Plot the Hamiltionian coefficients for all channels.
        Unitaries are represented by vertical lines.

        Args:
            subplots (optional, bool): If True, plot each channel
                on a different axis. Default: True.
            sharex (optional, bool): Share x axes if subplots is True.
                Default: True.
            sharey (optional, bool): Share y axes if subplots is True.
                Default: True.

        Returns:
            tuple: (fig, ax): matplotlib Figure and axes.
        """
        import matplotlib.pyplot as plt

        self._t = 0
        channels = {}
        for item in self:
            item = self._validate(item)
            if isinstance(item, PulseSequence):
                if item.system != self.system:
                    raise ValueError("All operations must have the same system.")
                # The CompiledPulseSequence created by PulseSequence.run()
                # should start at the current Sequence time.
                item.t0 = self._t
                seq = item.compile()
                seq.sync()
                new_times = seq.times
                self._t = new_times.max()
                new_channels = seq.channels
            elif isinstance(item, Operation):
                seq = CompiledPulseSequence(self.system, t0=self._t)
                seq.add_operation(item)
                seq.sync()
                new_times = seq.times
                self._t = new_times.max()
                new_channels = seq.channels
            else:
                new_channels = {"unitary": {"coeffs": np.array([1.0])}}
                new_times = [self._t]
                # Unitaries take zero time, so self._t
                # remains unchanged.
            # Assemble the results for this time step
            for name, info in new_channels.items():
                if name in [] or "coeffs" not in info:
                    continue
                if name in channels:
                    channel_times = np.concatenate(
                        (channels[name][0], np.asarray(new_times))
                    )
                    channel_coeffs = np.concatenate((channels[name][1], info["coeffs"]))
                    channels[name] = (channel_times, channel_coeffs)
                else:
                    channels[name] = (new_times, info["coeffs"])
        channel_names = [n for n in channels if n not in ["delay", "unitary"]]
        if not channel_names:
            raise ValueError("There are no channels with coefficients to plot.")
        if subplots:
            fig, axes = plt.subplots(len(channel_names), sharex=sharex, sharey=sharey)
        else:
            fig, ax = plt.subplots(1, 1)
            axes = [ax] * len(channel_names)
        for name, ax in zip(channel_names, axes):
            ctimes, coeffs = channels[name]
            ax.plot(ctimes, coeffs, label=name)
            ax.legend(loc=0)
            ax.grid(True)
        for ctimes, _ in zip(*channels["unitary"]):
            if subplots:
                for a in axes:
                    a.axvline(ctimes, ls="--", lw=1.5, color="k", alpha=0.25)
            else:
                ax.axvline(ctimes, ls="--", lw=1.5, color="k", alpha=0.25)
        fig.suptitle("Hamiltonian coefficients")
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        axes[-1].set_xlabel("Time")
        if subplots:
            return fig, axes
        return fig, ax


class SequenceResult(object):
    """An object that mimics qutip.solver.Result
    for Sequences.

    Attributes:
        times (np.ndarray): Array of times.
        states (list[qutip.Qobj]): List of states.
        expect (list[np.ndarray]): List of arrays of expectation
            values.
        num_expect (int): Number of expectation operators,
            ``num_expect == len(expect)``.
        num_collapse (int): Number of collapse operators involved
            in the Sequence.
        solver (str): Name of the 'solver' used to generate the
            SequenceResult. This is always 'sequencing.Sequence'.
    """

    def __init__(self, times=None, states=None, expect=None, num_collapse=0):
        if times is None:
            times = np.array([])
        self.times = times
        if expect is None:
            expect = []
        self.expect = expect
        if states is None:
            states = []
        self.states = states
        self.num_collapse = num_collapse
        self.solver = "sequencing.Sequence"

    @property
    def num_expect(self):
        return len(self.expect)

    def __str__(self):
        s = ["SequenceResult"]
        s.append("-" * len(s[-1]))
        if self.times is not None and len(self.times) > 0:
            s.append(f"Number of times: {len(self.times)}")
        if self.states is not None and len(self.states) > 0:
            s.append("states = True")
        if self.expect is not None and len(self.expect) > 0:
            s.append(f"expect = True, num_expect = {self.num_expect}")
        s.append(f"num_collapse = {self.num_collapse}")
        return "\n".join(s)

    def __repr__(self):
        return self.__str__()


_global_pulse_sequence = PulseSequence()
