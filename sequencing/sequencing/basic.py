# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import numpy as np
import qutip


class HamiltonianChannels(object):
    """Object prepresenting a set of Hamiltonian 'channels'
    with time-dependent coefficients.

    Args:
        channels (optional, dict): Dict of
            {channel_name: {'H': hamiltonian_term, 'time_dependent': td_bool}},
            where hamiltonian_term is a qutip.Qobj and td_bool is
            a boolean flag indicating whether this Hamiltonian will be
            given time-dependence (defaults to False). Default: None.
        collapse_channels (optional, dict): Similar to channels, but for
            collapse operators instead of Hamiltonian terms. Default: None.
    """

    def __init__(self, channels=None, collapse_channels=None, t0=0):
        self.channels = {}
        if channels is None:
            channels = {}
        for name, info in channels.items():
            H = info["H"]
            time_dependent = info.get("time_dependent", False)
            self.add_channel(name, H=H, time_dependent=time_dependent)
        self.collapse_channels = {}
        if collapse_channels is None:
            collapse_channels = {}
        for name, info in collapse_channels.items():
            op = info["op"]
            time_dependent = info.get("time_dependent", False)
            self.add_channel(name, C_op=op, time_dependent=time_dependent)
        self.tmin = t0
        self.tmax = t0
        self.dt = 1

    @property
    def times(self):
        """Array of times."""
        return np.arange(self.tmin, self.tmax + self.dt, self.dt).astype(float)

    def add_channel(
        self, name, H=None, C_op=None, time_dependent=False, error_if_exists=True
    ):
        """Add a channel with specified Hamiltonian term H
        or collapse operator C_op.

        Args:
            name (str): Name of the Hamiltonian channel.
            H (optional, qutip.Qobj): Hamiltonian term for this channel.
            C_op (optional, qutip.Qobj): Collapse operator for this channel.
            time_dependent (optional, bool): Whether this channel has
                time-dependent Hamiltonian coefficients. Default: False.
            error_if_exists (optional, bool): Whether to raise an exception
                if a channel with the same name already exists. Default: True.
        """
        if name in self.channels and error_if_exists:
            raise ValueError(f"Channel {name} already exists!")
        if H is not None:
            if H != 0 and not isinstance(H, qutip.Qobj):
                # H could be zero if it's sum([])
                raise TypeError(
                    f"Excpected instance of qutip.Qobj, " f"but got {type(H)}."
                )
            self.channels[name] = {"H": H, "time_dependent": time_dependent, "delay": 0}
        elif C_op is not None:
            # C_op could be zero if it's sum([])
            if C_op != 0 and not isinstance(C_op, qutip.Qobj):
                raise TypeError(
                    f"Excpected instance of qutip.Qobj, " f"but got {type(C_op)}."
                )
            self.collapse_channels[name] = {
                "op": C_op,
                "time_dependent": time_dependent,
                "delay": 0,
            }
        else:
            raise ValueError("Expected H or C_op.")

    def add_operation(
        self,
        channel_name,
        t0=None,
        duration=None,
        times=None,
        H=None,
        C_op=None,
        coeffs=1,
        coeffs_args=None,
        coeffs_kwargs=None,
    ):
        """Add an operation to a given channel,
        padding all other channels accordingly.

        Args:
            channel_name (str): Name of the Hamiltonian channel this
                operation acts on.
            t0 (optional, int): Starting time of this operation
                (required along with duration if you do not provide
                an array of times). Default: None.
            duration (optional, int): Length of this operation
                (required along with t0 if you do not provide
                an array of times). Default: None.
            times (optional, sequence[int]): Sequence of time points.
                Default: None.
            H (optional, qutip.Qobj): Hamiltonian for the given channel
                (required if this channel is not yet in self.channels).
                Default: None.
            C_op (optional, qutip.Qobj): Collapse operator for
                the given channel (required if this channel is
                not yet in self.collapse_channels).
            coeffs (optional, number | sequence | callable): Hamiltonian
                coefficients for the given time points.
                You can specify single int/float/complex for constant
                coefficients, or provide a function that takes time
                as its first argument and returns coefficiencts.
                Default: None.
            coeffs_args (optional, sequence): Positional arguments passed
                to coeffs if coeffs is callable. Default: None.
            coeffs_kwargs (optional, dict): Keyword arguments passed to coeffs
                if coeffs is a function. Default: None.

        **Important note:** if ``reset_t0`` is in coeffs_kwargs and is True,
        then the first argument for coeffs will be ``times-times[0]``
        instead of ``times``. This allows for the user to take care
        of phase bookkeeping.
        """
        if H is not None and C_op is not None:
            raise ValueError("Expected only H or C_op.")
        all_channels = list(self.channels) + list(self.collapse_channels)
        if channel_name not in all_channels:
            self.add_channel(channel_name, H=H, C_op=C_op, time_dependent=True)
        if channel_name in self.channels:
            channel_dict = self.channels
        else:
            channel_dict = self.collapse_channels
        if not channel_dict[channel_name]["time_dependent"]:
            raise ValueError(
                f"Channel {channel_name} is time-independent, "
                "so you cannot add time-dependent coefficients."
            )
        if times is None:
            if t0 is None or duration is None:
                raise ValueError(
                    "You must either specify an array of times " "or t0 and duration."
                )
            delay = int(channel_dict[channel_name]["delay"])
            times = np.arange(t0 + delay, t0 + delay + duration, self.dt)
        if isinstance(coeffs, (int, float, complex)):
            coeffs = coeffs * np.ones_like(times)
        elif callable(coeffs):
            if coeffs_args is None:
                coeffs_args = []
            if coeffs_kwargs is None:
                coeffs_kwargs = {}
            else:
                coeffs_kwargs = coeffs_kwargs.copy()
            reset_t0 = coeffs_kwargs.pop("reset_t0", False)
            ts = times - times[0] if reset_t0 else times
            coeffs = coeffs(ts, *coeffs_args, **coeffs_kwargs)
        # First, pad the new coeffs to line up with self.times
        lpad = int(max((times.min() - self.tmin) // self.dt, 0))
        rpad = int(max((self.tmax - times.max()) // self.dt, 0))
        coeffs = np.concatenate([np.zeros(lpad), coeffs, np.zeros(rpad)])
        # Next, calculcate pad widths needed to make other channels
        # line up with this one
        lpad = int(max((self.tmin - times.min()) // self.dt, 0))
        rpad = int(max((times.max() - self.tmax) // self.dt, 0))
        self.tmin = min(times.min(), self.tmin)
        self.tmax = max(times.max(), self.tmax)
        for name, channel in channel_dict.items():
            if "coeffs" in channel:
                # This channel already has an operation on it,
                # so we have to pad it
                old_coeffs = np.concatenate(
                    [np.zeros(lpad), channel["coeffs"], np.zeros(rpad)]
                )
                new_coeffs = old_coeffs
                if name == channel_name:
                    # Sum signals on the same Hamiltonian channel
                    # if they are not separated by a sync().
                    new_coeffs = coeffs + old_coeffs
                channel["coeffs"] = new_coeffs
            elif name == channel_name:
                # This is the channel we're currently adding an operation to,
                # and it doesn't have any operations on it yet
                channel["coeffs"] = coeffs
        # check that padding was done properly
        for name, channel in channel_dict.items():
            if "coeffs" in channel:
                if len(self.times) != len(channel["coeffs"]):
                    msg = (
                        f"Channel {name}: "
                        f"Number of time points {len(self.times)} "
                        f"does not match number of coefficients "
                        f'{len(channel["coeffs"])}.'
                    )
                    raise ValueError(msg)

    def delay_channels(self, channel_names, duration):
        """Add a delay of length ``duration``
        to channels specified in the list ``channel_names``.
        """
        if duration == 0:
            return
        if duration < 0:
            raise ValueError("Delay cannot be less than 0.")
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        for name in channel_names:
            if name not in list(self.channels) + list(self.collapse_channels):
                raise ValueError(f"Channel {name} does not exist.")
        for name in channel_names:
            self.channels[name]["delay"] += duration

    def build_hamiltonian(self):
        """Assemble the channels into a list of Hamiltonian terms and
        collapse operators which can be ingested by ``qutip.mesolve``.

        Returns:
            tuple[list, list, np.ndarray]: H, C_ops, times
        """
        H = []
        for channel in self.channels.values():
            if "coeffs" in channel:
                H.append([channel["H"], channel["coeffs"]])
            elif isinstance(channel["H"], (list, tuple)):
                H.extend(channel["H"])
            else:
                H.append(channel["H"])
        C_ops = []
        for channel in self.collapse_channels.items():
            if "coeffs" in channel:
                C_ops.append([channel["op"], channel["coeffs"]])
            elif isinstance(channel["op"], (list, tuple)):
                C_ops.extend(channel["op"])
            else:
                C_ops.append(channel["op"])
        return H, C_ops, self.times

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
        import matplotlib.pyplot as plt

        channels = [
            name
            for name, ch in self.channels.items()
            if name != "delay" and "coeffs" in ch
        ]
        xs = self.times
        if subplots:
            fig, axes = plt.subplots(len(channels), 1, sharex=True)
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
            for i, (ch, ax) in enumerate(zip(channels, axes)):
                c = "C" + str(i)
                ys = self.channels[ch]["coeffs"]
                plot_func = ax.step if step else ax.plot
                plot_func(xs, ys.real, "-", label=ch, color=c)
                if plot_imag:
                    plot_func(xs, ys.imag, "--", color=c)
                ax.grid(True)
                ax.legend(loc=0)
            axes[0].set_title("Hamiltonian coefficients")
            axes[-1].set_xlabel("Time")
        else:
            fig, axes = plt.subplots(1, 1)
            plot_func = axes.step if step else axes.plot
            for i, ch in enumerate(channels):
                c = "C" + str(i)
                ys = self.channels[ch]["coeffs"]
                plot_func(xs, ys.real, "-", label=ch, color=c)
                if plot_imag:
                    plot_func(xs, ys.imag, "--", color=c)
            axes.set_xlabel("Time")
            axes.set_ylabel("Hamiltonian coefficient")
            axes.legend(loc=0)
            axes.grid(True)
        return fig, axes


class CompiledPulseSequence(object):
    """Creates time-dependent Hamiltonian channels
    from a sequence of Operations.

    Args:
        system (optional, subsystems.System): System containing
            the Modes on which to operate.
        channels (opional, dict): Dict of initial channels to pass the
            HamiltonianChannels. Default: None.
    """

    def __init__(self, system=None, channels=None, t0=0):
        self.system = None
        self.modes = None
        self.hc = None
        self._t = None
        if system is not None:
            self.set_system(system, channels=channels, t0=t0)

    def set_system(self, system, channels=None, t0=0):
        self.system = system
        self.modes = system.modes
        self.hc = HamiltonianChannels(channels=channels, t0=t0)
        self._t = self.hc.tmax

    @property
    def times(self):
        return self.hc.times

    @property
    def channels(self):
        return self.hc.channels

    def add_channel(
        self, name, H=None, C_op=None, time_dependent=False, error_if_exists=True
    ):
        """Add a channel with specified Hamiltonian term H
        or collapse operator C_op.

        Args:
            name (str): Name of the Hamiltonian channel.
            H (optional, qutip.Qobj): Hamiltonian term for this channel.
            C_op (optional, qutip.Qobj): Collapse operator for this channel.
            time_dependent (optional, bool): Whether this channel has
                time-dependent Hamiltonian coefficients. Default: False.
            error_if_exists (optional, bool): Whether to raise an exception
                if a channel with the same name already exists. Default: True.
        """
        self.hc.add_channel(
            name,
            H=H,
            C_op=C_op,
            time_dependent=time_dependent,
            error_if_exists=error_if_exists,
        )

    def add_operation(self, operation):
        """Add an Operation (time dependent Hamiltonian or collapse terms)
        to the HamiltonianChannels at the current time.

        Args:
            operation (Operation): Operation object to add
                to the pulse sequence.
        """
        from .common import HTerm, Operation

        if not isinstance(operation, Operation):
            raise TypeError(f"Expected an Operation, but got {type(operation)}.")
        duration, terms = operation
        for channel_name, term in terms.items():
            if isinstance(term, HTerm):
                H, coeffs, args, kwargs = term
                C_op = None
            else:
                # its a CTerm
                C_op, coeffs, args, kwargs = term
                H = None
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            self.hc.add_operation(
                channel_name,
                t0=self._t,
                duration=duration,
                H=H,
                C_op=C_op,
                coeffs=coeffs,
                coeffs_args=args,
                coeffs_kwargs=kwargs,
            )

    def delay(self, length, sync_before=True, sync_after=True):
        """Add a global delay to the sequence.

        Args:
            length (int): Length of the delay in ns.
            sync_before (optional, bool): Whether to sync before the delay.
                Default: True.
            sync_after (optional, bool): Whether to sync after the delay:
                Default: True.
        """
        if not length:
            if sync_before:
                self.sync()
            return
        if sync_before:
            self.sync()
        self.hc.add_operation(
            "delay",
            H=0 * self.system.active_modes[0].I,
            t0=self._t,
            duration=length,
            coeffs=1,
        )
        if sync_after:
            self.sync()

    def sync(self):
        """Ensure that the Hamiltonian channels all align up to this point.

        This means that all operations which follow the sync will be
        executed after all those before the sync. Sequences are constructed
        in terms of blocks of operations separated by sync()s.
        Within a block, channels are made to have equal duration by padding
        shorter channels to the maximum channel length.
        """
        self._t = self.hc.tmax + self.hc.dt

    def build_hamiltonian(self):
        """Assemble the channels into a list of Hamiltonian terms and
        collapse operators which can be ingested by qutip.mesolve.

        Returns:
            tuple[list, list, np.ndarray]: H, C_ops, times
        """
        return self.hc.build_hamiltonian()

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
            ``qutip.solver.Result``: qutip.Solver.Result instance.
        """
        from .common import ops2dms

        if c_ops is None:
            c_ops = []
        if e_ops is None:
            e_ops = []
        if options is None:
            options = qutip.Options()
            options.max_step = self.hc.dt
            options.store_states = True
        if "H0" not in self.hc.channels:
            H0 = sum(self.system.H0())
            if isinstance(H0, int) and H0 == 0:
                H0 = 0 * self.system.I()
            self.hc.add_channel("H0", H=H0, time_dependent=False)
        system_c_ops = self.system.c_ops()
        c_ops.extend(system_c_ops)

        H, C_ops, times = self.build_hamiltonian()
        c_ops.extend(C_ops)
        e_ops = ops2dms(e_ops)

        return qutip.mesolve(
            H,
            init_state,
            times,
            c_ops,
            e_ops,
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
        if c_ops is None:
            c_ops = []
        if options is None:
            options = qutip.Options()
            options.max_step = self.hc.dt
        if "H0" not in self.hc.channels:
            H0 = sum(self.system.H0())
            if isinstance(H0, int) and H0 == 0:
                H0 = 0 * self.system.I()
            self.hc.add_channel("H0", H=H0, time_dependent=False)
        system_c_ops = self.system.c_ops()
        c_ops.extend(system_c_ops)

        H, C_ops, times = self.build_hamiltonian()
        c_ops.extend(C_ops)
        props = qutip.propagator(
            H,
            times,
            c_op_list=c_ops,
            options=options,
            progress_bar=progress_bar,
        )
        # It seems that newer versions of numpy automatically coerce
        # objects of type np.ndarray[qutip.Qobj] into bare np.ndarray,
        # which is not what we want in this situation.
        if not all(isinstance(prop, qutip.Qobj) for prop in props):
            # Prevent numpy from coercing all of the Qobj into ndarray
            # by first creating an ndarray with dtype object,
            # then populating it with our Qobj.
            eye = self.system.I()
            arr = np.ndarray(len(props), dtype=object)
            arr[:] = [qutip.Qobj(prop, dims=eye.dims) for prop in props]
            props = arr
        return props

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
        return self.hc.plot_coefficients(
            subplots=subplots, plot_imag=plot_imag, step=step
        )
