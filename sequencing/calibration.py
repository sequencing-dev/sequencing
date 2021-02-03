# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
import lmfit
from .sequencing import get_sequence, sync, ket2dm, ops2dms, tqdm


def fit_line(xs, ys):
    model = lmfit.models.LinearModel()
    return model.fit(ys, x=xs)


def fit_sine(xs, ys):
    def sine(xs, amp=1, f0=1, phi=0, ofs=0.5):
        return ofs + amp * np.sin(2 * np.pi * f0 * xs + phi)

    ys = np.asarray(ys)
    # extract a guess for initial guess for f0 and phi
    num_pts = 1000
    fs = np.fft.rfftfreq(num_pts, xs[1] - xs[0])
    fft = np.fft.rfft(ys - ys.mean(), num_pts)
    ix = np.argmax(np.abs(fft))
    f0 = fs[ix]
    phi0 = 2 * np.pi * f0 * xs[0]
    phi = np.angle(fft[ix]) - phi0
    phi = (phi + np.pi) % (2 * np.pi) - np.pi / 2

    model = lmfit.Model(sine)
    model.set_param_hint("f0", value=f0, min=fs.min(), max=fs.max())
    model.set_param_hint("ofs", value=ys.mean(), min=ys.min(), max=ys.max())
    model.set_param_hint("amp", value=np.ptp(ys) / 2, min=0, max=np.ptp(ys))
    model.set_param_hint("phi", value=phi, min=-2 * np.pi, max=2 * np.pi)
    return model.fit(ys, xs=xs)


def fit_displacement(xs, ys):
    def displacement(xs, xscale=1.0, amp=1, ofs=0, n=0):
        alphas = xs * xscale
        nbars = alphas ** 2
        return ofs + amp * nbars ** n / factorial(int(n)) * np.exp(-nbars)

    if xs[-1] > xs[0]:
        amp = ys[0] - ys[-1]
        ofs = np.min(ys)
    else:
        amp = ys[-1] - ys[0]
        ofs = np.max(ys)

    model = lmfit.Model(displacement)
    model.set_param_hint("xscale", value=1)
    model.set_param_hint("amp", value=amp)
    model.set_param_hint("ofs", value=ofs)
    model.set_param_hint("n", value=0)

    return model.fit(ys, xs=xs)


def tune_rabi(
    system,
    init_state,
    e_ops=None,
    mode_name="qubit",
    pulse_name=None,
    amp_range=(-2, 2, 51),
    progbar=True,
    plot=True,
    ax=None,
    ylabel=None,
    update=True,
    verify=True,
):
    """Tune the amplitude of a Transmon pulse using
    an amplitude-Rabi experiment.

    Args:
        system (System): System containing the Transmon whose
            pulse you want to tune.
        init_state (qutip.Qobj): Initial state of the system.
        e_ops (optional, list[qutip.Qobj]): List of expectation
            operators. If none, defaults to init_state. Default: None.
        mode_name (optional, str): Name of the Transmon mode. Default: 'qubit'.
        pulse_name (optional, str): Name of the pulse to tune. If None,
            will use transmon.default_pulse. Default: None.
        amp_range (optional, tuple[float, float, int]): Range over which to
            sweep the pulse amplitude, specified by (start, stop, num_steps).
            The units are such that, if the pulse is tuned up, amplitude of 1
            generates a rotation by pi. Default: (-2, 2, 51).
        progbar (optional, bool): Whether to display a tqdm progress bar.
            Default: True.
        plot (optional, bool): Whether to plot the results: Default: True.
        ax (optional, matplotlib axis): Axis on which to plot results. If None,
            one is automatically created. Default: None.
        ylabel (optional, str): ylabel for the plot. Default: None.
        update (optional, bool): Whether to update the pulse amplitude based on
            the fit result. Default: True.
        verify (optional, bool): Whether to re-run the Rabi sequence with the
            newly-determined amplitude to verify correctness. Default: True.

    Returns:
        tuple[tuple, float, float]: (fig, ax), old_amp, new_amp
    """
    init_state = ket2dm(init_state)
    qubit = system.get_mode(mode_name)
    pulse_name = pulse_name or qubit.default_pulse
    pulse = getattr(qubit, pulse_name)

    if e_ops is None:
        e_ops = [init_state]
    e_ops = ops2dms(e_ops)
    e_pop = []
    amps = np.linspace(*amp_range)
    prog = tqdm if progbar else lambda x, **kwargs: x
    for amp in prog(amps):
        seq = get_sequence(system)
        with qubit.pulse_scale(amp, pulse_name=pulse_name):
            qubit.rotate_x(np.pi, pulse_name=pulse_name)
        result = seq.run(init_state, e_ops=e_ops)
        e_pop.append(result.expect[0][-1])

    fit_result = fit_sine(amps, e_pop)
    amp_scale = 1 / (2 * fit_result.params["f0"])

    old_amp = pulse.amp
    new_amp = amp_scale * old_amp

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()
        ax.plot(amps, e_pop, "o")
        ax.plot(amps, fit_result.best_fit, "-")
        ax.set_xlabel("Pulse scale")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_title(f"{pulse.name} Rabi")
        plt.pause(0.1)
    else:
        fig = None
        ax = None
    if update:
        pulse.amp = new_amp
        print(
            f"Updating {qubit.name} unit amp from " f"{old_amp:.2e} to {new_amp:.2e}.",
            flush=True,
        )
    if verify:
        _ = tune_rabi(
            system,
            init_state,
            e_ops=e_ops,
            mode_name=mode_name,
            pulse_name=pulse_name,
            amp_range=amp_range,
            progbar=progbar,
            plot=True,
            ax=ax,
            update=False,
            verify=False,
        )
    return (fig, ax), old_amp, new_amp


def tune_drag(
    system,
    init_state,
    e_ops=None,
    mode_name="qubit",
    pulse_name=None,
    drag_range=(-5, 5, 21),
    progbar=True,
    plot=True,
    ax=None,
    ylabel=None,
    update=True,
):
    """Tune the DRAG coefficient for a Transmon pulse by executing
    Rx(pi) - Ry(pi/2) and Ry(pi) - Rx(pi/2) using different DRAG
    values.

    Args:
        system (System): System containing the Transmon whose
            pulse you want to tune.
        init_state (qutip.Qobj): Initial state of the system.
        e_ops (optional, list[qutip.Qobj]): List of expectation
            operators. If None, defaults to init_state. Default: None.
        mode_name (optional, str): Name of the Cavity mode. Default: 'qubit'.
        pulse_name (optional, str): Name of the pulse to tune. If None,
            will use qubit.default_pulse. Default: None.
        drag_range (optional, tuple[float, float, int]): Range over which to
            sweep the DRAG value, specified by (start, stop, num_steps).
            Default: (-5, 5, 21)
        progbar (optional, bool): Whether to display a tqdm progress bar.
            Default: True.
        plot (optional, bool): Whether to plot the results: Default: True.
        ax (optional, matplotlib axis): Axis on which to plot results. If None,
            one is automatically created. Default: None.
        ylabel (optional, str): ylabel for the plot. Default: None.
        update (optional, bool): Whether to update the DRAG value based on
            the fit result. Default: True.

    Returns:
        tuple[tuple, float, float]: (fig, ax), old_drag, new_drag
    """
    init_state = ket2dm(init_state)
    qubit = system.get_mode(mode_name)
    pulse_name = pulse_name or qubit.default_pulse
    pulse = getattr(qubit, pulse_name)

    if e_ops is None:
        e_ops = [init_state]
    e_ops = ops2dms(e_ops)

    XpY9 = []
    YpX9 = []

    old_drag = pulse.drag
    drags = np.linspace(*drag_range)
    progbar = tqdm if progbar else lambda x, **kw: x
    for drag in progbar(drags):
        pulse.drag = drag
        seq = get_sequence(system)
        qubit.rotate_x(np.pi, pulse_name=pulse_name)
        sync()
        qubit.rotate_y(np.pi / 2, pulse_name=pulse_name)
        result = seq.run(init_state, e_ops=e_ops)
        XpY9.append(result.expect[0][-1])

        seq = get_sequence(system)
        qubit.rotate_y(np.pi, pulse_name=pulse_name)
        sync()
        qubit.rotate_x(np.pi / 2, pulse_name=pulse_name)
        result = seq.run(init_state, e_ops=e_ops)
        YpX9.append(result.expect[0][-1])

    r0 = fit_line(drags, XpY9)
    r1 = fit_line(drags, YpX9)
    b0, b1 = [r.params["intercept"].value for r in [r0, r1]]
    m0, m1 = [r.params["slope"].value for r in [r0, r1]]

    xopt = (b1 - b0) / (m0 - m1)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()
        ax.plot(drags, XpY9, "o", label="Rx(pi) - Ry(pi/2)")
        ax.plot(drags, r0.best_fit, "k-")
        ax.plot(drags, YpX9, "o", label="Ry(pi) - Rx(pi/2)")
        ax.plot(drags, r1.best_fit, "k-")
        ax.axvline(xopt, color="k", ls="--", label=f"DRAG: {xopt:.2e}")
        ax.legend(loc=0)
        ax.grid(True)
        ax.set_xlabel("DRAG coefficient")
        if ylabel:
            ax.set_ylabel("|e> population")
        ax.set_title(f"{pulse.name} DRAG")
        plt.pause(0.1)
    else:
        fig = None
        ax = None
    if update:
        pulse.drag = xopt
        print(
            f"Updating {pulse.name}.drag from " f"{old_drag:.2e} to {xopt:.2e}.",
            flush=True,
        )
    else:
        pulse.drag = old_drag
    return (fig, ax), old_drag, xopt


def tune_displacement(
    system,
    init_state,
    e_ops=None,
    mode_name="cavity",
    amp_range=(0.1, 3, 51),
    progbar=True,
    plot=True,
    ax=None,
    ylabel=None,
    update=True,
    verify=True,
):
    """Tune the amplitude of a Cavity pulse using
    a displacement experiment.

    Args:
        system (System): System containing the Cavity whose
            pulse you want to tune.
        init_state (qutip.Qobj): Initial state of the system.
        e_ops (optional, list[qutip.Qobj]): List of expectation
            operators. If None, defaults to init_state. Default: None.
        mode_name (optional, str): Name of the Cavity mode. Default: 'cavity'.
        pulse_name (optional, str): Name of the pulse to tune. If None,
            will use cavity.default_pulse. Default: None.
        amp_range (optional, tuple[float, float, int]): Range over which to
            sweep the pulse amplitude, specified by (start, stop, num_steps).
            The units are such that, if the pulse is tuned up, amplitude of 1
            generates a displacement of alpha = 1. Default: (0.1, 3, 51).
        progbar (optional, bool): Whether to display a tqdm progress bar.
            Default: True.
        plot (optional, bool): Whether to plot the results: Default: True.
        ax (optional, matplotlib axis): Axis on which to plot results. If None,
            one is automatically created. Default: None.
        ylabel (optional, str): ylabel for the plot. Default: None.
        update (optional, bool): Whether to update the pulse amplitude based on
            the fit result. Default: True.
        verify (optional, bool): Whether to re-run the sequence with the
            newly-determined amplitude to verify correctness. Default: True.

    Returns:
        tuple[tuple, float, float]: (fig, ax), old_amp, new_amp
    """
    init_state = ket2dm(init_state)
    cavity = system.get_mode(mode_name)
    pulse = cavity.gaussian_pulse
    if e_ops is None:
        e_ops = [init_state]
    e_ops = ops2dms(e_ops)
    zero_pop = []
    amps = np.linspace(*amp_range)
    prog = tqdm if progbar else lambda x, **kwargs: x
    for amp in prog(amps):
        seq = get_sequence(system)
        with cavity.pulse_scale(amp):
            cavity.displace(1)
        result = seq.run(init_state, e_ops=e_ops)
        zero_pop.append(result.expect[0][-1])

    fit_result = fit_displacement(amps, zero_pop)
    amp_scale = 1 / fit_result.params["xscale"].value

    old_amp = pulse.amp
    new_amp = amp_scale * old_amp

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()
        ax.plot(amps, zero_pop, "o")
        ax.plot(amps, fit_result.best_fit, "-")
        ax.set_xlabel("Pulse scale")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_title(f"{pulse.name} displacement")
        plt.pause(0.1)
    else:
        fig = None
        ax = None
    if update:
        pulse.amp = new_amp
        print(
            f"Updating {cavity.name} unit amp from " f"{old_amp:.2e} to {new_amp:.2e}.",
            flush=True,
        )
    if verify:
        _ = tune_displacement(
            system,
            init_state,
            e_ops=e_ops,
            mode_name=mode_name,
            amp_range=amp_range,
            progbar=progbar,
            plot=True,
            ax=ax,
            update=False,
            verify=False,
        )
    return (fig, ax), old_amp, new_amp
