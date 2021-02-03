# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import re
import inspect
from functools import lru_cache

import attr
import numpy as np
from scipy.integrate import quad
from colorednoise import powerlaw_psd_gaussian


from .parameters import (
    Parameterized,
    StringParameter,
    IntParameter,
    FloatParameter,
    NanosecondParameter,
    GigahertzParameter,
    RadianParameter,
    BoolParameter,
)


def array_pulse(
    i_wave,
    q_wave=None,
    amp=1,
    phase=0,
    detune=0,
    noise_sigma=0,
    noise_alpha=0,
    scale_noise=False,
):
    """Takes a real or complex waveform and applies an amplitude
    scaling, phase offset, time-dependent phase, and additive Gaussian noise.

    Args:
        i_wave (array-like): Real part of the waveform.
        q_wave (optional, array-like): Imaginary part of the waveform.
            If None, the imaginary part is set to 0. Default: None.
        amp (float): Factor by which to scale the waveform amplitude.
            Default: 1.
        phase (optionla, float): Phase offset in radians. Default: 0.
        detune (optional, float): Software detuning/time-dependent phase to
            apply to the waveform, in GHz. Default: 0.
        noise_sigma (optional, float): Standard deviation of additive Gaussian
            noise applied to the pulse (see scale_noise).
            Default: 0.
        noise_alpha (optional, float): Exponent for the noise PSD S(f).
            S(f) is proportional to (1/f)**noise_alpha.
            noise_alpha = 0 for white noise, noise_alpha = 1 for 1/f noise,
            etc. Default: 0 (white noise).
        scale_noise (optional, bool): Whether to scale the noise by ``amp``
            before adding it to the signal. If False, then noise_sigma has
            units of GHz. Default: False.

    Returns:
        ``np.ndarray``: Complex waveform.
    """
    i_wave = np.asarray(i_wave)
    if q_wave is None:
        q_wave = np.zeros_like(i_wave)
    if detune:
        ts = np.arange(len(i_wave))
        c_wave = (i_wave + 1j * q_wave) * np.exp(-2j * np.pi * ts * detune)
        i_wave, q_wave = c_wave.real, c_wave.imag
    if noise_sigma:
        i_noise, q_noise = noise_sigma * powerlaw_psd_gaussian(
            noise_alpha, [2, i_wave.size]
        )
    else:
        i_noise, q_noise = 0, 0
    if scale_noise:
        i_wave = amp * (i_wave + i_noise)
        q_wave = amp * (q_wave + q_noise)
    else:
        i_wave = amp * i_wave + i_noise
        q_wave = amp * q_wave + q_noise
    c_wave = (i_wave + 1j * q_wave) * np.exp(-1j * phase)
    return c_wave


def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop // 2 * sigma, chop // 2 * sigma, int(chop * sigma // 4) * 4)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)


def gaussian_deriv_wave(sigma, chop=4):
    ts = np.linspace(-chop // 2 * sigma, chop // 2 * sigma, int(chop * sigma // 4) * 4)
    ofs = np.exp(-ts[0] ** 2 / (2 * sigma ** 2))
    return (0.25 / sigma ** 2) * ts * np.exp(-(ts ** 2) / (2 * sigma ** 2)) / (1 - ofs)


def gaussian_chop(t, sigma, t0):
    P = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    ofs = np.exp(-(t0 ** 2) / (2.0 * sigma ** 2))
    return (P - ofs) / (1 - ofs)


@lru_cache()
def gaussian_chop_norm(sigma, chop):
    t0 = sigma * chop / 2
    norm, _ = quad(gaussian_chop, -t0, t0, args=(sigma, -t0))
    return 2 * norm


def ring_up_wave(length, reverse=False, shape="tanh"):
    if shape == "cos":
        wave = ring_up_cos(length)
    elif shape == "tanh":
        wave = ring_up_tanh(length)
    else:
        raise ValueError(f"Shape must be 'cos' or 'tanh', not {shape}.")
    if reverse:
        wave = wave[::-1]
    return wave


def ring_up_cos(length):
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, length)))


def ring_up_tanh(length):
    ts = np.linspace(-2, 2, length)
    return (1 + np.tanh(ts)) / 2


def smoothed_constant_wave(length, sigma, shape="tanh"):
    if sigma == 0:
        return np.ones(length)
    ring_up = ring_up_wave(sigma, shape=shape)
    constant = np.ones(int(length - 2 * sigma))
    ring_down = ring_up[::-1]
    return np.concatenate([ring_up, constant, ring_down])


def constant_pulse(length=None):
    i_wave, q_wave = np.ones(length), np.zeros(length)
    return i_wave, q_wave


def gaussian_pulse(sigma=None, chop=4, drag=0):
    i_wave = gaussian_wave(sigma, chop=chop)
    q_wave = drag * gaussian_deriv_wave(sigma, chop=chop)
    return i_wave, q_wave


def smoothed_constant_pulse(length=None, sigma=None, shape="tanh"):
    i_wave = smoothed_constant_wave(length, sigma, shape=shape)
    q_wave = np.zeros_like(i_wave)
    return i_wave, q_wave


def sech_wave(sigma, chop=4):
    # https://arxiv.org/pdf/1704.00803.pdf
    # https://doi.org/10.1103/PhysRevA.96.042339
    rho = np.pi / (2 * sigma)
    t0 = chop * sigma // 2
    ts = np.linspace(-t0, t0, int(chop * sigma // 4) * 4)
    P = 1 / np.cosh(rho * ts)
    ofs = P[0]
    return (P - ofs) / (1 - ofs)


def sech_deriv_wave(sigma, chop=4):
    rho = np.pi / (2 * sigma)
    t0 = chop * sigma // 2
    ts = np.linspace(-t0, t0, int(chop * sigma // 4) * 4)
    ofs = 1 / np.cosh(rho * ts[0])
    P = -np.sinh(rho * ts) / np.cosh(rho * ts) ** 2
    return (P - ofs) / (1 - ofs)


def sech_pulse(sigma=None, chop=4, drag=0):
    i_wave = sech_wave(sigma, chop=chop)
    # q_wave = drag * sech_deriv_wave(sigma, chop=chop)
    q_wave = drag * np.gradient(i_wave)
    return i_wave, q_wave


def slepian_pulse(tau=None, width=10, drag=0):
    # bandwidth is relative, i.e. scaled by 1/tau
    from scipy.signal.windows import slepian

    i_wave = slepian(tau, width / tau)
    q_wave = drag * np.gradient(i_wave)
    return i_wave, q_wave


@attr.s
class Pulse(Parameterized):
    """Generates a parameterized complex pulse waveform
    using callable ``pulse_func``.

    Args:
        amp (float): Maximum amplitude of the pulse. Default: 1.
        detune (float): "Software detuning" (time-dependent phase)
            to apply to the waveform, in GHz. Default: 0.
        phase (float): Phase offset to apply to the waveform,
            in radians. Default: 0.
        noise_sigma (float): Standard deviation of additive Gaussian noise
            applied to the pulse (in the same units as ``amp``).
            Default: 0.
        noise_alpha (float): Exponent for the noise PSD S(f).
            S(f) is proportional to (1/f)**noise_alpha.
            noise_alpha = 0 for white noise, noise_alpha = 1 for 1/f noise,
            etc. Default: 0 (white noise).
        scale_noise (optional, bool): Whether to scale the noise by ``amp``
            before adding it to the signal. If False, then noise_sigma has
            units of GHz. Default: True.
    """

    pulse_func = staticmethod(constant_pulse)
    amp = FloatParameter(1)
    detune = GigahertzParameter(0)
    phase = RadianParameter(0)
    noise_sigma = FloatParameter(0)
    noise_alpha = FloatParameter(0)
    scale_noise = BoolParameter(False)

    def __call__(self, **kwargs):
        """Returns the Pulse's complex waveform.

        Keyword arguments are passed to either ``pulse_func`` or
        ``array_pulse``, or used to override the pulse's parameters.

        Returns:
            ``np.ndarray``: complex waveform
        """
        pulse_kwargs = {}
        pulse_arg_names = inspect.signature(self.pulse_func).parameters
        array_pulse_kwargs = {}
        array_pulse_arg_names = inspect.signature(array_pulse).parameters
        # first populate pulse kwargs with values from Parameters
        for name, value in self.as_dict().items():
            if name in pulse_arg_names:
                pulse_kwargs[name] = value
            elif name in array_pulse_arg_names:
                array_pulse_kwargs[name] = value
        for name in list(kwargs):
            # populate array_pulse kwargs
            if name in array_pulse_arg_names:
                array_pulse_kwargs[name] = kwargs.pop(name)
            # override pulse kwargs from Parameters with those from kwargs
            elif name in pulse_arg_names:
                pulse_kwargs[name] = kwargs.pop(name)
        waves = self.pulse_func(**pulse_kwargs)
        if len(waves) == 2:
            i_wave, q_wave = waves
        else:
            i_wave, q_wave = waves, None
        return array_pulse(i_wave, q_wave=q_wave, **array_pulse_kwargs)

    def plot(self, ax=None, grid=True, legend=True, **kwargs):
        """Plots the waveform and returns the Axes.

        Keyword arguments are passed to ``__call__()``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        c_wave = self(**kwargs)
        (line,) = ax.plot(c_wave.real, ls="-", label=self.name)
        ax.plot(c_wave.imag, color=line._color, ls="--")
        ax.grid(grid)
        if legend:
            ax.legend(loc="best")
        return ax


@attr.s
class ConstantPulse(Pulse):
    """A constant (rectangular) pulse.

    Args:
        amp (float): Maximum amplitude of the pulse. Default: 1.
        detune (float): "Software detuning" (time-dependent phase)
            to apply to the waveform, in GHz. Default: 0.
        phase (float): Phase offset to apply to the waveform,
            in radians. Default: 0.
    """

    pass


@attr.s
class SmoothedConstantPulse(Pulse):
    """A constant pulse with smoothed ring-up and ring-down.

    Args:
        amp (float): Maximum amplitude of the pulse. Default: 1.
        detune (float): "Software detuning" (time-dependent phase)
            to apply to the waveform, in GHz. Default: 0.
        phase (float): Phase offset to apply to the waveform,
            in radians. Default: 0.
        length (int): Total length of the pulse in ns. Default: 100.
        sigma (int): Ring-up and ring-down time in ns. If sigma == 0, then
            this is equivalent to ControlPulse. The length of the contant
            portion of the pulse is ``length - 2 * sigma``. Default: 0.
        shape (str): String specifying the type of ring-up and ring-down.
            Calid options are 'tanh' and 'cos' (see ``ringup_wave``).
            Default: 'tanh'.
    """

    VALID_SHAPES = ["tanh", "cos"]
    pulse_func = staticmethod(smoothed_constant_pulse)
    length = NanosecondParameter(100)
    sigma = NanosecondParameter(0)
    shape = StringParameter("tanh", validator=attr.validators.in_(VALID_SHAPES))


@attr.s
class GaussianPulse(Pulse):
    """A Gaussian that is "chopped" at
    +/- ``(chop / 2) * sigma``. The full
    pulse length is therefore ``sigma * chop``.

    Args:
        amp (float): Maximum amplitude of the pulse. Default: 1.
        detune (float): "Software detuning" (time-dependent phase)
            to apply to the waveform, in GHz. Default: 0.
        phase (float): Phase offset to apply to the waveform,
            in radians. Default: 0.
        sigma (float): Gaussian sigma in ns. Default: 10.
        chop (int): The Gaussian is truncated at
            +/- ``chop/2 * sigma``. Default: 4.
        drag (float): DRAG coefficient. Default: 0.
    """

    pulse_func = staticmethod(gaussian_pulse)
    sigma = NanosecondParameter(10)
    chop = IntParameter(4, unit="sigma")
    drag = FloatParameter(0)


@attr.s
class SechPulse(Pulse):
    r"""Hyperbolic secant pulse that is "chopped" at
    +/- ``(chop / 2) * sigma``.

    .. math::
        A(t) &= \text{sech}(\rho t)\\
        \rho &= \pi / (2\sigma)

    See: https://doi.org/10.1103/PhysRevA.96.042339

    Args:
        sigma (int): Pulse "sigma" in ns (see equation above).
            Default: 10.
        chop (int): The waveform is truncated at
            +/- ``chop/2 * sigma``. Default: 4.
        drag (float): DRAG coefficient:
            imag(wave) = drag * d/dt real(wave). Default: 0.
    """
    pulse_func = staticmethod(sech_pulse)
    sigma = NanosecondParameter(10)
    chop = IntParameter(4, unit="sigma")
    drag = FloatParameter(0)


@attr.s
class SlepianPulse(Pulse):
    """A Slepian Pulse.

    See ``scipy.signal.windows.slepian``.

    Args:
        tau (int): Pulse length in ns. Default: 40.
        width (int): Pulse width in ns
            (similar to a Gaussian sigma). Default: 10.
        drag (float): DRAG coefficient:
            imag(wave) = drag * d/dt real(wave). Default: 0.
    """

    pulse_func = staticmethod(slepian_pulse)
    tau = NanosecondParameter(40)
    width = NanosecondParameter(10)
    drag = FloatParameter(0)


def pulse_factory(cls, name=None, **kwargs):
    """Returns a function that creates an instance
    if the given pulse class.

    Keyword arguments are passed to ``cls.__init__()``.

    Args:
        cls (type): Subclass of Pulse of which to create an instance.
        name (optional, str): Name of the resulting pulse. If None,
            will use a snake-case version of the class name,
            e.g. 'GaussianPulse' -> 'gaussian_pulse'. Default: None.

    Returns:
        callable: A function that takes no arguments and returns
        an instance of ``cls``.
    """
    if name is None:
        # turn 'GaussianPulse' into 'gaussian_pulse'
        name = "_".join(re.findall("[a-zA-Z][^A-Z]*", cls.__name__)).lower()
    return lambda: cls(name=name, **kwargs)
