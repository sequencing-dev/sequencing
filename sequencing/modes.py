# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

from asteval import make_symbol_table, Interpreter
from contextlib import contextmanager

import numpy as np
import attr
import qutip

from .parameters import (
    Parameterized,
    StringParameter,
    IntParameter,
    FloatParameter,
    DictParameter,
    NanosecondParameter,
    GigahertzParameter,
)
from .pulses import (
    pulse_factory,
    SmoothedConstantPulse,
    GaussianPulse,
    gaussian_chop_norm,
)
from .sequencing import capture_operation, Operation, HTerm


def sort_modes(modes):
    """Sorts a list of ``Modes`` based on the logic described below.

    Mode ordering is decided primarily by mode Hilbert space size:
    modes with more levels go on the right.

    Among modes with the same number of levels,
    the ordering is decided alphanumerically from right to left.

    For example, assuming all cavities have the same number of levels
    and all qubits have the same, smaller, number of levels:
    [qubit1, qubit0, cavity2, cavity1, cavity0]

    Args:
        modes (list[Mode]): List of modes to sort.

    Returns:
        ``list[Mode]``: Sorted list of Modes.
    """
    return sorted(modes, key=lambda m: (m.name, m.levels), reverse=True)


@attr.s
class Mode(Parameterized):
    """An oscillator that can be embedded in a larger Hilbert space.

    ``Modes`` have a ``logical_zero`` state and ``logical_one state``,
    based on which Pauli operators and rotations are defined.

    Args:
        name (str): Name of the mode for which to construct operators.
        levels (optional, int): Number of levels in the mode subspace. Default: 2.
        t1 (optional, float): Mode T1 time in nanoseconds. Default: inf.
        t2 (optional, float): Mode T2 time in nanoseconds. Default: inf.
        thermal_population (optional, float): Mode excited state population
            (in [0..1]). Default: 0.
        df (optional, float): Mode detuning in GHz. Default: 0.
        kerr (optional, float): Mode self-Kerr in GHz. Default: 0.
    """

    levels = IntParameter(2)
    t1 = NanosecondParameter(np.inf, base=FloatParameter)
    t2 = NanosecondParameter(np.inf, base=FloatParameter)
    thermal_population = FloatParameter(0)
    df = GigahertzParameter(0)
    kerr = GigahertzParameter(0)

    order_modes = True

    OPERATORS = (
        "I",
        "a",
        "ad",
        "n",
        "x",
        "y",
        "sigmax",
        "sigmay",
        "sigmaz",
        "Raxis",
        "Rx",
        "Ry",
        "Rz",
        "Rphi",
    )

    def initialize(self):
        super().initialize()
        self._space = [self]
        self._logical_zero = None
        self._logical_one = None

    @property
    def space(self):
        """A list of Modes.

        This is the Hilbert space in which the mode exists.
        """
        return self._space

    @space.setter
    def space(self, modes):
        if self.order_modes:
            modes = sort_modes(modes)
        self._space = modes

    @property
    def index(self):
        """The index of the Mode in its Hilbert space."""
        return self.space.index(self)

    @property
    def I(self):  # noqa: E741, E743
        """Identity operator."""
        return self.tensor_with_I(qutip.qeye(self.levels))

    @property
    def a(self):
        """Annihilation operator."""
        return self.tensor_with_I(qutip.destroy(self.levels))

    @property
    def ad(self):
        """Creation operator."""
        return self.a.dag()

    @property
    def n(self):
        """Number operator."""
        a = self.a
        return a.dag() * a

    @property
    def x(self):
        """Position operator."""
        a = self.a
        return a.dag() + a

    @property
    def y(self):
        """Momentum (quadrature) operator."""
        a = self.a
        return -1j * (a - a.dag())

    @property
    def detuning(self):
        """Operator representation of a detuning on this mode."""
        return 2 * np.pi * self.df * self.n

    @property
    def self_kerr(self):
        """Operator representation of the mode's self-Kerr."""
        a = self.a
        ad = a.dag()
        return np.pi * self.kerr * (ad * ad * a * a)

    @property
    def tphi(self):
        """Pure dephasing time, calculated from ``t1`` and ``t2``."""
        if np.isinf(self.t2):
            return np.inf
        if self.t2 > 2 * self.t1:
            raise ValueError("Cannot have T2 > 2 * T1.")
        try:
            return 1 / (1 / self.t2 - 1 / (2 * self.t1))
        except ZeroDivisionError:  # t2 == 2 * t1
            return np.inf

    @property
    def Gamma_up(self):
        """Excitation rate, calulcated from ``t1`` and
        ``thermal_population``.
        """
        return self.thermal_population / self.t1

    @property
    def Gamma_down(self):
        """Decay rate, calculated from ``t1`` and ``Gamma_up``."""
        return 1 / self.t1 - self.Gamma_up

    @property
    def decay(self):
        """Collapse operator for energy decay."""
        return np.sqrt(self.Gamma_down) * self.a

    @property
    def excitation(self):
        """Collapse operator for excitation."""
        return np.sqrt(self.Gamma_up) * self.ad

    @property
    def dephasing(self):
        """Collapse operator for pure dephasing."""
        return np.sqrt(2 / self.tphi) * self.n

    @contextmanager
    def no_loss(self):
        """A context manager that temporarily sets a Mode's coherence
        parameters to the ideal values:
        T1 = inf, T2 = inf, thermal population = 0.
        """
        kwargs = dict(t1=np.inf, t2=np.inf, thermal_population=0)
        with self.temporarily_set(**kwargs):
            yield

    @contextmanager
    def use_space(self, modes):
        """A context manager that temporarily sets ``self.modes``
        to ``modes``, then reverts it to the previous value.

        Args:
            modes (list[Mode]): List of ``Modes`` specifying the
                temporary Hilbert space to use.
        """
        if isinstance(modes, Mode):
            modes = [modes]
        old_space = self.space
        try:
            self.space = modes
            yield
        finally:
            self.space = old_space

    def fock(self, n=0, full_space=True):
        """Returns the Fock state ``|n>``, optionally embedded in the
        full Hilbert space.

        Args:
            n (optional, int): Integer corresponding to desired Fock state.
                Default: 0
            full_space (optional, bool): Whether to return the basis state
                embedded in the full space. Default: True.

        Returns:
            ``qutip.Qobj``: Ket representing the Fock state.
        """
        if not full_space:
            return qutip.basis(self.levels, n)
        return self.tensor_with_zero(qutip.basis(self.levels, n))

    basis = fock

    def fock_dm(self, n=0, full_space=True):
        """Returns the basis state ``|n>``, optionally embedded in the
        full Hilbert space, as a density matrix.

        Args:
            n (optional, int): Integer corresponding to desired Fock state.
                Default: 0
            full_space (optional, bool): Whether to return the basis state
                embedded in the full space. Default: True.

        Returns:
            ``qutip.Qobj``: Density matrix representing the Fock product state.
        """
        ket = self.fock(n=n, full_space=full_space)
        return qutip.ket2dm(ket)

    @staticmethod
    def tensor(*args):
        """Calculates the tensor product of input operators."""
        return qutip.tensor(*args)

    def tensor_with_zero(self, state):
        """Returns ``state`` on ``self``, tensored with
        ``ket(0)`` on all other modes in ``self.space``.

        Args:
            state (qutip.Qobj): State of the mode.

        Returns:
            ``qutip.Qobj``: ``state`` on the mode, tensored with ``ket(0)``
            on all other modes in self.space.
        """
        states = [mode.basis(0, full_space=False) for mode in self.space]
        states[self.index] = state
        return self.tensor(*states)

    def tensor_with_I(self, operator):
        r"""Returns `operator` for ``self``, tensored with
        ``I`` for all other modes in ``self.space``.

        Args:
            state (qutip.Qobj): State of the mode.

        Returns:
            ``qutip.Qobj``: ``operator`` for the mode, tensored with ``I``
            on all other modes in ``self.space``.
        """
        ops = [qutip.qeye(mode.levels) for mode in self.space]
        ops[self.index] = operator
        return self.tensor(*ops)

    def set_logical_states(self, logical_zero=None, logical_one=None):
        """Sets the mode's logical zero and logical one states.

        Args:
            logical_zero (optional, qutip.Qobj): Ket representing
                the mode's logical zero state. If None, the Fock state ket(0)
                is used as logical zero. ``logical_zero`` must have the same
                dimensions as the mode itself. Default: None.
            logical_one (optional, qutip.Qobj): Ket representing
                the mode's logical one state. If None, the Fock state ket(1)
                is used as logical one. ``logical_one`` must ahve the same
                dimensions as the mode itself. Default: None.
        """
        dims = self.basis(0, full_space=False).dims
        if logical_zero is not None and logical_zero.dims != dims:
            raise ValueError(
                "Logical zero state must have the same dimension as the Mode."
            )
        if logical_one is not None and logical_one.dims != dims:
            raise ValueError(
                "Logical one state must have the same dimension as the Mode."
            )
        self._logical_zero = logical_zero
        self._logical_one = logical_one

    def logical_zero(self, full_space=True):
        """Returns the mode's logical zero state,
        optionally embedded in the full Hilbert space defined by
        ``self.space``.

        Args:
            full_space (optional, bool): Whether to embed
                ``logical_zero`` in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Ket representing the mode's logical zero state.
        """
        state = self._logical_zero
        if state is None:
            state = self.basis(0, full_space=False)
        if full_space:
            state = self.tensor_with_zero(state)
        return state

    def logical_one(self, full_space=True):
        r"""Returns the mode's logical one state,
        optionally embedded in the full Hilbert space defined by
        ``self.space``.

        Args:
            full_space (optional, bool): Whether to embed
                ``logical_one`` in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Ket representing the mode's logical one state.
        """
        state = self._logical_one
        if state is None:
            state = self.basis(1, full_space=False)
        if full_space:
            state = self.tensor_with_zero(state)
        return state

    def logical_states(self, full_space=True):
        r"""Returns the mode's logical zero and logical one states,
        optionally embedded in the full Hilbert space defined by
        ``self.space``.

        Args:
            full_space (optional, bool): Whether to embed
                logical states in the full Hilbert space.
                Default: True.

        Returns:
            tuple[qutip.Qobj, qutip.Qobj]: Kets representing
            the mode's logical zero and logical ones states.
        """
        zeroL = self.logical_zero(full_space=full_space)
        oneL = self.logical_one(full_space=full_space)
        return zeroL, oneL

    def sigmax(self, full_space=True):
        r"""Pauli X operator

        .. math::
            X = \left|0_L\rangle\langle1_L\right|
            + \left|1_L\rangle\langle0_L\right|

        Args:
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Pauli X operator.
        """
        zeroL, oneL = self.logical_states(full_space=False)
        op = zeroL * oneL.dag() + oneL * zeroL.dag()
        if full_space:
            op = self.tensor_with_I(op)
        return op

    def sigmay(self, full_space=True):
        r"""Pauli Y operator

        .. math::
            Y = -i\left(\left|0_L\rangle\langle1_L\right|
            - \left|1_L\rangle\langle0_L\right|\right)

        Args:
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Pauli Y operator.
        """
        zeroL, oneL = self.logical_states(full_space=False)
        op = -1j * (zeroL * oneL.dag() - oneL * zeroL.dag())
        if full_space:
            op = self.tensor_with_I(op)
        return op

    def sigmaz(self, full_space=True):
        r"""Pauli Z operator.

        .. math::
            Z = \left|0_L\rangle\langle0_L\right|
            - \left|1_L\rangle\langle1_L\right|

        Args:
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Pauli Z operator.
        """
        zeroL, oneL = self.logical_states(full_space=False)
        op = zeroL * zeroL.dag() - oneL * oneL.dag()
        if full_space:
            op = self.tensor_with_I(op)
        return op

    def Raxis(self, theta, phi, full_space=True):
        r"""Operator for a rotation of angle ``theta`` about an axis
        specified by ``phi``.

        .. math::
            R_\text{axis}(\theta,\phi)
            = \exp\left(-i\theta/2 \cdot (\cos(\phi)X + \sin(\phi)Y)\right)

        Args:
            theta (float): Rotation angle in radians.
            phi (float): Angle between the axis of rotation
                and the x axis.
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Raxis operator.
        """
        sx = self.sigmax(full_space=full_space)
        sy = self.sigmay(full_space=full_space)
        return (-1j * theta / 2 * (np.cos(phi) * sx + np.sin(phi) * sy)).expm()

    def Rx(self, theta, full_space=True):
        r"""Operator for a rotation of angle ``theta`` about the x axis.

        .. math::
            R_x(\theta) = \exp(-i\theta/2 \cdot X)

        Args:
            theta (float): Rotation angle in radians.
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Rx operator.
        """
        return (-1j * theta / 2 * self.sigmax(full_space=full_space)).expm()

    def Ry(self, theta, full_space=True):
        r"""Operator for a rotation of angle ``theta`` about the y axis.

        .. math::
            R_y(\theta) = \exp(-i\theta/2 \cdot Y)

        Args:
            theta (float): Rotation angle in radians.
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Ry operator.
        """
        return (-1j * theta / 2 * self.sigmay(full_space=full_space)).expm()

    def Rz(self, theta, full_space=True):
        r"""Operator for a rotation of angle ``theta`` about the z axis.

        .. math::
            R_z(\theta) = \exp(-i\theta/2 \cdot Z)

        Args:
            theta (float): Rotation angle in radians.
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Rz operator.
        """
        return (-1j * theta / 2 * self.sigmaz(full_space=full_space)).expm()

    def hadamard(self, full_space=True):
        r"""Hadamard operator.

        .. math::
            H = \frac{1}{\sqrt{2}}\left(X + Z\right)

        Args:
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Hadamard operator.
        """
        kwargs = dict(full_space=full_space)
        return 1 / np.sqrt(2) * (self.sigmax(**kwargs) + self.sigmaz(**kwargs))

    def Rphi(self, phi, full_space=True):
        r"""Phase shift operator.

        .. math::
            R_\phi = \exp(i\phi a^\dagger a)

        Args:
            phi (float): Number state-dependent phase shift
                to apply to the mode.
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Phase shift operator.
        """
        op = self.n
        if not full_space:
            with self.use_space(self):
                op = self.n
        return (1j * phi * op).expm()

    def operator_expr(self, expr):
        """Evaluate an expression composed of single-mode operators.

        See ``Mode.OPERATORS`` for the full list of supported operators.

        Args:
            expr (str): String representation of the operator expression
                to evaluate.

        Returns:
            ``qutip.Qobj``: Evaluated operator expression.
        """
        symbols = {name: getattr(self, name) for name in self.OPERATORS}
        symtable = make_symbol_table(use_numpy=True, **symbols)
        aeval = Interpreter(symtable=symtable)
        return aeval.eval(expr)


@attr.s
class PulseMode(Mode):
    """A Mode that has Pulses.

    Args:
        name (str): Name of the mode for which to construct operators.
        levels (optional, int): Number of levels in the mode subspace. Default: 2.
        t1 (optional, float): Mode T1 time in nanoseconds. Default: inf.
        t2 (optional, float): Mode T2 time in nanoseconds. Default: inf.
        thermal_population (optional, float): Mode excited state population
            (in [0..1]). Default: 0.
        df (optional, float): Mode detuning in GHz. Default: 0.
        kerr (optional, float): Mode self-Kerr in GHz. Default: 0.
        pulses (optional, dict[str, Pulse]): Dict of pulses defined for this Mode.
            Default: empty dict {}.
        default_pulse (optional, str): Name of the default Pulse for this Mode.
            Default: "gaussian_pulse".
    """

    pulses = DictParameter()
    default_pulse = StringParameter("gaussian_pulse")

    def initialize(self):
        super().initialize()
        self.add_pulse(cls=SmoothedConstantPulse)
        self.add_pulse(cls=GaussianPulse)

    def add_pulse(self, cls=GaussianPulse, name=None, error_if_exists=False, **kwargs):
        """Creates a new pulse of type ``cls`` and adds it to ``self.pulses``.

        Keyword arguments are passed to ``cls.__init__()``.

        Args:
            cls (optional, type): Pulse class to instantiate.
                Default: ``GaussianPulse``.
            name (optional, str): Name of the new pulse. If None,
                will use the "snake case" version of the class name,
                e.g. "GaussianPulse" -> "gaussian_pulse". Default: None.
            error_if_exists (optional, bool): Whether to raise an exception
                if a pulse with the same name already exists. Default: False.
        """
        pulse = pulse_factory(cls, name=name, **kwargs)()
        name = pulse.name
        if name in self.pulses and error_if_exists:
            raise ValueError(f"Pulse {name} already defined on {self.name}.")
        self.pulses[name] = pulse
        setattr(self, name, pulse)

    @contextmanager
    def amplitude(self, amp, pulse_name=None):
        """A context manager that temporarily sets the amplitude
        of a given pulse to ``amp``, then reverts it to its
        previous value.

        Args:
            amp (float): Amplitude to temporarilty set.
            pulse_name (optional, str): Name of the pulse whose
                amplitude you would like to set. If None,
                uses ``self.default_pulse``. Default: None.
        """
        if pulse_name is None:
            pulse_name = self.default_pulse
        pulse = self.pulses[pulse_name]
        old_amp = pulse.amp
        try:
            pulse.amp = amp
            yield
        finally:
            pulse.amp = old_amp

    @contextmanager
    def pulse_scale(self, scale, pulse_name=None):
        """A context manager that temporarily scales the amplitude
        of a given pulse by ``scale``, then reverts it to its
        previous value (i.e. ``scale = 1``).

        Args:
            scale (float): Factor by which to temporarilty
                scale the pulse amplitude.
            pulse_name (optional, str): Name of the pulse whose
                amplitude you would like to scale. If None,
                uses ``self.default_pulse``. Default: None.
        """
        if pulse_name is None:
            pulse_name = self.default_pulse
        pulse = self.pulses[pulse_name]
        amp = pulse.amp
        try:
            pulse.amp = amp * scale
            yield
        finally:
            pulse.amp = amp


@attr.s
class Transmon(PulseMode):
    """Fixed-frequency transmon.

    A PulseMode whose primary operations are rotations on the Bloch Sphere.

    Args:
        name (str): Name of the mode for which to construct operators.
        levels (optional, int): Number of levels in the mode subspace. Default: 2.
        t1 (optional, float): Mode T1 time in nanoseconds. Default: inf.
        t2 (optional, float): Mode T2 time in nanoseconds. Default: inf.
        thermal_population (optional, float): Mode excited state population
            (in [0..1]). Default: 0.
        df (optional, float): Mode detuning in GHz. Default: 0.
        kerr (optional, float): Mode self-Kerr in GHz. Default: 0.
        pulses (optional, dict[str, Pulse]): Dict of pulses defined for this Mode.
            Default: empty dict {}.
        default_pulse (optional, str): Name of the default Pulse for this Mode.
            Default: "gaussian_pulse".
    """

    @property
    def anharmonicity(self):
        """Alias for ``Transmon.kerr``"""
        return self.kerr

    @anharmonicity.setter
    def anharmonicity(self, alpha):
        self.kerr = alpha

    @capture_operation
    def rotate(self, angle, phase, pulse_name=None, **kwargs):
        """Generate a pulse to rotate the transmon by a given angle
        about a given axis.

        Keyword arguments are passed to the ``Pulse`` object.

        Args:
            angle (float): Rotation angle in radians.
            phase (float): Rotation axis relative to the x axis.
            pulse_name (optional, str): Name of the pulse to use. If None,
                will use ``self.default_pulse``. Default: None.

        Returns:
            Operation: The resulting ``Operation`` object.
        """
        pulse_name = pulse_name or self.default_pulse
        pulse = getattr(self, pulse_name)
        # Assuming that default_pulse.amp = 1 corresponds to rotation of pi
        norm = gaussian_chop_norm(pulse.sigma, pulse.chop)
        amp = angle * pulse.amp / norm
        c_wave = pulse(amp=amp, phase=phase, **kwargs)
        terms = {
            f"{self.name}.x": HTerm(self.x, c_wave.real),
            f"{self.name}.y": HTerm(self.y, c_wave.imag),
        }
        return Operation(len(c_wave), terms)

    def rotate_x(self, angle, unitary=False, **kwargs):
        """Generate a rotation about the x axis.

        Keyword arguments are passed to ``Transmon.rotate()`` if
        ``unitary`` is False.

        Args:
            angle (float): Rotation angle in radians.
            unitary (optional, bool): Whether to return the corresponding
                unitary operator instead of executing the pulse.
                Default: False.

        Returns:
            ``qutip.Qobj`` or Operation: If ``unitary`` is True, returns
            the unitary operator Rx(angle), otherwise returns the resulting
            ``Operation`` object.
        """
        if unitary:
            full_space = kwargs.get("full_space", True)
            return self.Rx(angle, full_space=full_space)
        return self.rotate(angle, 0, **kwargs)

    def rotate_y(self, angle, unitary=False, **kwargs):
        """Generate a rotation about the y axis.

        Keyword arguments are passed to ``Transmon.rotate()`` if
        ``unitary`` is False.

        Args:
            angle (float): Rotation angle in radians.
            unitary (optional, bool): Whether to return the corresponding
                unitary operator instead of executing the pulse.
                Default: False.

        Returns:
            ``qutip.Qobj`` or Operation: If ``unitary`` is True, returns
            the unitary operator Ry(angle), otherwise returns the resulting
            ``Operation`` object.
        """
        if unitary:
            full_space = kwargs.get("full_space", True)
            return self.Ry(angle, full_space=full_space)
        return self.rotate(-angle, np.pi / 2, **kwargs)


@attr.s
class Cavity(PulseMode):
    """Weakly non-linear cavity.

    A PulseMode whose primary operation is displacements.

    Args:
        name (str): Name of the mode for which to construct operators.
        levels (optional, int): Number of levels in the mode subspace. Default: 2.
        t1 (optional, float): Mode T1 time in nanoseconds. Default: inf.
        t2 (optional, float): Mode T2 time in nanoseconds. Default: inf.
        thermal_population (optional, float): Mode excited state population
            (in [0..1]). Default: 0.
        df (optional, float): Mode detuning in GHz. Default: 0.
        kerr (optional, float): Mode self-Kerr in GHz. Default: 0.
        pulses (optional, dict[str, Pulse]): Dict of pulses defined for this Mode.
            Default: empty dict {}.
        default_pulse (optional, str): Name of the default Pulse for this Mode.
            Default: "gaussian_pulse".
    """

    OPERATORS = Mode.OPERATORS + ("D",)

    def D(self, alpha, full_space=True):
        """Returns the displacement operator.

        Args:
            alpha (complex): Displacement amplitude.
            full_space (optional, bool): Whether to embed
                the operator in the full Hilbert space.
                Default: True.

        Returns:
            ``qutip.Qobj``: Displacement operator.
        """
        op = qutip.displace(self.levels, alpha)
        if full_space:
            return self.tensor_with_I(op)
        return op

    @capture_operation
    def displace(self, alpha, unitary=False, pulse_name=None, **kwargs):
        """Generate a displacement of amplitude ``alpha``.

        Keyword arguments are passed to the ``Pulse`` object
        if ``unitary`` is False.

        Args:
            alpha (complex): Displacement amplitude.
            unitary (optional, bool): Whether to return the unitary operator
                instead of generating a pulse. Default: False.
            pulse_name (optional, str): Name of the pulse to use. If None,
                will use ``self.default_pulse``. Default: None.

        Returns:
            ``qutip.Qobj`` or Operation: If ``unitary`` is True, returns
            the unitary operator Ry(angle), otherwise returns the resulting
            ``Operation`` object.
        """
        if unitary:
            return self.D(alpha)
        # Assuming default_pulse.amp = 1 corresponds to displacement of 1
        pulse_name = pulse_name or self.default_pulse
        pulse = getattr(self, pulse_name)
        norm = gaussian_chop_norm(pulse.sigma, pulse.chop)
        amp = 2 * alpha * pulse.amp / norm
        c_wave = pulse(amp=amp, **kwargs)
        i = -c_wave.imag
        q = c_wave.real
        terms = {
            f"{self.name}.x": HTerm(self.x, i),
            f"{self.name}.y": HTerm(self.y, q),
        }
        return Operation(len(c_wave), terms)
