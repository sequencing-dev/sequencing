from functools import wraps

import numpy as np

from ..modes import Mode, Transmon


def single_qubit_gate(func):
    """A decorator used to specify that the decorated function
    is a single-qubit gate acting on one or more Modes.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        space = None
        for arg in args:
            if isinstance(arg, Mode):
                if space is None:
                    space = arg.space
                if arg.space != space:
                    raise ValueError("All Modes must share the same Hilbert space.")
        result = func(*args, **kwargs)
        if isinstance(result, list):
            if all(r is None for r in result):
                return None
            if len(result) == 1:
                return result[0]
        return result

    return wrapped


def pulsed_gate_exists(*types):
    """A decorator used to specify types of Modes
    for which a pulse-based version of the gate exists.

    If the first argument is ``None``, then the gate is required
    to be unitary-only for all types of Modes.
    """
    unitary_only = False
    if types[0] is None:
        unitary_only = True

    def pulsed_gate_exists_decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            kwargs = kwargs.copy()
            unitary = kwargs.get("unitary", True)
            if unitary_only:
                _ = kwargs.pop("unitary", None)
            if not unitary:
                if unitary_only:
                    raise ValueError(
                        f"No pulse-based version of "
                        f"the {func.__name__} gate exists."
                    )
                for arg in args:
                    if isinstance(arg, Mode) and not isinstance(arg, types):
                        raise TypeError(
                            f"No pulse-based version of the {func.__name__} "
                            f"gate exists for Modes of type {type(arg)}."
                        )
            return func(*args, **kwargs)

        return wrapped

    return pulsed_gate_exists_decorator


@pulsed_gate_exists(None)
@single_qubit_gate
def U(theta, phi, lamda, *qubits, **kwargs):
    r"""
    .. math::
            U(\theta,\phi,\lambda) = R_z(\phi)R_y(\theta)R_z(\lambda)

    Args:
        theta, phi, lamda (float): Euler angles, in radians.
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return [q.Rz(phi) * q.Ry(theta) * q.Rz(lamda) for q in qubits]


@pulsed_gate_exists(Transmon)
@single_qubit_gate
def rx(theta, *qubits, **kwargs):
    r"""Rotates each mode about its x axis by a given angle.

    .. math::
        R_x(\theta) = \exp(-i\theta/2 \cdot X)

    Args:
        theta (float): Rotation angle in radians.
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return [q.rotate_x(theta, **kwargs) for q in qubits]


@pulsed_gate_exists(Transmon)
@single_qubit_gate
def ry(theta, *qubits, **kwargs):
    r"""Rotates each mode about its y axis by a given angle.

    .. math::
        R_y(\theta) = \exp(-i\theta/2 \cdot Y)

    Args:
        theta (float): Rotation angle in radians.
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return [q.rotate_y(theta, **kwargs) for q in qubits]


@pulsed_gate_exists(None)
@single_qubit_gate
def rz(phi, *qubits, **kwargs):
    r"""Rotates each mode about its z axis by a given angle.

    .. math::
        R_z(\phi) = \exp(-i\phi/2 \cdot Z)

    Args:
        phi (float): Rotation angle in radians.
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return [q.Rz(phi, **kwargs) for q in qubits]


@pulsed_gate_exists(Transmon)
@single_qubit_gate
def x(*qubits, **kwargs):
    r"""X gate.

    .. math::
        X = R_x(\pi)

    Args:
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return rx(np.pi, *qubits, **kwargs)


@pulsed_gate_exists(Transmon)
@single_qubit_gate
def y(*qubits, **kwargs):
    r"""Y gate.

    .. math::
        Y = R_y(\pi)

    Args:
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return ry(np.pi, *qubits, **kwargs)


@pulsed_gate_exists(None)
@single_qubit_gate
def z(*qubits, **kwargs):
    r"""Z gate.

    .. math::
        Z = R_z(\pi)

    Args:
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return rz(np.pi, *qubits, **kwargs)


@pulsed_gate_exists(None)
@single_qubit_gate
def h(*qubits, **kwargs):
    r"""Hadamard gate.

    .. math::
        H = \frac{1}{\sqrt{2}}\left(X + Z\right)

    Args:
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return [q.hadamard(**kwargs) for q in qubits]


@pulsed_gate_exists(None)
@single_qubit_gate
def r(theta, phi, *qubits, **kwargs):
    r"""Rotate each mode by an angle ``theta`` about the axis
    given by ``phi``.

    .. math::
        R_\text{axis}(\theta,\phi)
        = \exp\left(-i\theta/2 \cdot (\cos(\phi)X + \sin(\phi)Y)\right)

    Args:
        theta (float): Rotation angle in radians.
        phi (float): Angle between the axis of rotation
            and the x axis.
        *qubits (tuple[Mode]): Modes to which to apply the gate.

    Returns:
        list[Operation or qutip.Qobj] or None:
        List of length ``len(qubits)``, or None if the
        gates were captured.
    """
    return [q.Raxis(theta, phi, **kwargs) for q in qubits]
