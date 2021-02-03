"""
All two-qubit gates are implemented as unitary-only, i.e. there are no
pulse-based two-qubit gates.
"""

import numpy as np
import qutip

from ..modes import Mode


class TwoQubitGate(object):
    """An object representing a 2-qubit gate, which handles validation of the
    objects on which it acts, as well as the creation logical states for use
    in constructing the gate.

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.
    """

    def __init__(self, qubit1, qubit2):
        for role, mode in zip(["qubit1", "qubit2"], [qubit1, qubit2]):
            if not isinstance(mode, Mode):
                raise TypeError(
                    f"Expected object of type Mode for {role}, "
                    f"but got {type(mode)}."
                )
        if qubit1.space != qubit2.space:
            raise ValueError("The qubits must share a Hilbert space.")

        self.qubit1 = qubit1
        self.qubit2 = qubit2

    def logical_state(self, qubit1_state, qubit2_state):
        """Return the logical state ket(qubit1_state, qubit2_state).

        Args:
            qubit1_state (0 or 1): qubit1 state.
            qubit2_state (0 or 1): qubit2 state.

        Returns:
            ``qutip.Qobj``: Ket representing the requested logical state.
        """
        q1 = self.qubit1
        q2 = self.qubit2
        psi_q1 = q1.logical_states(full_space=False)
        psi_q2 = q2.logical_states(full_space=False)
        states = [qutip.qeye(mode.levels) for mode in q1.space]
        states[q1.index] = psi_q1[qubit1_state]
        states[q2.index] = psi_q2[qubit2_state]
        return qutip.tensor(*states)

    def __call__(self):
        raise NotImplementedError(
            "Call method must be implemented by subclasses "
            "of TwoQubitGate to specify the action of the gate."
        )


class ControlledTwoQubitGate(TwoQubitGate):
    """An object representing a controlled 2-qubit gate, which handles
    validation of the objects on which it acts, as well as the creation
    of logical states for use in constructing the gate.

    Control qubit is listed first, target is second.

    Args:
        control (Mode): Mode acting as the control qubit.
        target (Mode): Mode acting as the target qubit.
    """

    def __init__(self, control=None, target=None):
        super().__init__(control, target)


class CUGate(ControlledTwoQubitGate):
    r"""Controlled-U gate.

    .. math::
        CU(\theta,\phi,\lambda) &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ \cos(\theta/2)|10\rangle\langle10|\\
        &+ e^{i(\phi + \lambda)}\cos(\theta/2)|11\rangle\langle11|\\
        &+ e^{i\phi}\sin(\theta/2)|11\rangle\langle10|\\
        &- e^{i\lambda}\sin(\theta/2)|10\rangle\langle11|\\

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self, theta, phi, lamda):
        """
        Args:
            theta, phi, lamda (float): Euler angles, in radians.
        Returns:
            ``qutip.Qobj``: The CU operator.
        """
        psi00 = self.logical_state(0, 0)
        psi10 = self.logical_state(1, 0)
        psi01 = self.logical_state(0, 1)
        psi11 = self.logical_state(1, 1)

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        CU = (
            psi00 * psi00.dag()
            + psi01 * psi01.dag()
            + c * psi10 * psi10.dag()
            + np.exp(1j * (phi + lamda)) * c * psi11 * psi11.dag()
            + np.exp(1j * phi) * s * psi11 * psi10.dag()
            - np.exp(1j * lamda) * s * psi10 * psi11.dag()
        )
        return CU


class CXGate(CUGate):
    r"""Controlled-X gate.

    .. math::
        CX &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ |11\rangle\langle10|\\
        &+ |10\rangle\langle11|\\
        &= CU(\pi, 0, \pi)

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The CX operator.
        """
        return CUGate.__call__(self, np.pi, 0, np.pi)


class CYGate(CUGate):
    r"""Controlled-Y gate.

    .. math::
        CY &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ i|11\rangle\langle10|\\
        &- i|10\rangle\langle11|\\
        &= CU(\pi,\pi/2,\pi/2)

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The CY operator.
        """
        return CUGate.__call__(self, np.pi, np.pi / 2, np.pi / 2)


class CZGate(CUGate):
    r"""Controlled-Z gate.

    .. math::
        CZ &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ |10\rangle\langle10|\\
        &- |11\rangle\langle11|\\
        &= CU(0,0,\pi)

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The CZ operator.
        """
        return CUGate.__call__(self, 0, 0, np.pi)


class CPhaseGate(ControlledTwoQubitGate):
    r"""Controlled-phase gate.

    .. math::
        C_\text{phase} &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ |10\rangle\langle10|\\
        &+ e^{i\varphi}|11\rangle\langle11|\\
        &= CU(0, 0, \varphi)

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self, phi):
        """
        Args:
            phi (float): Phase in radians.

        Returns:
            ``qutip.Qobj``: The Cphase operator.
        """
        return CUGate.__call__(self, 0, 0, phi)


class SWAPGate(TwoQubitGate):
    r"""SWAP gate.

    .. math::
        \text{SWAP} &= |00\rangle\langle00|\\
        &+ |01\rangle\langle10|\\
        &+ |10\rangle\langle01|\\
        &+ |11\rangle\langle11|\\

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The SWAP operator.
        """
        psi00 = self.logical_state(0, 0)
        psi10 = self.logical_state(1, 0)
        psi01 = self.logical_state(0, 1)
        psi11 = self.logical_state(1, 1)

        SWAP = (
            psi00 * psi00.dag()
            + psi01 * psi10.dag()
            + psi10 * psi01.dag()
            + psi11 * psi11.dag()
        )
        return SWAP


class SWAPphiGate(ControlledTwoQubitGate):
    r"""SWAPphi gate.

    .. math::
        \text{SWAP}_\varphi &= |00\rangle\langle00|\\
        &+ ie^{-i\varphi}|01\rangle\langle10|\\
        &- ie^{i\varphi}|10\rangle\langle01|\\
        &+ |11\rangle\langle11|\\

    .. note::
        SWAPphi is only a "controlled" gate for certain values of :math:`\varphi`.
        For :math:`\varphi = n\pi + \frac{\pi}{2}` the gate is symmetric
        with respect to qubit ordering.

    Args:
        control (Mode): Mode acting as the control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self, phi):
        """
        Args:
            phi (float): Phase in radians.

        Returns:
            ``qutip.Qobj``: The SWAPphi operator.
        """
        psi00 = self.logical_state(0, 0)
        psi10 = self.logical_state(1, 0)
        psi01 = self.logical_state(0, 1)
        psi11 = self.logical_state(1, 1)

        SWAPphi = (
            psi00 * psi00.dag()
            + 1j * np.exp(-1j * phi) * psi01 * psi10.dag()
            - 1j * np.exp(1j * phi) * psi10 * psi01.dag()
            + psi11 * psi11.dag()
        )
        return SWAPphi


class iSWAPGate(TwoQubitGate):
    r"""iSWAP gate.

    .. math::
        i\text{SWAP} &= |00\rangle\langle00|\\
        &+ i|01\rangle\langle10|\\
        &+ i|10\rangle\langle01|\\
        &+ |11\rangle\langle11|\\

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The iSWAP operator.
        """
        psi00 = self.logical_state(0, 0)
        psi10 = self.logical_state(1, 0)
        psi01 = self.logical_state(0, 1)
        psi11 = self.logical_state(1, 1)

        iSWAP = (
            psi00 * psi00.dag()
            + 1j * psi01 * psi10.dag()
            + 1j * psi10 * psi01.dag()
            + psi11 * psi11.dag()
        )
        return iSWAP


class eSWAPGate(ControlledTwoQubitGate):
    r"""eSWAP gate.

    .. math::
        e\text{SWAP}_\varphi(\theta_c) &=
        \exp(-i\theta_c/2\cdot\text{SWAP}\varphi)\\
        &= \cos(\theta_c/2) I - i\sin(\theta_c/2)\text{SWAP}_\varphi

    .. note::
        eSWAP is only a "controlled" gate for certain values of :math:`\varphi`.
        For :math:`\varphi = n\pi + \frac{\pi}{2}` the gate is symmetric
        with respect to qubit ordering.

    Args:
        control (Mode): Mode acting as the control qubit.
        target (Mode): Mode acting as the target qubit.

    .. automethod:: __call__
    """

    def __call__(self, theta_c, phi=np.pi / 2):
        """
        Args:
            theta_c (float): Control angle in radians.
            phi (optional, float): Phase in radians. Default: pi/2.

        Returns:
            ``qutip.Qobj``: The eSWAP operator.
        """
        q1 = self.qubit1
        q2 = self.qubit2
        swapphi = SWAPphiGate(q1, q2)(phi)
        return (-1j * theta_c / 2 * swapphi).expm()


class SqrtSWAPGate(TwoQubitGate):
    r"""SqrtSWAP gate.

    .. math::
        \sqrt{\text{SWAP}}

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The SqrtSWAP operator.
        """
        q1 = self.qubit1
        q2 = self.qubit2
        swap = SWAPGate(q1, q2)()
        return swap.sqrtm()


class SqrtiSWAPGate(TwoQubitGate):
    r"""SqrtiSWAP gate.

    .. math::
        \sqrt{i\text{SWAP}}

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    .. automethod:: __call__
    """

    def __call__(self):
        """
        Returns:
            ``qutip.Qobj``: The SqrtiSWAP operator.
        """
        q1 = self.qubit1
        q2 = self.qubit2
        iswap = iSWAPGate(q1, q2)()
        return iswap.sqrtm()


def cu(control, target, theta, phi, lamda):
    r"""Controlled-U gate.

    .. math::
        CU(\theta,\phi,\lambda) &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ \cos(\theta/2)|10\rangle\langle10|\\
        &+ e^{i(\phi + \lambda)}\cos(\theta/2)|11\rangle\langle11|\\
        &+ e^{i\phi}\sin(\theta/2)|11\rangle\langle10|\\
        &- e^{i\lambda}\sin(\theta/2)|10\rangle\langle11|\\

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.
        theta, phi, lamda (float): Euler angles, in radians.

    Returns:
        ``qutip.Qobj``: The CU operator.
    """
    return CUGate(control=control, target=target)(theta, phi, lamda)


def cx(control, target):
    r"""Controlled-X gate.

    .. math::
        CX &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ |11\rangle\langle10|\\
        &+ |10\rangle\langle11|\\

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    Returns:
        ``qutip.Qobj``: The CX operator.
    """
    return CXGate(control=control, target=target)()


def cy(control, target):
    r"""Controlled-Y gate.

    .. math::
        CY &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ i|11\rangle\langle10|\\
        &- i|10\rangle\langle11|\\

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    Returns:
        ``qutip.Qobj``: The CY operator.
    """
    return CYGate(control=control, target=target)()


def cz(control, target):
    r"""Controlled-Z gate.

    .. math::
        CZ &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ |10\rangle\langle10|\\
        &- |11\rangle\langle11|\\

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.

    Returns:
        ``qutip.Qobj``: The CZ operator.
    """
    return CZGate(control=control, target=target)()


def cphase(control, target, phi):
    r"""Controlled-phase gate.

    .. math::
        C_\text{phase} &= |00\rangle\langle00|\\
        &+ |01\rangle\langle01|\\
        &+ |10\rangle\langle10|\\
        &+ e^{i\varphi}|11\rangle\langle11|\\

    Args:
        control (Mode): Mode acting as control qubit.
        target (Mode): Mode acting as the target qubit.
        phi (float): Phase in radians.

    Returns:
        ``qutip.Qobj``: The Cphase operator.
    """
    return CPhaseGate(control=control, target=target)(phi)


def swap(qubit1, qubit2):
    r"""SWAP gate.

    .. math::
        \text{SWAP} &= |00\rangle\langle00|\\
        &+ |01\rangle\langle10|\\
        &+ |10\rangle\langle01|\\
        &+ |11\rangle\langle11|\\

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    Returns:
        ``qutip.Qobj``: The SWAP operator.
    """
    return SWAPGate(qubit1, qubit2)()


def swapphi(control, target, phi):
    r"""SWAPphi gate.

    .. math::
        \text{SWAP}_\varphi &= |00\rangle\langle00|\\
        &+ ie^{-i\varphi}|01\rangle\langle10|\\
        &- ie^{i\varphi}|10\rangle\langle01|\\
        &+ |11\rangle\langle11|\\

    .. note::
        SWAPphi is only a "controlled" gate for certain values of :math:`\varphi`.
        For :math:`\varphi = n\pi + \frac{\pi}{2}` the gate is symmetric
        with respect to qubit ordering.

    Args:
        control (Mode): Mode acting as the control qubit.
        target (Mode): Mode acting as the target qubit.
        phi (float): Phase in radians.

    Returns:
        ``qutip.Qobj``: The SWAPphi operator.
    """
    return SWAPphiGate(control=control, target=target)(phi)


def iswap(qubit1, qubit2):
    r"""iSWAP gate.

    .. math::
        i\text{SWAP} &= |00\rangle\langle00|\\
        &+ i|01\rangle\langle10|\\
        &+ i|10\rangle\langle01|\\
        &+ |11\rangle\langle11|\\

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    Returns:
        ``qutip.Qobj``: The iSWAP operator.
    """
    return iSWAPGate(qubit1, qubit2)()


def eswap(control, target, theta_c, phi=np.pi / 2):
    r"""eSWAP gate.

    .. math::
        e\text{SWAP}_\varphi(\theta_c) &=
        \exp(-i\theta_c/2\cdot\text{SWAP}\varphi)\\
        &= \cos(\theta_c/2) I - i\sin(\theta_c/2)\text{SWAP}_\varphi

    .. note::
        eSWAP is only a "controlled" gate for certain values of :math:`\varphi`.
        For :math:`\varphi = n\pi + \frac{\pi}{2}` the gate is symmetric
        with respect to qubit ordering.

    Args:
        control (Mode): Mode acting as the control qubit.
        target (Mode): Mode acting as the target qubit.
        theta_c (float): Control angle in radians.
        phi (optional, float): Phase in radians. Default: pi/2.

    Returns:
        ``qutip.Qobj``: The eSWAP operator.
    """
    return eSWAPGate(control=control, target=target)(theta_c, phi=phi)


def sqrtswap(qubit1, qubit2):
    r"""SqrtSWAP gate.

    .. math::
        \sqrt{\text{SWAP}}

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    Returns:
        ``qutip.Qobj``: The SqrtSWAP operator.
    """
    return SqrtSWAPGate(qubit1, qubit2)()


def sqrtiswap(qubit1, qubit2):
    r"""SqrtiSWAP gate.

    .. math::
        \sqrt{i\text{SWAP}}

    Args:
        qubit1 (Mode): Mode acting as the first qubit.
        qubit2 (Mode): Mode acting as the second qubit.

    Returns:
        ``qutip.Qobj``: The SqrtiSWAP operator.
    """
    return SqrtiSWAPGate(qubit1, qubit2)()
