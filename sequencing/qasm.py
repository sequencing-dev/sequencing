# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import ast
import operator
import inspect
from functools import reduce

import numpy as np
import qutip

from .sequencing import Sequence, PulseSequence, Operation, SyncOperation


def _eval_expr(expr):
    """
    https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
    >>> _eval_expr('2^6')
    4
    >>> _eval_expr('2**6')
    64
    >>> _eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    def eval_(node):
        if isinstance(node, ast.Num):
            # <number>
            return node.n
        elif isinstance(node, ast.BinOp):
            # <left> <operator> <right>
            return operators[type(node.op)](eval_(node.left), eval_(node.right))
        elif isinstance(node, ast.UnaryOp):
            # <operator> <operand> e.g., -1
            return operators[type(node.op)](eval_(node.operand))
        else:
            raise TypeError(node)

    return eval_(ast.parse(expr, mode="eval").body)


def _str_between(text, start, stop):
    """Returns the string found between substrings `start` and `stop`."""
    return text[text.find(start) + 1 : text.rfind(stop)]


def parse_qasm_gate(qasm_str):
    """Parses a QASM string like ``'u3(pi/2,0,pi)'``
    into the gate name ``'u3'`` and a tuple of float
    arguments ``(np.pi/2, 0, np.pi/2)``.

    Args:
        qasm_str (str): String like 'u3(pi/2,0,pi)'

    Returns:
        tuple[str, tuple[float]]: (gate name, gate args)
    """
    if "barrier" in qasm_str:
        return "barrier", tuple()
    gate_call = qasm_str.split(" ")[0]
    if "(" not in gate_call:
        return gate_call, tuple()
    gate = gate_call.split("(")[0]
    gate_args = _str_between(gate_call, "(", ")").split(",")
    args = tuple(_eval_expr(a.replace("pi", f"{np.pi:.15f}")) for a in gate_args)
    return gate, args


class QasmSequence(Sequence):
    """A Sequence with methods implementing the single-qubit
    gates defined in the OpenQASM specification.

    References:

        1) https://qiskit.github.io/openqasm/
        2) https://arxiv.org/abs/1707.03429
        3) https://qiskit.org/textbook/ch-states/single-qubit-gates.html

    Args:
        system (System): System upon which the
            sequence will act.
        operations
            (optional, list[CompiledPulseSequence, Operation, Sequence]):
            Initial list of Operations or unitaries. Default: None.
    """

    VALID_TYPES = Sequence.VALID_TYPES + (SyncOperation,)

    def _validate(self, item):
        if isinstance(item, list):
            assert len(item) == 3
            assert isinstance(item[0], qutip.Qobj)
            assert isinstance(item[1], (Operation, qutip.Qobj))
            assert isinstance(item[2], qutip.Qobj)
            return [Sequence._validate(self, i) for i in item]
        return Sequence._validate(self, item)

    def U(self, theta, phi, lamda, *qubits, unitary=True, append=True):
        r"""
        .. math::
            U(\theta,\phi,\lambda) = u_3(\theta,\phi,\lambda) =
            R_z(\phi)R_y(\theta)R_z(\lambda)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """

        Rys = []
        Ry_ops = []
        if not unitary:
            for qubit in qubits:
                if not hasattr(qubit, "rotate_y"):
                    raise AttributeError(
                        "If unitary is False, all qubits must "
                        "implement rotate_y "
                        f"({qubit.name} does not)."
                    )
        for qubit in qubits:
            if unitary:
                Rys.append(qubit.Ry(theta))
            else:
                Ry_ops.append(qubit.rotate_y(theta, unitary=False, capture=False))
        if not unitary:
            Ry_terms = {}
            duration = None
            for operation in Ry_ops:
                if duration is None:
                    duration = operation.duration
                if operation.duration != duration:
                    raise ValueError("All Operations must have the same duration.")
                Ry_terms.update(operation.terms)
            Rys = [Operation(duration, Ry_terms)]

        def Rz(angle):
            return reduce(lambda a, b: a * b, (qubit.Rz(angle) for qubit in qubits))

        gate = [Rz(phi)] + Rys + [Rz(lamda)]
        if append:
            # reverse order - we want Rz(lamda) applied first
            self.append(gate[::-1])
            return
        if unitary:
            gate = reduce(lambda a, b: a * b, gate)
        else:
            # reverse order - we want Rz(lamda) applied first
            gate = gate[::-1]
        return gate

    u3 = U

    def u2(self, phi, lamda, *qubits, unitary=True, append=True):
        r"""
        .. math::
            u_2(\phi,\lambda) = U(\pi/2,\phi,\lambda)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.U(np.pi / 2, phi, lamda, *qubits, unitary=unitary, append=append)

    def u1(self, lamda, *qubits, append=True):
        r"""
        .. math::
            p(\lambda) = u_1(\lambda) = U(0,0,\lambda) = R_z(\lambda)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.U(0, 0, lamda, *qubits, unitary=True, append=append)

    p = u1

    def id(self, *qubits, unitary=True, append=True):
        r"""Identity.

        .. math::
            I = U(0,0,0)

        Returns:
            ``qutip.Qobj`` or None
        """
        return self.u3(0, 0, 0, *qubits, unitary=unitary, append=append)

    def x(self, *qubits, unitary=True, append=True):
        r"""
        .. math::
            x = u_3(\pi,0,\pi)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u3(np.pi, 0, np.pi, *qubits, unitary=unitary, append=append)

    def y(self, *qubits, unitary=True, append=True):
        r"""
        .. math::
            y = u_3(\pi,\pi/2,\pi/2)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u3(
            np.pi, np.pi / 2, np.pi / 2, *qubits, unitary=unitary, append=append
        )

    def z(self, *qubits, append=True):
        r"""
        .. math::
            z = u_1(\pi) = R_z(\pi)

        Returns:
            ``qutip.Qobj`` or None
        """
        return self.u1(np.pi, *qubits, append=append)

    def h(self, *qubits, unitary=True, append=True):
        r"""
        .. math::
            h = i \cdot u_2(0,\pi)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        gate = self.u2(0, np.pi, *qubits, unitary=unitary, append=append)
        if gate is not None:
            phase = self.gphase(np.pi / 2, append=False)
            if isinstance(gate, qutip.Qobj):
                gate = phase * gate
            else:
                gate[0] = phase * gate[0]
            return gate

    def s(self, *qubits, append=True):
        r"""
        .. math::
            s = \sqrt{z} = u_1(\pi/2) = R_z(\pi/2)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u1(np.pi / 2, *qubits, append=append)

    def sdg(self, *qubits, append=True):
        r"""
        .. math::
            s^\dagger =  u_1(-\pi/2) = R_z(-\pi/2)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u1(-np.pi / 2, *qubits, append=append)

    def t(self, *qubits, append=True):
        r"""
        .. math::
            t = u_1(\pi/4) = R_z(\pi/4)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u1(np.pi / 4, *qubits, append=append)

    def tdg(self, *qubits, append=True):
        r"""
        .. math::
            t^\dagger = u_1(-\pi/4) = R_z(-\pi/4)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u1(-np.pi / 4, *qubits, append=append)

    def rx(self, theta, *qubits, unitary=True, append=True):
        r"""
        .. math::
            R_x(\theta) = u_3(\theta,-\pi/2,\pi/2)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u3(
            theta, -np.pi / 2, np.pi / 2, *qubits, unitary=unitary, append=append
        )

    def ry(self, theta, *qubits, unitary=True, append=True):
        r"""
        .. math::
            R_y(\theta) = u_3(\theta,0,0)

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        return self.u3(theta, 0, 0, *qubits, unitary=unitary, append=append)

    def rz(self, phi, *qubits, append=True):
        r"""
        .. math::
            R_z(\phi) = u_1(\phi)

        Returns:
            ``qutip.Qobj`` or None
        """
        return self.u1(phi, *qubits, append=append)

    def sx(self, *qubits, unitary=True, append=True):
        r"""
        .. math::
            \sqrt{X} &= \exp(i\pi/4)R_x(\pi/2)\\
            &= \exp(-i\pi/4)R_z(-\pi/2)\cdot H\cdot R_z(-\pi/2)\\
            &= \exp(-i\pi/4)s^\dagger\cdot H\cdot s^\dagger

        Returns:
            ``qutip.Qobj`` or list[qutip.Qobj] or None
        """
        gate = self.rx(np.pi / 2, *qubits, unitary=unitary, append=append)
        if gate is not None:
            phase = self.gphase(np.pi / 4, append=False)
            if isinstance(gate, qutip.Qobj):
                gate = phase * gate
            else:
                gate[0] = phase * gate[0]
            return gate

    def gphase(self, gamma, *qubits, append=True):
        """Adds a global phase to the sequence.

        Args:
            gamma (float): Phase angle in radians.
            *qubits: Sequence of qubits (ignored completely).
            append (optional, bool): If True, append the gate
                to self rather than returning it. Default: True.

        Returns:
            ``qutip.Qobj`` or None: If append is False, returns
            the gphase unitary, otherwise returns None.
        """
        gphase = np.exp(1j * gamma) * self.system.I()
        if append:
            self.append(gphase)
        else:
            return gphase

    def CX(self, control, target, append=True):
        r"""Controlled-X gate.

        .. math::
            CX &= |00\rangle\langle00|\\
            &+ |01\rangle\langle01|\\
            &+ |11\rangle\langle10|\\
            &+ |10\rangle\langle11|\\

        Args:
            control (Mode): Mode acting as control qubit.
            target (Mode): Mode acting as the target qubit.
            append (optional, bool): If True, append the gate
                to self rather than returning it. Default: True.

        Returns:
            ``qutip.Qobj`` or None: If append is False, returns
            the CX unitary, otherwise returns None.
        """
        if control.space != target.space:
            raise ValueError("Control and target must share a Hilbert space.")
        psi_ctrl = control.logical_states(full_space=False)
        psi_trgt = target.logical_states(full_space=False)

        def logical_state(ctrl_state, trgt_state):
            states = [qutip.qeye(mode.levels) for mode in control.space]
            states[control.index] = psi_ctrl[ctrl_state]
            states[target.index] = psi_trgt[trgt_state]
            return self.system.tensor(*states)

        psi00 = logical_state(0, 0)
        psi10 = logical_state(1, 0)
        psi01 = logical_state(0, 1)
        psi11 = logical_state(1, 1)

        CX = (
            psi00 * psi00.dag()
            + psi01 * psi01.dag()
            + psi11 * psi10.dag()
            + psi10 * psi11.dag()
        )
        if append:
            self.append(CX)
        return CX

    def barrier(self, *args, append=True):
        """Equivalent to ``sync()``.

        NOTE: Currently only a "global" barrier/sync is supported.
        """
        if append:
            self.append(SyncOperation())
        else:
            return SyncOperation()

    def assemble(self):
        """Rearrange the sequence so that blocks of operations
        not separated by a barrier are executed in parallel.
        """
        if not any(isinstance(item, list) for item in self):
            # A bare SyncOperation is not a valid type for a Sequence
            gates = self[:]
            self.clear()
            self.extend([g for g in gates if not isinstance(g, SyncOperation)])
            return
        gates = []
        # Blocks are collections of unitaries or Operations
        # that are separated by barriers.
        blocks = [[]]
        # Assemble blocks
        for item in self:
            if isinstance(item, SyncOperation):
                blocks.append([])
            elif isinstance(item, PulseSequence):
                if len(item) == 0:
                    continue
                elif len(item) == 1 and isinstance(item[0], SyncOperation):
                    blocks.append([])
                else:
                    blocks[-1].extend(item)
            else:
                blocks[-1].append(item)
        # Now rearrange the blocks so operations that are
        # intended to occur in parallel actually do.
        for block in blocks:
            # Each gate is assumed to be either a single unitary/Operation,
            # or a list of length 3:
            # [pre_operation, operation, post_operation].
            pre = []
            operations = []
            post = []
            for item in block:
                if isinstance(item, list):
                    assert len(item) == 3
                    assert isinstance(item[0], qutip.Qobj)
                    assert isinstance(item[1], (Operation, qutip.Qobj))
                    assert isinstance(item[2], qutip.Qobj)
                    pre.append(item[0])
                    if isinstance(item[1], qutip.Qobj):
                        post.append(item[1])
                    else:
                        operations.append(item[1])
                    post.append(item[2])
                else:
                    pre.append(item)
            # All pre_operations happen in the same time slice.
            gates.extend(pre)
            # Followed by the PulseSequence,
            # which must end in a sync.
            seq = PulseSequence(self.system)
            seq.extend(operations)
            if len(seq):
                seq.append(SyncOperation())
                gates.append(seq)
            # All post_unitaries happen in the same time slice.
            gates.extend(post)
        self.clear()
        # A bare SyncOperation is not a valid type for a Sequence
        self.extend([g for g in gates if not isinstance(g, SyncOperation)])

    def qasm(self, qasm_str, unitary=True, append=True):
        """Executes a gate specified by a single QASM instruction,
        e.g. 'rx(pi) q[0];'. The qubit index in the QuantumRegister
        corresponds to the index in self.system.active_modes.

        Args:
            qasm_str (str): String specifying the QASM gate to execute.
            unitary (optional, bool): Whether to use the unitary
                instead of pulse-based version of the gate. This argument
                is ignored if only the unitary version exists. Default: True.
            append (optional, bool): Whether to append the gate to self
                instead of returning it to the user. This argument is
                ignored for `barrier`. Default: True.

        Returns:
            list, Operation, ``qutip.Qobj``, or None:
            If append is False, returns the result of the gate,
            which will be either an Operation, a qutip.Qobj,
            or a list composed of those types.
            If append is True, returns None.
        """
        if "barrier" in qasm_str:
            return self.barrier(append=append)
        gate_str, qubit_str = qasm_str.split(" ")
        gate_name, args = parse_qasm_gate(gate_str)
        if "qreg" in qubit_str:
            indices = list(range(len(self.system.active_modes)))
        else:
            indices = [
                int(_str_between(qstr, "[", "]")) for qstr in qubit_str.split(",")
            ]
        qubits = [self.system.active_modes[-(1 + i)] for i in indices]
        gate = getattr(self, gate_name)
        gate_args = inspect.signature(gate).parameters
        kwargs = {}
        if "append" in gate_args:
            kwargs["append"] = append
        if "unitary" in gate_args:
            kwargs["unitary"] = unitary
        args = list(args) + qubits
        return gate(*args, **kwargs)

    def qasm_circuit(self, circuit, unitary=True, append=True):
        """Executes a full QASM circuit, ignoring reset, measure,
        and conditional instructions.

        Args:
            circuit (str or list[str]): The circuit to execute, either
                as a list of gates or as a newline-delimited string.
            unitary (optional, bool): Whether to use the unitary
                instead of pulse-based version of the each gate. This argument
                is ignored if only the unitary version exists. Default: True.
            append (optional, bool): Whether to append each gate to self
                instead of returning it to the user. This argument is
                ignored for `barrier`. Default: True.

        Returns:
            list or None:
            If append is False, returns a list of ``Operation``
            (if unitary is False) and ``qutip.Qobj``.
            If append is True, returns None.
        """
        ignore = [
            "OPENQASM",
            "include",
            "measure",
            "reset",
            "if",
            "post",
            "opaque",
            "gate",
            "creg",
        ]
        gates = []
        if isinstance(circuit, str):
            circuit = [c.strip() for c in circuit.split(";")]
        for line in circuit:
            if (
                not line
                or line.startswith("qreg")
                or line.startswith("//")
                or any(phrase in line for phrase in ignore)
            ):
                continue
            gate = self.qasm(line, unitary=unitary, append=False)
            gates.append(gate)
        if append:
            self.extend(gates)
        else:
            return gates

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
        self.assemble()
        return super().run(
            init_state,
            e_ops=e_ops,
            options=options,
            full_evolution=full_evolution,
            progress_bar=progress_bar,
        )

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
        self.assemble()
        return super().plot_coefficients(
            subplots=subplots, sharex=sharex, sharey=sharey
        )

    def measure(self, state, *qubits):
        """Meausure each qubit in its logical basis by tracing over
        all other modes and then taking the expectation value of
        the projector onto logical one acting on the state.

        Args:
            state (qutip.Qobj): The state to measure.
            *qubits: (sequence[Mode]): Tuple of Modes to measure.
                If qubits is empty, then all Modes in self.system.active_modes
                are measured.

        Returns:
            list[float]: List of expectation values.
        """
        if len(qubits) == 0:
            qubits = self.system.active_modes
        expect = []
        for qubit in qubits:
            qstate = state.ptrace(qubit.index)
            one = qubit.logical_one(full_space=False)
            proj = one * one.dag()
            expect.append(qutip.expect(proj, qstate))
        return expect
