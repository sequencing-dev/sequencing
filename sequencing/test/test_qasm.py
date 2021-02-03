import unittest
import numpy as np
from qutip import fidelity

from sequencing import Transmon, System
from sequencing.calibration import tune_rabi, tune_drag
from sequencing.qasm import QasmSequence


class TestQasmSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q0 = Transmon("q0", levels=2)
        q1 = Transmon("q1", levels=2)
        system = System("system", modes=[q0, q1])

        for qubit in [q0, q1]:
            init_state = system.fock()
            for _ in range(1):
                _ = tune_rabi(
                    system, init_state=init_state, mode_name=qubit.name, verify=False
                )

        cls.system = system

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_idxyzsx_unitary(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            gate = seq.qasm(f"id q[{i}];", append=False)
            self.assertEqual(gate, qubit.I)

            gate = seq.qasm(f"x q[{i}];", append=False)
            self.assertEqual(gate, qubit.Rx(np.pi))

            gate = seq.qasm(f"y q[{i}];", append=False)
            self.assertEqual(gate, qubit.Ry(np.pi))

            gate = seq.qasm(f"z q[{i}];", append=False)
            self.assertEqual(gate, qubit.Rz(np.pi))

            gate = seq.qasm(f"sx q[{i}];", append=False)
            self.assertEqual(gate, np.exp(1j * np.pi / 4) * qubit.Rx(np.pi / 2))

            sdg = qubit.Rz(-np.pi / 2)
            self.assertEqual(sdg, seq.qasm(f"sdg q[{i}];", append=False))
            gate = seq.qasm(f"sx q[{i}];", append=False)
            self.assertEqual(
                gate, np.exp(-1j * np.pi / 4) * sdg * qubit.hadamard() * sdg
            )

    def test_u2_unitary(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            # U2(\phi, \lambda) = R(\phi) RY(\frac{\pi}{2}) RZ(\lambda)
            for phi_denom in [1, 2, 3, 4]:
                for lam_denom in [1, 2, 3, 4]:
                    gate = seq.qasm(
                        f"u2(pi/{phi_denom},pi/{lam_denom}) q[{i}];", append=False
                    )
                    ideal = (
                        qubit.Rz(np.pi / phi_denom)
                        * qubit.Ry(np.pi / 2)
                        * qubit.Rz(np.pi / lam_denom)
                    )
                    self.assertEqual(gate, ideal)

            gate = seq.qasm(f"h q[{i}];", append=False)
            self.assertEqual(gate, qubit.hadamard())

            # U2(0, \pi) = H
            # It seems to actually equal -iH...
            self.assertEqual(
                -1j * seq.qasm(f"h q[{i}];", append=False),
                seq.qasm(f"u2(0,pi) q[{i}];", append=False),
            )

            # U2(0, 0) = RY(\pi/2)
            self.assertEqual(
                seq.qasm(f"u2(0,0) q[{i}];", append=False), qubit.Ry(np.pi / 2)
            )

            # U2(-\pi/2, \pi/2) = RX(\pi/2)
            self.assertEqual(
                seq.qasm(f"u2(-pi/2,pi/2) q[{i}];", append=False), qubit.Rx(np.pi / 2)
            )

    def test_st_unitary(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            gate = seq.qasm(f"s q[{i}];", append=False)
            self.assertEqual(gate, qubit.Rz(np.pi / 2))

            gate = seq.qasm(f"sdg q[{i}];", append=False)
            self.assertEqual(gate, qubit.Rz(-np.pi / 2))

            gate = seq.qasm(f"t q[{i}];", append=False)
            self.assertEqual(gate, qubit.Rz(np.pi / 4))

            gate = seq.qasm(f"tdg q[{i}];", append=False)
            self.assertEqual(gate, qubit.Rz(-np.pi / 4))

    def test_rotations_unitary(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            for denom in [1, 2, 3, 4]:
                gate = seq.qasm(f"rx(pi/{denom}) q[{i}];", append=False)
                self.assertEqual(gate, qubit.Rx(np.pi / denom))

                gate = seq.qasm(f"ry(-pi/{denom}) q[{i}];", append=False)
                self.assertEqual(gate, qubit.Ry(-np.pi / denom))

                gate = seq.qasm(f"rz(pi/{denom}) q[{i}];", append=False)
                self.assertEqual(gate, qubit.Rz(np.pi / denom))

                gate = seq.qasm(f"p(pi/{denom}) q[{i}];", append=False)
                self.assertEqual(gate, qubit.Rz(np.pi / denom))

                # U(\theta, -\pi/2, pi/2) = R_x(\theta)
                self.assertEqual(
                    seq.qasm(f"U(pi/{denom},-pi/2,pi/2) q[{i}];", append=False),
                    qubit.Rx(np.pi / denom),
                )

                # U(\theta, 0, 0) = R_y(\theta)
                self.assertEqual(
                    seq.qasm(f"U(pi/{denom},0,0) q[{i}];", append=False),
                    qubit.Ry(np.pi / denom),
                )

    def test_x_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            # x
            ideal = qubit.Rx(np.pi)
            seq.qasm(f"x q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]
            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )

    def test_y_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            # y
            ideal = qubit.Ry(np.pi)
            seq.qasm(f"y q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]
            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )

    def test_u2_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            # U2(\phi, \lambda) = RZ(\phi) RY(\frac{\pi}{2}) RZ(\lambda)
            for phi_denom in [1, 2, 3, 4]:
                for lam_denom in [1, 2, 3, 4]:
                    seq = QasmSequence(system)
                    ideal = (
                        qubit.Rz(np.pi / phi_denom)
                        * qubit.Ry(np.pi / 2)
                        * qubit.Rz(np.pi / lam_denom)
                    )
                    seq.qasm(
                        f"u2(pi/{phi_denom},pi/{lam_denom}) q[{i}];",
                        unitary=False,
                        append=True,
                    )
                    result = seq.run(init_state)
                    state = result.states[-1]
                    self.assertGreater(
                        fidelity(state, ideal * init_state) ** 2, target_fidelity
                    )

    def test_u2_rx_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            # U2(-\pi/2, \pi/2) = RX(\pi/2)
            ideal = qubit.Rx(np.pi / 2)
            seq.qasm(f"u2(-pi/2,pi/2) q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]
            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )

    def test_u2_ry_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)
            # U2(0, 0) = RY(\pi/2)
            ideal = qubit.Ry(np.pi / 2)
            seq.qasm(f"u2(0,0) q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]
            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )

    def test_hadamard_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            # h
            ideal = qubit.hadamard()
            seq.qasm(f"h q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]

            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )
            seq.clear()

            # U2(0, \pi) = H
            ideal = qubit.hadamard()
            seq.qasm(f"u2(0,pi) q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]
            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )
            seq.clear()

    def test_rx_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            for denom in [1, 2, 3, 4]:
                ideal = qubit.Rx(np.pi / denom)
                seq.qasm(f"rx(pi/{denom}) q[{i}];", unitary=False, append=True)
                result = seq.run(init_state)
                state = result.states[-1]
                self.assertGreater(
                    fidelity(state, ideal * init_state) ** 2, target_fidelity
                )
                seq.clear()

                # U(\theta, -\pi/2, pi/2) = R_x(\theta)
                ideal = qubit.Rx(np.pi / denom)
                seq.qasm(
                    f"U(pi/{denom},-pi/2,pi/2) q[{i}];", unitary=False, append=True
                )
                result = seq.run(init_state)
                state = result.states[-1]
                self.assertGreater(
                    fidelity(state, ideal * init_state) ** 2, target_fidelity
                )
                seq.clear()

    def test_ry_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            for denom in [1, 2, 3, 4]:
                ideal = qubit.Ry(-np.pi / denom)
                seq.qasm(f"ry(-pi/{denom}) q[{i}];", unitary=False, append=True)
                result = seq.run(init_state)
                state = result.states[-1]
                self.assertGreater(
                    fidelity(state, ideal * init_state) ** 2, target_fidelity
                )
                seq.clear()

                # U(\theta, 0, 0) = R_y(\theta)
                ideal = qubit.Ry(np.pi / denom)
                seq.qasm(f"U(pi/{denom},0,0) q[{i}];", unitary=False, append=True)
                result = seq.run(init_state)
                state = result.states[-1]
                self.assertGreater(
                    fidelity(state, ideal * init_state) ** 2, target_fidelity
                )
                seq.clear()

    def test_sx_pulse(self):
        system = self.system
        q0 = system.q0
        q1 = system.q1
        init_state = system.fock()
        target_fidelity = 1 - 1e-6

        for i, qubit in enumerate([q0, q1]):

            seq = QasmSequence(system)

            # sx
            gate = seq.qasm(f"sx q[{i}];", append=False)
            ideal = np.exp(1j * np.pi / 4) * qubit.Rx(np.pi / 2)
            self.assertEqual(gate, ideal)
            seq.qasm(f"sx q[{i}];", unitary=False, append=True)
            result = seq.run(init_state)
            state = result.states[-1]
            self.assertGreater(
                fidelity(state, ideal * init_state) ** 2, target_fidelity
            )


class TestQasmCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 5
        qubits = [
            Transmon(
                f"q{i}",
                # levels=2,
                # # comment out the above line and uncomment the
                # # below line to use 3-level transmons with
                # # various anharmonicities
                levels=3,
                kerr=-100e-3 * (i + 1),
            )
            for i in reversed(range(n))
        ]
        qreg = System("qreg", modes=qubits)

        # Tune pi pulses for all qubits
        for i, qubit in enumerate(qubits):
            # Use different sigmas so that pulses can be visually distinguished
            qubit.gaussian_pulse.set(sigma=(10 + 2 * i), chop=4)
            with qreg.use_modes([qubit]):
                init_state = qubit.fock(0)
                e_ops = [qubit.fock_dm(1)]
                _ = tune_rabi(
                    qreg,
                    init_state,
                    e_ops=e_ops,
                    mode_name=qubit.name,
                    plot=False,
                    verify=False,
                )
                # # Below line is unnecessary if the qubits
                # # only have two levels
                _ = tune_drag(
                    qreg, init_state, e_ops=e_ops, mode_name=qubit.name, plot=False
                )

        def bell_state(qreg):
            zeros = [0] * len(qreg.active_modes)
            ones = [1] * len(qreg.active_modes)
            return (qreg.logical_basis(*zeros) + qreg.logical_basis(*ones)).unit()

        cls.ideal_state = bell_state(qreg)
        cls.qreg = qreg

    def test_qasm_circuit_unitary(self):
        qreg = self.qreg
        n = len(qreg.active_modes)
        QASM_CIRCUIT = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "qreg q[{n}];",
            "h q[0];",
            "barrier;",
        ]
        QASM_CIRCUIT.extend([f"CX q[0],q[{i}];" for i in range(1, n)])

        QASM_CIRCUIT = "\n\t".join([""] + QASM_CIRCUIT)

        print("Running the following QASM circuit:")
        print(QASM_CIRCUIT)

        seq = QasmSequence(qreg)
        seq.qasm_circuit(QASM_CIRCUIT, unitary=True, append=True)

        result = seq.run(qreg.ground_state())

        fid = fidelity(result.states[-1], self.ideal_state) ** 2
        print(
            f"Unitary qasm_circuit Bell state fidelity "
            f"(n = {len(qreg.active_modes)}): {fid:.7f}."
        )
        self.assertLess(1 - fid, 1e-6)

    def test_qasm_circuit_pulse(self):
        qreg = self.qreg
        n = len(qreg.active_modes)
        QASM_CIRCUIT = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{n}];",
            "h q[0];",
            "barrier;",
        ]
        QASM_CIRCUIT.extend([f"CX q[0],q[{i}];" for i in range(1, n)])

        QASM_CIRCUIT = "\n\t".join([""] + QASM_CIRCUIT)

        print("Running the following QASM circuit:")
        print(QASM_CIRCUIT)

        seq = QasmSequence(qreg)
        seq.qasm_circuit(QASM_CIRCUIT, unitary=False, append=True)

        _ = seq.plot_coefficients(subplots=False)

        result = seq.run(qreg.ground_state())

        fid = fidelity(result.states[-1], self.ideal_state) ** 2
        print(
            f"Pulsed qasm_circuit Bell state fidelity "
            f"(n = {len(qreg.active_modes)}): {fid:.7f}."
        )
        self.assertLess(1 - fid, 1e-6)

    def test_qasm_sequence_unitary(self):
        qreg = self.qreg
        qubits = qreg.active_modes
        seq = QasmSequence(qreg)
        seq.h(qubits[-1], unitary=True)
        seq.barrier()
        for q in qubits[:-1]:
            seq.CX(qubits[-1], q)
        seq.gphase(np.pi / 2)
        result = seq.run(qreg.ground_state())

        fid = fidelity(result.states[-1], self.ideal_state) ** 2
        print(
            f"Unitary qasm sequence Bell state fidelity "
            f"(n = {len(qreg.active_modes)}): {fid:.7f}."
        )
        self.assertLess(1 - fid, 1e-6)

    def test_qasm_sequence_pulse(self):
        qreg = self.qreg
        qubits = qreg.active_modes
        seq = QasmSequence(qreg)
        seq.h(qubits[-1], unitary=False)
        seq.barrier()
        for q in qubits[:-1]:
            seq.CX(qubits[-1], q)
        seq.gphase(np.pi / 2)

        _ = seq.plot_coefficients(subplots=False)
        result = seq.run(qreg.ground_state())

        fid = fidelity(result.states[-1], self.ideal_state) ** 2
        print(
            f"Pulsed qasm sequence Bell state fidelity "
            f"(n = {len(qreg.active_modes)}): {fid:.7f}."
        )
        self.assertLess(1 - fid, 1e-6)

    def test_qasm_measure(self):
        qreg = self.qreg
        qubits = qreg.active_modes
        seq = QasmSequence(qreg)
        seq.h(qubits[-1], unitary=False)
        seq.x(qubits[-2], unitary=False)
        seq.barrier()

        result = seq.run(qreg.ground_state())
        creg = seq.measure(result.states[-1])
        for c in creg[:-2]:
            self.assertAlmostEqual(c, 0)
        self.assertLess(abs(creg[-2] - 1), 1e-5)
        self.assertLess(abs(creg[-1] - 0.5), 5e-3)


if __name__ == "__main__":
    unittest.main()
