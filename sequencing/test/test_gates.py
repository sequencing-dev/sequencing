import unittest

import numpy as np
import qutip

from sequencing import Transmon, Cavity, System, get_sequence, Operation
from sequencing import gates


class TestSingleQubitGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubits = [Transmon(f"q{i}", levels=2) for i in range(5)][::-1]
        cls.system = System("system", modes=qubits)

    def test_rx(self):
        qubits = self.system.modes

        result = gates.rx(np.pi / 2, *qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Rx(np.pi / 2))

        result = gates.rx(np.pi / 2, *qubits, capture=False)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, Operation) for r in result))
        for r, q in zip(result, qubits):
            expected = q.rotate_x(np.pi / 2, capture=False)
            for channel, info in r.terms.items():
                self.assertTrue(
                    np.array_equal(info.coeffs, expected.terms[channel].coeffs)
                )

        _ = get_sequence(self.system)
        result = gates.rx(np.pi / 2, *qubits)
        self.assertIsNone(result)

        result = gates.rx(np.pi / 2, qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_ry(self):
        qubits = self.system.modes

        result = gates.ry(-np.pi / 2, *qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Ry(-np.pi / 2))

        result = gates.ry(-np.pi / 2, *qubits, capture=False)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, Operation) for r in result))
        for r, q in zip(result, qubits):
            expected = q.rotate_y(-np.pi / 2, capture=False)
            for channel, info in r.terms.items():
                self.assertTrue(
                    np.array_equal(info.coeffs, expected.terms[channel].coeffs)
                )

        _ = get_sequence(self.system)
        result = gates.ry(-np.pi / 2, *qubits)
        self.assertIsNone(result)

        result = gates.ry(-np.pi / 2, qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_rz(self):
        qubits = self.system.modes

        result = gates.rz(np.pi / 4, *qubits)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Rz(np.pi / 4))

        result = gates.rz(np.pi / 4, qubits[0])
        self.assertIsInstance(result, qutip.Qobj)

    def test_x(self):
        qubits = self.system.modes

        result = gates.x(*qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Rx(np.pi))

        result = gates.x(*qubits, capture=False)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, Operation) for r in result))
        for r, q in zip(result, qubits):
            expected = q.rotate_x(np.pi, capture=False)
            for channel, info in r.terms.items():
                self.assertTrue(
                    np.array_equal(info.coeffs, expected.terms[channel].coeffs)
                )

        _ = get_sequence(self.system)
        result = gates.x(*qubits)
        self.assertIsNone(result)

        result = gates.x(qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_y(self):
        qubits = self.system.modes

        result = gates.y(*qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Ry(np.pi))

        result = gates.y(*qubits, capture=False)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, Operation) for r in result))
        for r, q in zip(result, qubits):
            expected = q.rotate_y(np.pi, capture=False)
            for channel, info in r.terms.items():
                self.assertTrue(
                    np.array_equal(info.coeffs, expected.terms[channel].coeffs)
                )

        _ = get_sequence(self.system)
        result = gates.y(*qubits)
        self.assertIsNone(result)

        result = gates.y(qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_z(self):
        qubits = self.system.modes

        result = gates.z(*qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Rz(np.pi))

        result = gates.z(qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_h(self):
        qubits = self.system.modes

        result = gates.h(*qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.hadamard())

        result = gates.h(qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_r(self):
        qubits = self.system.modes

        result = gates.r(np.pi / 8, np.pi / 2, *qubits, unitary=True)
        self.assertEqual(len(result), len(qubits))
        self.assertTrue(all(isinstance(r, qutip.Qobj) for r in result))
        for r, q in zip(result, qubits):
            self.assertEqual(r, q.Raxis(np.pi / 8, np.pi / 2))

        result = gates.r(np.pi / 8, np.pi / 2, qubits[0], unitary=True)
        self.assertIsInstance(result, qutip.Qobj)

    def test_different_spaces(self):
        qubits = self.system.modes
        with self.assertRaises(ValueError):
            with qubits[0].use_space(qubits[0]):
                _ = gates.r(np.pi / 8, np.pi / 2, *qubits, unitary=True)

    def test_unitary_only(self):
        qubits = self.system.modes
        with self.assertRaises(ValueError):
            _ = gates.z(*qubits, unitary=False)

    def test_invalid_mode_type(self):
        cavities = [Cavity(f"c{i}", levels=3, kerr=-2e-3) for i in range(5)]
        _ = System("system", modes=cavities)
        with self.assertRaises(TypeError):
            _ = gates.x(*cavities, unitary=False)


class TestTwoQubitGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit1 = Transmon("q1", levels=2)
        qubit2 = Transmon("q0", levels=2)
        cls.system = System("system", modes=[qubit1, qubit2])

    def test_cu(self):
        control, target = self.system.modes

        theta = np.pi / 5
        phi = 0.1
        lamda = -0.34

        cu = gates.CUGate(control=control, target=target)(theta, phi, lamda)

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, c, -np.exp(1j * lamda) * s],
                [0, 0, np.exp(1j * phi) * s, np.exp(1j * (phi + lamda)) * c],
            ]
        )
        self.assertTrue(np.array_equal(cu.full(), ideal))

        cu = gates.cu(control, target, theta, phi, lamda)
        self.assertTrue(np.array_equal(cu.full(), ideal))

    def test_cx(self):
        control, target = self.system.modes

        cx = gates.CXGate(control=control, target=target)()
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        self.assertTrue(np.array_equal(cx.full(), ideal))

        cx = gates.cx(control, target)
        self.assertTrue(np.array_equal(cx.full(), ideal))

    def test_cy(self):
        control, target = self.system.modes

        cy = gates.CYGate(control=control, target=target)()
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ]
        )
        self.assertTrue(np.array_equal(cy.full(), ideal))

        cy = gates.cy(control, target)
        self.assertTrue(np.array_equal(cy.full(), ideal))

    def test_cz(self):
        control, target = self.system.modes

        cz = gates.CZGate(control=control, target=target)()
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ]
        )
        self.assertTrue(np.array_equal(cz.full(), ideal))

        cz = gates.cz(control, target)
        self.assertTrue(np.array_equal(cz.full(), ideal))

    def test_cphase(self):
        control, target = self.system.modes

        phi = np.pi / 4

        cphase = gates.CPhaseGate(control=control, target=target)(phi)
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * phi)],
            ]
        )
        self.assertTrue(np.array_equal(cphase.full(), ideal))

        cphase = gates.cphase(control, target, phi)
        self.assertTrue(np.array_equal(cphase.full(), ideal))

    def test_swap(self):
        q1, q2 = self.system.modes

        swap = gates.SWAPGate(q1, q2)()
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(swap.full(), ideal))

        swap = gates.swap(q1, q2)
        self.assertTrue(np.array_equal(swap.full(), ideal))

    def test_swapphi(self):
        q1, q2 = self.system.modes
        phi = -np.pi / 4
        swapphi = gates.SWAPphiGate(q1, q2)(phi)
        p = -1j * np.exp(1j * phi)
        m = 1j * np.exp(-1j * phi)
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, m, 0],
                [0, p, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(swapphi.full(), ideal))

        swapphi = gates.swapphi(q1, q2, phi)
        self.assertTrue(np.array_equal(swapphi.full(), ideal))

        swapphi = gates.swapphi(q1, q2, np.pi / 2)
        swap = gates.swap(q1, q2)
        self.assertTrue(np.allclose(swapphi.full(), swap.full()))

    def test_iswap(self):
        q1, q2 = self.system.modes

        iswap = gates.iSWAPGate(q1, q2)()
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.array_equal(iswap.full(), ideal))

        iswap = gates.iswap(q1, q2)
        self.assertTrue(np.array_equal(iswap.full(), ideal))

    def test_eswap(self):
        q1, q2 = self.system.modes

        theta_c = -np.pi / 4
        eswap = gates.eSWAPGate(q1, q2)(theta_c, phi=np.pi / 2)
        swap = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        ideal = np.cos(theta_c / 2) * np.eye(4) - 1j * np.sin(theta_c / 2) * swap
        self.assertTrue(np.allclose(eswap.full(), ideal))

        eswap = gates.eswap(q1, q2, theta_c, phi=np.pi / 2)
        self.assertTrue(np.allclose(eswap.full(), ideal))

    def test_sqrtswap(self):
        q1, q2 = self.system.modes

        sqrtswap = gates.SqrtSWAPGate(q1, q2)()
        p = (1 + 1j) / 2
        m = (1 - 1j) / 2
        ideal = np.array(
            [
                [1, 0, 0, 0],
                [0, p, m, 0],
                [0, m, p, 0],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(sqrtswap.full(), ideal))

        sqrtswap = gates.sqrtswap(q1, q2)
        self.assertTrue(np.allclose(sqrtswap.full(), ideal))

    def test_invalid_mode_type(self):
        with self.assertRaises(TypeError):
            _ = gates.CXGate(0, 1)

    def test_eswap_qubit_order(self):
        q1, q2 = self.system.modes

        theta_c = np.pi / 4

        phi = np.pi / 4
        eswapq1q2 = gates.eswap(q1, q2, theta_c, phi=phi)
        eswapq2q1 = gates.eswap(q2, q1, theta_c, phi=np.pi - phi)
        self.assertEqual(eswapq1q2, eswapq2q1)

    def test_swapphi_qubit_order(self):
        q1, q2 = self.system.modes

        phi = -np.pi / 4
        q1q2 = gates.swapphi(q1, q2, phi)
        q2q1 = gates.swapphi(q2, q1, np.pi - phi)
        self.assertEqual(q1q2, q2q1)


if __name__ == "__main__":
    unittest.main()
