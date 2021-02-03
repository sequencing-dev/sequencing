import unittest
import numpy as np
import qutip

from sequencing import Mode, Transmon, Cavity, System, get_sequence
from sequencing.calibration import tune_rabi, tune_displacement


class TestMode(unittest.TestCase):
    def test_basis(self):
        mode0 = Mode("mode0", levels=5)
        state = mode0.basis(0, full_space=False)
        self.assertEqual([[mode0.levels], [1]], state.dims)

        mode1 = Mode("mode1", levels=10)
        mode0.space = mode1.space = [mode1, mode0]
        state = mode0.basis(0, full_space=False)
        self.assertEqual([[mode0.levels], [1]], state.dims)

        state = mode0.basis(0, full_space=True)
        self.assertEqual([[mode1.levels, mode0.levels], [1, 1]], state.dims)

    def test_order_modes(self):
        mode0 = Mode("mode0", levels=5)
        mode1 = Mode("mode1", levels=10)

        Mode.order_modes = True
        mode0.space = [mode1, mode0]
        self.assertEqual(mode0.space, [mode1, mode0])

        mode0.space = [mode0, mode1]
        self.assertEqual(mode0.space, [mode1, mode0])

        Mode.order_modes = False
        mode0.space = [mode1, mode0]
        self.assertEqual(mode0.space, [mode1, mode0])

        mode0.space = [mode0, mode1]
        self.assertEqual(mode0.space, [mode0, mode1])

        Mode.order_modes = True

    def test_use_space(self):
        mode0 = Mode("mode0", levels=5)
        mode1 = Mode("mode1", levels=10)

        mode0.space = [mode1, mode0]
        with mode0.use_space([mode0]):
            self.assertEqual(mode0.space, [mode0])
        self.assertEqual(mode0.space, [mode1, mode0])

    def test_no_loss(self):
        mode = Mode("mode", levels=5)
        t1 = 100e3
        t2 = 200e3
        thermal_population = 0.05
        mode.t1 = t1
        mode.t2 = t2
        mode.thermal_population = thermal_population

        with mode.no_loss():
            self.assertTrue(np.isinf(mode.t1))
            self.assertTrue(np.isinf(mode.t2))
            self.assertEqual(mode.thermal_population, 0)

        self.assertEqual(mode.t1, t1)
        self.assertEqual(mode.t2, t2)
        self.assertEqual(mode.thermal_population, thermal_population)

    def test_operator_expr(self):
        mode = Mode("mode", levels=2)
        self.assertEqual(mode.operator_expr("a.dag() * a"), mode.n)
        self.assertEqual(mode.operator_expr("Rx(pi/2)"), mode.Rx(np.pi / 2))
        self.assertEqual(mode.operator_expr("Rx(pi/2)**2"), mode.Rx(np.pi))


class TestTransmon(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_paulis(self):
        q0 = Transmon("q0", levels=2)
        q1 = Transmon("q1", levels=3, kerr=-200e-3)

        q0_0 = q0.basis(0, full_space=False)
        q0_1 = q0.basis(1, full_space=False)
        q1_0 = q1.basis(0, full_space=False)
        q1_1 = q1.basis(1, full_space=False)
        q0_I = q0.I
        q1_I = q1.I

        single_mode_paulis = {
            "X0": q0_0 * q0_1.dag() + q0_1 * q0_0.dag(),
            "X1": q1_0 * q1_1.dag() + q1_1 * q1_0.dag(),
            "Y0": -1j * (q0_0 * q0_1.dag() - q0_1 * q0_0.dag()),
            "Y1": -1j * (q1_0 * q1_1.dag() - q1_1 * q1_0.dag()),
            "Z0": q0_0 * q0_0.dag() - q0_1 * q0_1.dag(),
            "Z1": q1_0 * q1_0.dag() - q1_1 * q1_1.dag(),
        }

        q0.space = q1.space = [q1, q0]
        self.assertEqual(q0.sigmax(full_space=False), single_mode_paulis["X0"])
        self.assertEqual(q0.sigmay(full_space=False), single_mode_paulis["Y0"])
        self.assertEqual(q0.sigmaz(full_space=False), single_mode_paulis["Z0"])

        self.assertEqual(q1.sigmax(full_space=False), single_mode_paulis["X1"])
        self.assertEqual(q1.sigmay(full_space=False), single_mode_paulis["Y1"])
        self.assertEqual(q1.sigmaz(full_space=False), single_mode_paulis["Z1"])

        self.assertEqual(
            q0.sigmax(full_space=True), qutip.tensor(q1_I, single_mode_paulis["X0"])
        )
        self.assertEqual(
            q0.sigmay(full_space=True), qutip.tensor(q1_I, single_mode_paulis["Y0"])
        )
        self.assertEqual(
            q0.sigmaz(full_space=True), qutip.tensor(q1_I, single_mode_paulis["Z0"])
        )

        self.assertEqual(
            q1.sigmax(full_space=True), qutip.tensor(single_mode_paulis["X1"], q0_I)
        )
        self.assertEqual(
            q1.sigmay(full_space=True), qutip.tensor(single_mode_paulis["Y1"], q0_I)
        )
        self.assertEqual(
            q1.sigmaz(full_space=True), qutip.tensor(single_mode_paulis["Z1"], q0_I)
        )

    def test_Rphi(self):
        q0 = Transmon("q0", levels=2)
        q1 = Transmon("q1", levels=3, kerr=-200e-3)
        q0_I = q0.I
        q1_I = q1.I
        q0_n = q0.n
        q1_n = q1.n

        q0.space = q1.space = [q1, q0]

        self.assertEqual(q0.Rphi(np.pi, full_space=False), (1j * np.pi * q0_n).expm())
        self.assertEqual(q1.Rphi(np.pi, full_space=False), (1j * np.pi * q1_n).expm())

        self.assertEqual(
            q0.Rphi(np.pi, full_space=True),
            qutip.tensor(q1_I, (1j * np.pi * q0_n).expm()),
        )
        self.assertEqual(q0.Rphi(np.pi, full_space=True), (1j * np.pi * q0.n).expm())
        self.assertEqual(
            q1.Rphi(np.pi, full_space=True),
            qutip.tensor((1j * np.pi * q1_n).expm(), q0_I),
        )
        self.assertEqual(q1.Rphi(np.pi, full_space=True), (1j * np.pi * q1.n).expm())

    def test_rotations_unitary(self):
        q0 = Transmon("q0", levels=2)
        q1 = Transmon("q1", levels=3)

        for full_space in [True, False]:
            self.assertEqual(
                q0.rotate_x(np.pi / 2, unitary=True, full_space=full_space),
                q0.Rx(np.pi / 2, full_space=full_space),
            )
            self.assertEqual(
                q0.rotate_y(np.pi / 2, unitary=True, full_space=full_space),
                q0.Ry(np.pi / 2, full_space=full_space),
            )
            self.assertEqual(
                q1.rotate_x(np.pi / 2, unitary=True, full_space=full_space),
                q1.Rx(np.pi / 2, full_space=full_space),
            )
            self.assertEqual(
                q1.rotate_y(np.pi / 2, unitary=True, full_space=full_space),
                q1.Ry(np.pi / 2, full_space=full_space),
            )

        q0.space = q1.space = [q1, q0]

        for full_space in [True, False]:
            self.assertEqual(
                q0.rotate_x(np.pi / 2, unitary=True, full_space=full_space),
                q0.Rx(np.pi / 2, full_space=full_space),
            )
            self.assertEqual(
                q0.rotate_y(np.pi / 2, unitary=True, full_space=full_space),
                q0.Ry(np.pi / 2, full_space=full_space),
            )
            self.assertEqual(
                q1.rotate_x(np.pi / 2, unitary=True, full_space=full_space),
                q1.Rx(np.pi / 2, full_space=full_space),
            )
            self.assertEqual(
                q1.rotate_y(np.pi / 2, unitary=True, full_space=full_space),
                q1.Ry(np.pi / 2, full_space=full_space),
            )

    def test_rotations_pulse(self):
        q0 = Transmon("q0", levels=2)
        q1 = Transmon("q1", levels=3, kerr=-200e-3)
        q0.gaussian_pulse.sigma = 40
        q1.gaussian_pulse.sigma = 40
        system = System("system", modes=[q0, q1])
        init_state = system.fock()

        for qubit in [q0, q1]:
            for _ in range(1):
                _ = tune_rabi(
                    system, init_state=init_state, mode_name=qubit.name, verify=False
                )

        angles = np.linspace(-np.pi, np.pi, 5)
        for angle in angles:
            for qubit in [q0, q1]:
                seq = get_sequence(system)
                qubit.rotate_x(angle)
                unitary = qubit.rotate_x(angle, unitary=True)
                result = seq.run(init_state)
                fidelity = qutip.fidelity(result.states[-1], unitary * init_state) ** 2
                self.assertGreater(fidelity, 1 - 1e-2)

                seq = get_sequence(system)
                qubit.rotate_y(angle)
                unitary = qubit.rotate_y(angle, unitary=True)
                result = seq.run(init_state)
                fidelity = qutip.fidelity(result.states[-1], unitary * init_state) ** 2
                self.assertGreater(fidelity, 1 - 1e-2)


class TestCavity(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_paulis_fock(self):
        q0 = Cavity("q0", levels=6)
        q1 = Cavity("q1", levels=10)

        q0_0 = q0.basis(0, full_space=False)
        q0_1 = q0.basis(1, full_space=False)
        q1_0 = q1.basis(0, full_space=False)
        q1_1 = q1.basis(1, full_space=False)
        q0_I = q0.I
        q1_I = q1.I

        single_mode_paulis = {
            "X0": q0_0 * q0_1.dag() + q0_1 * q0_0.dag(),
            "X1": q1_0 * q1_1.dag() + q1_1 * q1_0.dag(),
            "Y0": -1j * (q0_0 * q0_1.dag() - q0_1 * q0_0.dag()),
            "Y1": -1j * (q1_0 * q1_1.dag() - q1_1 * q1_0.dag()),
            "Z0": q0_0 * q0_0.dag() - q0_1 * q0_1.dag(),
            "Z1": q1_0 * q1_0.dag() - q1_1 * q1_1.dag(),
        }

        q0.space = q1.space = [q1, q0]
        self.assertEqual(q0.sigmax(full_space=False), single_mode_paulis["X0"])
        self.assertEqual(q0.sigmay(full_space=False), single_mode_paulis["Y0"])
        self.assertEqual(q0.sigmaz(full_space=False), single_mode_paulis["Z0"])

        self.assertEqual(q1.sigmax(full_space=False), single_mode_paulis["X1"])
        self.assertEqual(q1.sigmay(full_space=False), single_mode_paulis["Y1"])
        self.assertEqual(q1.sigmaz(full_space=False), single_mode_paulis["Z1"])

        self.assertEqual(
            q0.sigmax(full_space=True), qutip.tensor(q1_I, single_mode_paulis["X0"])
        )
        self.assertEqual(
            q0.sigmay(full_space=True), qutip.tensor(q1_I, single_mode_paulis["Y0"])
        )
        self.assertEqual(
            q0.sigmaz(full_space=True), qutip.tensor(q1_I, single_mode_paulis["Z0"])
        )

        self.assertEqual(
            q1.sigmax(full_space=True), qutip.tensor(single_mode_paulis["X1"], q0_I)
        )
        self.assertEqual(
            q1.sigmay(full_space=True), qutip.tensor(single_mode_paulis["Y1"], q0_I)
        )
        self.assertEqual(
            q1.sigmaz(full_space=True), qutip.tensor(single_mode_paulis["Z1"], q0_I)
        )

    def test_paulis_custom_logical_states(self):
        c0 = Cavity("c0", levels=6)
        c1 = Cavity("c1", levels=10)

        c0.set_logical_states(c0.basis(0), c0.basis(1))
        c1.set_logical_states(c1.basis(2), c1.basis(3))

        c0_0 = c0.logical_zero(full_space=False)
        c0_1 = c0.logical_one(full_space=False)
        c1_0 = c1.logical_zero(full_space=False)
        c1_1 = c1.logical_one(full_space=False)
        c0_I = c0.I
        c1_I = c1.I

        single_mode_paulis = {
            "X0": c0_0 * c0_1.dag() + c0_1 * c0_0.dag(),
            "X1": c1_0 * c1_1.dag() + c1_1 * c1_0.dag(),
            "Y0": -1j * (c0_0 * c0_1.dag() - c0_1 * c0_0.dag()),
            "Y1": -1j * (c1_0 * c1_1.dag() - c1_1 * c1_0.dag()),
            "Z0": c0_0 * c0_0.dag() - c0_1 * c0_1.dag(),
            "Z1": c1_0 * c1_0.dag() - c1_1 * c1_1.dag(),
        }

        c0.space = c1.space = [c1, c0]
        self.assertEqual(c0.sigmax(full_space=False), single_mode_paulis["X0"])
        self.assertEqual(c0.sigmay(full_space=False), single_mode_paulis["Y0"])
        self.assertEqual(c0.sigmaz(full_space=False), single_mode_paulis["Z0"])

        self.assertEqual(c1.sigmax(full_space=False), single_mode_paulis["X1"])
        self.assertEqual(c1.sigmay(full_space=False), single_mode_paulis["Y1"])
        self.assertEqual(c1.sigmaz(full_space=False), single_mode_paulis["Z1"])

        self.assertEqual(
            c0.sigmax(full_space=True), qutip.tensor(c1_I, single_mode_paulis["X0"])
        )
        self.assertEqual(
            c0.sigmay(full_space=True), qutip.tensor(c1_I, single_mode_paulis["Y0"])
        )
        self.assertEqual(
            c0.sigmaz(full_space=True), qutip.tensor(c1_I, single_mode_paulis["Z0"])
        )

        self.assertEqual(
            c1.sigmax(full_space=True), qutip.tensor(single_mode_paulis["X1"], c0_I)
        )
        self.assertEqual(
            c1.sigmay(full_space=True), qutip.tensor(single_mode_paulis["Y1"], c0_I)
        )
        self.assertEqual(
            c1.sigmaz(full_space=True), qutip.tensor(single_mode_paulis["Z1"], c0_I)
        )

    def test_invalid_logical_state(self):
        c0 = Cavity("c0", levels=6)
        c1 = Cavity("c1", levels=10)

        c0.set_logical_states(c0.basis(0), c0.basis(1))
        c1.set_logical_states(c1.basis(2), c1.basis(3))

        c0.space = c1.space = [c1, c0]

        with self.assertRaises(ValueError):
            c0.set_logical_states(c0.basis(0), c0.basis(1))

        with self.assertRaises(ValueError):
            c1.set_logical_states(c1.basis(2), c1.basis(3))

    def test_set_logical_states_none(self):
        c0 = Cavity("c0", levels=6)
        c1 = Cavity("c1", levels=10)
        c0.set_logical_states(None, None)
        c1.set_logical_states(None, None)

        self.assertEqual(c0.logical_zero(), c0.basis(0))
        self.assertEqual(c0.logical_one(), c0.basis(1))

        self.assertEqual(c1.logical_zero(), c1.basis(0))
        self.assertEqual(c1.logical_one(), c1.basis(1))

    def test_displacement(self):
        c0 = Cavity("c0", levels=10, kerr=-10e-6)
        c1 = Cavity("c1", levels=12, kerr=-10e-6)
        system = System("system", modes=[c0, c1])
        init_state = system.fock()
        for cavity in [c0, c1]:
            for _ in range(1):
                _ = tune_displacement(
                    system, init_state, mode_name=cavity.name, verify=False
                )
            for alpha in [0, 1, -1, 1j, -1j, 2, -2, 2j, -2j]:
                ideal_state = cavity.tensor_with_zero(
                    qutip.coherent(cavity.levels, alpha)
                )
                seq = get_sequence(system)
                unitary = cavity.displace(alpha, unitary=True)
                cavity.displace(alpha)
                result = seq.run(init_state)
                unitary_fidelity = (
                    qutip.fidelity(unitary * init_state, ideal_state) ** 2
                )
                pulse_fidelity = qutip.fidelity(result.states[-1], ideal_state) ** 2
                self.assertGreater(unitary_fidelity, 1 - 1e-4)
                self.assertGreater(pulse_fidelity, 1 - 1e-4)


if __name__ == "__main__":
    unittest.main()
