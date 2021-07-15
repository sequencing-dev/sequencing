import unittest
import numpy as np
import qutip
from sequencing import Transmon, System, Sequence, sync
from sequencing.calibration import tune_rabi


class TestSequenceVsUnitary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit = Transmon("qubit", levels=2)
        system = System("system", modes=[qubit])
        _ = tune_rabi(system, system.ground_state(), plot=False, verify=False)
        cls.system = system

    def test_sequence_Rx(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()

        angles = np.linspace(-np.pi, np.pi, 11)
        for theta in angles:
            seq = Sequence(system)
            qubit.rotate_x(theta)
            sync()
            unitary = qubit.rotate_x(theta, unitary=True)

            result = seq.run(init_state)
            states = result.states
            fidelity = qutip.fidelity(states[-1], qubit.Rx(theta) * init_state) ** 2
            self.assertGreater(fidelity, 0.999)
            fidelity = qutip.fidelity(states[-1], unitary * init_state) ** 2
            self.assertGreater(fidelity, 0.999)

    def test_sequence_Ry(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()

        angles = np.linspace(-np.pi, np.pi, 11)
        for theta in angles:
            seq = Sequence(system)
            qubit.rotate_y(theta)
            sync()
            unitary = qubit.rotate_y(theta, unitary=True)

            result = seq.run(init_state)
            states = result.states
            fidelity = qutip.fidelity(states[-1], qubit.Ry(theta) * init_state) ** 2
            self.assertGreater(fidelity, 0.999)
            fidelity = qutip.fidelity(states[-1], unitary * init_state) ** 2
            self.assertGreater(fidelity, 0.999)

    def test_sequence_Raxis(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()

        angles = np.linspace(-np.pi, np.pi, 11)
        for theta in angles:
            for phi in angles:
                seq = Sequence(system)
                qubit.rotate(theta, phi)
                sync()
                unitary = qubit.rotate(theta, phi, unitary=True)

                result = seq.run(init_state)
                states = result.states
                fidelity = (
                    qutip.fidelity(states[-1], qubit.Raxis(theta, phi) * init_state)
                    ** 2
                )
                self.assertGreater(fidelity, 0.999)
                fidelity = qutip.fidelity(states[-1], unitary * init_state) ** 2
                self.assertGreater(fidelity, 0.999)

    def test_sequence_Raxis_vs_Rx_Ry(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()

        angles = np.linspace(-np.pi, np.pi, 11)
        for theta in angles:
            # Test Raxis vs. Rx
            seq = Sequence(system)
            qubit.rotate(theta, 0)
            sync()
            unitary = qubit.rotate(theta, 0, unitary=True)

            result = seq.run(init_state)
            states = result.states
            fidelity = qutip.fidelity(states[-1], qubit.Rx(theta) * init_state) ** 2
            self.assertGreater(fidelity, 0.999)

            fidelity = (
                qutip.fidelity(unitary * init_state, qubit.Rx(theta) * init_state) ** 2
            )
            self.assertGreater(fidelity, 0.999)

            # Test Raxis vs. Ry
            seq = Sequence(system)
            qubit.rotate(theta, np.pi / 2)
            sync()
            unitary = qubit.rotate(theta, np.pi / 2, unitary=True)

            result = seq.run(init_state)
            states = result.states
            fidelity = qutip.fidelity(states[-1], qubit.Ry(theta) * init_state) ** 2
            self.assertGreater(fidelity, 0.999)

            fidelity = (
                qutip.fidelity(unitary * init_state, qubit.Ry(theta) * init_state) ** 2
            )
            self.assertGreater(fidelity, 0.999)


if __name__ == "__main__":
    unittest.main()
