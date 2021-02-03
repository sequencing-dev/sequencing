import unittest
import numpy as np
import qutip

from sequencing import Transmon, Cavity, System, get_sequence
from sequencing.calibration import tune_rabi, tune_drag, tune_displacement


class TestRabi(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_rabi_two_levels(self):
        qubit = Transmon("qubit", levels=2)
        system = System("system", modes=[qubit])
        for _ in range(2):
            _, old_amp, new_amp = tune_rabi(system, qubit.fock(0))
        self.assertLess(abs(old_amp - new_amp), 1e-7)

        init = qubit.fock(0)
        seq = get_sequence(system)
        qubit.rotate_x(np.pi)
        result = seq.run(init)

        target = qubit.Rx(np.pi) * init
        fidelity = qutip.fidelity(result.states[-1], target) ** 2
        self.assertLess(abs(1 - fidelity), 1e-10)


class TestDrag(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_drag(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        qubit.gaussian_pulse.sigma = 10
        system = System("system", modes=[qubit])
        for _ in range(3):
            _, old_amp, new_amp = tune_rabi(
                system, qubit.fock(0), plot=False, verify=False
            )
        self.assertLess(abs(old_amp - new_amp), 1e-7)

        _, old_drag, new_drag = tune_drag(system, qubit.fock(0), update=True)
        self.assertNotAlmostEqual(new_drag, 0)

        init = qubit.fock(0)
        seq = get_sequence(system)
        qubit.rotate_x(np.pi)
        result = seq.run(init)

        target = qubit.Rx(np.pi) * init
        fidelity = qutip.fidelity(result.states[-1], target) ** 2
        self.assertLess(abs(1 - fidelity), 1e-5)


class TestDisplacement(unittest.TestCase):
    def test_displacement(self):
        cavity = Cavity("cavity", levels=12)
        system = System("system", modes=[cavity])
        for _ in range(3):
            _, old_amp, new_amp = tune_displacement(system, cavity.fock(0))
        self.assertLess(abs(old_amp - new_amp), 1e-7)

        init = cavity.fock(0)
        seq = get_sequence(system)
        cavity.displace(1 + 2j)
        result = seq.run(init)

        target = cavity.D(1 + 2j) * init
        fidelity = qutip.fidelity(result.states[-1], target) ** 2
        self.assertLess(abs(1 - fidelity), 1e-7)


if __name__ == "__main__":
    unittest.main()
