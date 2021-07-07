import unittest
import numpy as np
import qutip
from sequencing import (
    Transmon,
    Cavity,
    System,
    Sequence,
    CTerm,
    Operation,
    capture_operation,
    get_sequence,
    sync,
    delay,
    delay_channels,
    ket2dm,
    ops2dms,
)
from sequencing.sequencing import (
    ValidatedList,
    CompiledPulseSequence,
    PulseSequence,
    SyncOperation,
    HamiltonianChannels,
)


class TestSequenceVsUnitary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit = Transmon("qubit", levels=2)
        system = System("system", modes=[qubit])
        cls.system = system

    def test_sequence_Rx(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()

        seq = Sequence(system)
        qubit.rotate_x(np.pi / 2)
        sync()

        result = seq.run(init_state)
        states = result.states
        fidelity = qutip.fidelity(states[-1], qubit.Rx(np.pi / 2) * init_state) ** 2
        self.assertGreater(fidelity, 0.999)

    def test_sequence_Ry(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()

        seq = Sequence(system)
        qubit.rotate_y(np.pi / 2)
        sync()

        result = seq.run(init_state)
        states = result.states
        fidelity = qutip.fidelity(states[-1], qubit.Ry(np.pi / 2) * init_state) ** 2
        self.assertGreater(fidelity, 0.999)

    def test_sequence_Raxis(self):

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()
        theta = np.pi * 0.456
        phi = -np.pi * 0.567

        seq = Sequence(system)
        qubit.rotate(theta, phi)
        sync()

        result = seq.run(init_state)
        states = result.states
        fidelity = qutip.fidelity(states[-1], qubit.Raxis(theta, phi) * init_state) ** 2
        self.assertGreater(fidelity, 0.999)


if __name__ == "__main__":
    unittest.main()
