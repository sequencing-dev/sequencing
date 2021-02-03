import unittest
import numpy as np
import qutip
from sequencing import (
    Transmon,
    Cavity,
    System,
    get_sequence,
    Sequence,
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


class TestFunctions(unittest.TestCase):
    def test_ket2dm(self):
        ket = qutip.fock(5, 0)
        dm = ket2dm(ket)
        self.assertEqual(dm, qutip.fock_dm(5, 0))

        dm0 = qutip.fock_dm(3, 0)
        dm = ket2dm(dm0)
        self.assertEqual(dm, dm0)

    def test_ops2dms(self):
        ops = [qutip.fock(5, n) for n in range(5)]
        dms = ops2dms(ops)
        for n, dm in enumerate(dms):
            self.assertEqual(dm, qutip.fock_dm(5, n))

        ops = [qutip.fock_dm(5, n) for n in range(5)]
        dms = ops2dms(ops)
        for n, dm in enumerate(dms):
            self.assertEqual(dm, qutip.fock_dm(5, n))


class TestValidatedList(unittest.TestCase):
    def test_valid_types(self):
        class Test(ValidatedList):
            VALID_TYPES = (int, float, complex)

        inst = Test(iterable=[1, 1j, 0.5])

        with self.assertRaises(TypeError):
            inst.append("a string")

        with self.assertRaises(TypeError):
            inst._validate([1, 2, 3])

        inst = Test()

        with self.assertRaises(TypeError):
            inst.append("a string")

        with self.assertRaises(TypeError):
            inst._validate([1, 2, 3])

        inst.extend([0, float("inf"), -1j])

        self.assertEqual(inst.pop(), -1j)


class TestHamiltonianChannels(unittest.TestCase):
    def test_add_channel_operation(self):
        hc = HamiltonianChannels()
        with self.assertRaises(ValueError):
            hc.add_channel("H0")

        hc.add_channel("H0", H=qutip.qeye(3), time_dependent=False)
        with self.assertRaises(ValueError):
            hc.add_channel(
                "H0", H=qutip.qeye(3), time_dependent=False, error_if_exists=True
            )

        with self.assertRaises(ValueError):
            hc.add_operation("H0", t0=0, duration=100, H=None, C_op=None)

        with self.assertRaises(ValueError):
            # ValueError because H0 was defined to be
            # time-independent
            hc.add_operation("H0", t0=0, duration=100, H=qutip.qeye(3))

        with self.assertRaises(ValueError):
            hc.add_operation("H1", H=qutip.qeye(3))

        with self.assertRaises(ValueError):
            hc.delay_channels("H3", 5)

        hc.add_operation("H2", H=qutip.qeye(3), t0=0, duration=10)

        hc.delay_channels("H1", 5)
        hc.add_operation("H1", t0=0, duration=10)

        self.assertEqual(hc.channels["H1"]["delay"], 5)
        self.assertEqual(hc.channels["H2"]["delay"], 0)

        H, C_ops, times = hc.build_hamiltonian()
        self.assertEqual(len(H), 3)
        self.assertEqual(C_ops, [])
        self.assertEqual(times.size, 10 + 5)

        fig, ax = hc.plot_coefficients(subplots=True)
        self.assertIsInstance(ax, np.ndarray)

        fig, ax = hc.plot_coefficients(subplots=False)
        self.assertNotIsInstance(ax, np.ndarray)


class TestPulseSequence(unittest.TestCase):
    def test_pulse_sequence(self):
        qubit = Transmon("qubit", levels=2)
        pulse = qubit.gaussian_pulse
        pulse_len = pulse.sigma * pulse.chop
        system = System("system", modes=[qubit])

        seq = PulseSequence(system=system)
        seq.append(qubit.rotate_x(np.pi, capture=False))
        seq.append(SyncOperation())
        seq.append(qubit.rotate_x(np.pi, capture=False))
        self.assertIsInstance(seq.compile(), CompiledPulseSequence)
        self.assertEqual(seq.compile().hc.times.size, 2 * pulse_len)
        result = seq.run(qubit.fock(0))
        self.assertIsInstance(result, qutip.solver.Result)

    def test_pulse_sequence_propagator(self):
        qubit = Transmon("qubit", levels=2)
        pulse = qubit.gaussian_pulse
        pulse_len = pulse.sigma * pulse.chop
        system = System("system", modes=[qubit])

        seq = PulseSequence(system=system)
        seq.append(qubit.rotate_x(np.pi, capture=False))
        seq.append(SyncOperation())
        seq.append(qubit.rotate_x(np.pi, capture=False))
        seq.append(SyncOperation())
        seq.append(qubit.rotate_x(np.pi, capture=False))
        self.assertIsInstance(seq.compile(), CompiledPulseSequence)
        self.assertEqual(seq.compile().hc.times.size, 3 * pulse_len)
        prop = seq.propagator()
        result = seq.run(qubit.fock(0))
        self.assertIsInstance(prop, np.ndarray)
        for item in prop:
            self.assertIsInstance(item, qutip.Qobj)
        fid = qutip.fidelity(prop[-1] * qubit.fock(0), result.states[-1]) ** 2
        self.assertLess(abs(1 - fid), 5e-9)

    def test_compiled_pulse_sequence(self):
        qubit = Transmon("qubit", levels=2)
        pulse = qubit.gaussian_pulse
        pulse_len = pulse.sigma * pulse.chop
        system = System("system", modes=[qubit])

        seq = CompiledPulseSequence(system=system)
        seq.add_operation(qubit.rotate_x(np.pi, capture=False))
        seq.sync()
        seq.add_operation(qubit.rotate_x(np.pi, capture=False))
        self.assertEqual(seq.hc.times.size, 2 * pulse_len)
        result = seq.run(qubit.fock(0))
        self.assertIsInstance(result, qutip.solver.Result)

    def test_compiled_pulse_sequence_propagator(self):
        qubit = Transmon("qubit", levels=2)
        pulse = qubit.gaussian_pulse
        pulse_len = pulse.sigma * pulse.chop
        system = System("system", modes=[qubit])

        seq = CompiledPulseSequence(system=system)
        seq.add_operation(qubit.rotate_x(np.pi, capture=False))
        seq.sync()
        seq.add_operation(qubit.rotate_x(np.pi, capture=False))
        seq.sync()
        seq.add_operation(qubit.rotate_x(np.pi, capture=False))
        self.assertEqual(seq.hc.times.size, 3 * pulse_len)
        prop = seq.propagator()
        result = seq.run(qubit.fock(0))
        self.assertIsInstance(prop, np.ndarray)
        for item in prop:
            self.assertIsInstance(item, qutip.Qobj)
        fid = qutip.fidelity(prop[-1] * qubit.fock(0), result.states[-1]) ** 2
        self.assertLess(abs(1 - fid), 5e-9)


class TestTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sigma = 10
        chop = 4
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        qubit.gaussian_pulse.sigma = sigma
        qubit.gaussian_pulse.chop = chop
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        cavity.gaussian_pulse.sigma = sigma
        cavity.gaussian_pulse.chop = chop
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(cavity, qubit, chi=-2e-3)
        cls.system = system

    def test_sync(self):
        system = self.system
        qubit = system.qubit
        sigma = qubit.gaussian_pulse.sigma
        chop = qubit.gaussian_pulse.chop

        seq = get_sequence(system)
        qubit.rotate_x(np.pi / 2)
        qubit.rotate_x(np.pi / 2)
        for channel in seq.channels.values():
            self.assertEqual(channel["coeffs"].size, sigma * chop)

        seq = get_sequence(system)
        qubit.rotate_x(np.pi / 2)
        sync()
        qubit.rotate_x(np.pi / 2)
        for channel in seq.channels.values():
            self.assertEqual(channel["coeffs"].size, 2 * sigma * chop)

    def test_delay(self):
        delay_time = 100
        system = self.system
        qubit = system.qubit
        sigma = qubit.gaussian_pulse.sigma
        chop = qubit.gaussian_pulse.chop

        seq = get_sequence(system)
        qubit.rotate_x(np.pi / 2)
        delay(delay_time)
        qubit.rotate_x(np.pi / 2)
        for channel in seq.channels.values():
            self.assertEqual(channel["coeffs"].size, delay_time + 2 * sigma * chop)

    def test_delay_channels(self):
        system = self.system
        qubit = system.qubit
        cavity = system.cavity
        sigma = qubit.gaussian_pulse.sigma
        chop = qubit.gaussian_pulse.chop

        delay_time = 5

        seq = get_sequence(system)
        qubit.rotate_x(np.pi)
        cavity_channels = {"cavity.x": cavity.x, "cavity.y": cavity.y}
        delay_channels(cavity_channels, delay_time)
        cavity.displace(1)
        for channel in seq.channels.values():
            self.assertEqual(channel["coeffs"].size, delay_time + sigma * chop)

        seq = get_sequence(system)
        cavity.displace(1)
        qubit_channels = {"qubit.x": qubit.x, "qubit.y": qubit.y}
        delay_channels(qubit_channels, delay_time)
        qubit.rotate_x(np.pi)
        for channel in seq.channels.values():
            self.assertEqual(channel["coeffs"].size, delay_time + sigma * chop)


class TestSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(cavity, qubit, chi=-2e-3)
        cls.system = system

    def test_sequence(self):
        import matplotlib.pyplot as plt

        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()
        n_rotations = 20
        theta = np.pi / n_rotations

        seq = Sequence(system)
        for _ in range(n_rotations):
            qubit.rotate_x(theta / 2)
            # append the current sequence
            seq.capture()
            # append a unitary
            seq.append(qubit.Rx(theta / 2))

        fig, ax = seq.plot_coefficients()
        plt.pause(1)
        result = seq.run(init_state)
        states = result.states
        fidelity = qutip.fidelity(states[-1], qubit.Rx(np.pi) * init_state) ** 2
        self.assertGreater(fidelity, 0.999)

    def test_hybrid_sequence_operation(self):
        system = self.system
        qubit = system.qubit

        init_state = system.ground_state()
        n_rotations = 20
        theta = np.pi / n_rotations

        seq = Sequence(system)
        for _ in range(n_rotations):
            # append an Operation
            seq.append(qubit.rotate_x(theta / 4, capture=False))
            sync()
            seq.append(qubit.rotate_x(theta / 4, capture=False))
            # append a unitary
            seq.append(qubit.Rx(theta / 4))
            seq.append(qubit.Rx(theta / 4))
        result = seq.run(init_state)
        states = result.states
        self.assertEqual(len(result.states), result.times.size)
        fidelity = qutip.fidelity(states[-1], qubit.Rx(np.pi) * init_state) ** 2
        self.assertGreater(fidelity, 0.9995)


if __name__ == "__main__":
    unittest.main()
