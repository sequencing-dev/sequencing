import unittest
import numpy as np
import matplotlib.pyplot as plt

from sequencing import Transmon, Cavity, System, Benchmark, get_sequence, Sequence
from sequencing.calibration import tune_rabi


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qubit = Transmon("qubit", levels=2)
        cavity = Cavity("cavity", levels=6)
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(qubit, cavity, -2e-3)
        with system.use_modes([qubit]):
            _ = tune_rabi(system, init_state=qubit.fock(0), verify=False)

        cls.system = system

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_invalid_sequence(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        _ = get_sequence(system)
        system.qubit.rotate_x(np.pi)

        with self.assertRaises(TypeError):
            _ = Benchmark(system, init_state, target_unitary)

    def test_benchmark(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        seq = get_sequence(system)
        system.qubit.rotate_x(np.pi)

        bm = Benchmark(seq, init_state, target_unitary)

        self.assertAlmostEqual(bm.fidelity(), 1)
        self.assertLess(bm.tracedist(), 1e-5)
        self.assertAlmostEqual(bm.purity(), 1)

    def test_run_sequence_later(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        seq = get_sequence(system)
        system.qubit.rotate_x(np.pi)

        bm = Benchmark(seq, init_state, target_unitary, run_sequence=False)

        self.assertIsNone(bm.mesolve_state)
        self.assertIsNone(bm.fidelity())
        self.assertIsNone(bm.tracedist())
        self.assertIsNone(bm.purity())

        bm.run_sequence()

        self.assertAlmostEqual(bm.fidelity(), 1)
        self.assertLess(bm.tracedist(), 1e-5)
        self.assertAlmostEqual(bm.purity(), 1)

    def test_benchmark_compiled_pulse_sequence(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        seq = get_sequence(system)
        system.qubit.rotate_x(np.pi)

        bm = Benchmark(seq.compile(), init_state, target_unitary)

        self.assertAlmostEqual(bm.fidelity(), 1)
        self.assertLess(bm.tracedist(), 1e-5)
        self.assertAlmostEqual(bm.purity(), 1)

    def test_benchmark_sequence(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        seq = Sequence(system)
        system.qubit.rotate_x(np.pi / 2)
        seq.capture()
        seq.append(system.qubit.rotate_x(np.pi / 2, unitary=True))
        bm = Benchmark(seq, init_state, target_unitary)

        self.assertAlmostEqual(bm.fidelity(), 1)
        self.assertLess(bm.tracedist(), 2e-5)
        self.assertAlmostEqual(bm.purity(), 1)

    def test_plot_wigner(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        seq = get_sequence(system)
        system.qubit.rotate_x(np.pi)

        bm = Benchmark(seq, init_state, target_unitary)

        _fig, _ax = plt.subplots()

        fig, axes = bm.plot_wigners()
        self.assertIsInstance(fig, type(_fig))
        for ax in axes:
            self.assertIsInstance(ax, type(_ax))

    def test_plot_fock_distribution(self):
        system = self.system

        init_state = system.fock()
        target_unitary = system.qubit.rotate_x(np.pi, unitary=True)
        seq = get_sequence(system)
        system.qubit.rotate_x(np.pi)

        bm = Benchmark(seq, init_state, target_unitary)

        _fig, _ax = plt.subplots()

        fig, ax = bm.plot_fock_distribution()
        self.assertIsInstance(fig, type(_fig))
        self.assertIsInstance(ax, type(_ax))


if __name__ == "__main__":
    unittest.main()
