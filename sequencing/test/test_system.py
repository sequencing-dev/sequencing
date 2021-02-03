import unittest
import os

import numpy as np
import qutip

from sequencing import System, CouplingTerm, Transmon, Cavity


class TestSerialization(unittest.TestCase):
    def test_to_from_dict(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(qubit, cavity, chi=-2e-3)
        other_system = System.from_dict(system.as_dict())
        self.assertEqual(system, other_system)

    def test_to_from_json_str(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(qubit, cavity, chi=-2e-3)
        json_str = system.to_json(dumps=True)
        other_system = System.from_json(json_str=json_str)
        self.assertEqual(system, other_system)

    def test_to_from_json_file(self):
        json_path = "__test_to_from_json_file.json"
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(qubit, cavity, chi=-2e-3)
        system.to_json(json_path=json_path)
        other_system = System.from_json(json_path=json_path)
        os.remove(json_path)
        self.assertEqual(system, other_system)


class TestSystem(unittest.TestCase):
    def test_get_mode(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])

        self.assertIs(system.get_mode(qubit), qubit)
        self.assertIs(system.get_mode("qubit"), qubit)

        with self.assertRaises(ValueError):
            system.get_mode("other_qubit")

    def test_set_cross_kerr(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])

        with self.assertRaises(ValueError):
            system.set_cross_kerr(qubit, qubit, -1e-3)

        chi = -2e-3
        system.set_cross_kerr(qubit, cavity, chi)

        self.assertEqual(len(system.couplings()), 1)
        self.assertEqual(system.couplings()[0], 2 * np.pi * chi * qubit.n * cavity.n)

        system.set_cross_kerr(qubit, cavity, chi)

        self.assertEqual(len(system.couplings()), 1)
        self.assertEqual(system.couplings()[0], 2 * np.pi * chi * qubit.n * cavity.n)

        system.set_cross_kerr(cavity, qubit, chi)

        self.assertEqual(len(system.couplings()), 1)
        self.assertEqual(system.couplings()[0], 2 * np.pi * chi * qubit.n * cavity.n)

    def test_coupling_terms(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])
        chi = -2e-3
        key = frozenset([qubit.name, cavity.name])
        system.coupling_terms[key].append(
            CouplingTerm(qubit, "n", cavity, "n", strength=2 * np.pi * chi)
        )
        self.assertEqual(len(system.couplings()), 1)
        self.assertEqual(system.couplings()[0], 2 * np.pi * chi * qubit.n * cavity.n)
        system.set_cross_kerr(qubit, cavity, chi)
        self.assertEqual(len(system.couplings()), 1)
        self.assertEqual(system.couplings()[0], 2 * np.pi * chi * qubit.n * cavity.n)

    def test_coupling_terms_multimode(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity1 = Cavity("cavity1", levels=10, kerr=-10e-6)
        cavity2 = Cavity("cavity2", levels=6, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity2, cavity1])
        system.set_cross_kerr(qubit, cavity1, -2e-3)
        system.set_cross_kerr(qubit, cavity2, -1e-3)
        system.set_cross_kerr(cavity1, cavity2, -5e-6)
        self.assertEqual(len(system.couplings()), 3)

        for modes in [[qubit, cavity1], [qubit, cavity1, cavity2]]:
            with system.use_modes(modes):
                self.assertEqual(
                    system.couplings()[0], 2 * np.pi * -2e-3 * qubit.n * cavity1.n
                )

        for i, modes in enumerate([[qubit, cavity2], [qubit, cavity1, cavity2]]):
            with system.use_modes(modes):
                self.assertEqual(
                    system.couplings()[i], 2 * np.pi * -1e-3 * qubit.n * cavity2.n
                )

        for i, modes in enumerate([[cavity2, cavity1], [qubit, cavity1, cavity2]]):
            with system.use_modes(modes):
                self.assertEqual(
                    system.couplings()[2 * i], 2 * np.pi * -5e-6 * cavity1.n * cavity2.n
                )

    def test_order_modes(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)

        system = System("system", modes=[qubit, cavity])
        system.order_modes = True
        system.active_modes = [qubit, cavity]
        self.assertEqual(system.modes, [qubit, cavity])
        self.assertEqual(system.active_modes, [qubit, cavity])
        system.active_modes = [cavity, qubit]
        self.assertEqual(system.modes, [qubit, cavity])
        self.assertEqual(system.active_modes, [qubit, cavity])

        system = System("system", modes=[qubit, cavity])
        system.order_modes = False
        system.active_modes = [qubit, cavity]
        self.assertEqual(system.modes, [qubit, cavity])
        self.assertEqual(system.active_modes, [qubit, cavity])
        system.active_modes = [cavity, qubit]
        self.assertEqual(system.modes, [qubit, cavity])
        self.assertEqual(system.active_modes, [cavity, qubit])

    def test_use_modes(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])

        with system.use_modes([qubit]):
            self.assertEqual(system.ground_state(), qubit.fock(0, full_space=False))

        with system.use_modes([cavity]):
            self.assertEqual(system.ground_state(), cavity.fock(0, full_space=False))

        with system.use_modes([qubit, cavity]):
            self.assertEqual(
                system.ground_state(),
                qutip.tensor(
                    qubit.fock(0, full_space=False), cavity.fock(0, full_space=False)
                ),
            )

        system.order_modes = False
        with system.use_modes([cavity, qubit]):
            self.assertEqual(
                system.ground_state(),
                qutip.tensor(
                    cavity.fock(0, full_space=False), qubit.fock(0, full_space=False)
                ),
            )

    def test_active_modes(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit])

        system.active_modes = [qubit]

        with self.assertRaises(ValueError):
            system.active_modes = [qubit, cavity]

        system.modes = [qubit, cavity]
        system.active_modes = [qubit, cavity]
        system.modes = [cavity]
        with self.assertRaises(ValueError):
            _ = system.active_modes

    def test_fock(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])

        with self.assertRaises(ValueError):
            system.fock(0)

        with self.assertRaises(ValueError):
            system.fock(0, qubit=0)

    def test_H0(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])
        system.set_cross_kerr(qubit, cavity, 0)

        # [qubit.self_kerr, cavity.self_kerr]
        self.assertEqual(len(system.H0(clean=True)), 2)
        # [
        #     qubit.self_kerr, cavity.self_kerr,
        #     qubit.detuning, cavity.detuning,
        #     qubit-cavity cross-Kerr
        # ]
        self.assertEqual(len(system.H0(clean=False)), 5)

        system.set_cross_kerr(qubit, cavity, -2e-3)
        # [qubit.self_kerr, cavity.self_kerr, qubit-cavity cross-Kerr]
        self.assertEqual(len(system.H0(clean=True)), 3)

    def test_c_ops(self):
        qubit = Transmon("qubit", levels=3, kerr=-200e-3)
        cavity = Cavity("cavity", levels=10, kerr=-10e-6)
        system = System("system", modes=[qubit, cavity])

        # [
        #     qubit.decay, qubit.excitation, qubit.dephasing,
        #     cavity.decay, cavity.excitation, cavity.dephasing,
        # ]
        self.assertEqual(len(system.c_ops(clean=False)), 6)
        self.assertEqual(len(system.c_ops(clean=True)), 0)

        qubit.t1 = 100e3
        cavity.t2 = 500e3
        # [qubit.decay, cavity.dephasing]
        self.assertEqual(len(system.c_ops(clean=True)), 2)


if __name__ == "__main__":
    unittest.main()
