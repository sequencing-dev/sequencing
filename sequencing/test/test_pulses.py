import unittest
import numpy as np
import matplotlib.pyplot as plt

from sequencing.pulses import (
    Pulse,
    ConstantPulse,
    SmoothedConstantPulse,
    GaussianPulse,
    pulse_factory,
)


class TestPulseFactory(unittest.TestCase):
    def test_pulse_factory(self):

        factory = pulse_factory(cls=ConstantPulse)
        pulse = factory()
        self.assertIsInstance(pulse, ConstantPulse)
        self.assertEqual(pulse.name, "constant_pulse")

        factory = pulse_factory(cls=Pulse)
        pulse = factory()
        self.assertIsInstance(pulse, Pulse)
        self.assertEqual(pulse.name, "pulse")

        factory = pulse_factory(cls=GaussianPulse, name="customPulse")
        pulse = factory()
        self.assertIsInstance(pulse, GaussianPulse)
        self.assertEqual(pulse.name, "customPulse")


class TestPulses(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_call(self):
        pulse = Pulse("pulse")
        result = pulse(length=100)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.size, 100)

    def test_shape(self):
        pulse = SmoothedConstantPulse("smooth")

        pulse.sigma = 0
        result = pulse(length=100)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.size, 100)

        pulse.sigma = 10
        pulse.shape = "tanh"
        result = pulse(length=100)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.size, 100)

        pulse.sigma = 10
        pulse.shape = "cos"
        result = pulse(length=100)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.size, 100)

        pulse.sigma = 10
        pulse.shape = "other"
        with self.assertRaises(ValueError):
            result = pulse(length=100)

    def test_plot(self):
        pulse = GaussianPulse("gauss")

        fig, ax = plt.subplots()
        result = pulse.plot(ax=ax)
        self.assertIs(result, ax)

        result = pulse.plot()
        self.assertIsInstance(result, type(ax))


if __name__ == "__main__":
    unittest.main()
