import unittest
import os

import numpy as np
import attr

from sequencing.parameters import (
    Parameterized,
    BoolParameter,
    StringParameter,
    IntParameter,
    FloatParameter,
    NanosecondParameter,
    GigahertzParameter,
    RadianParameter,
)


@attr.s
class Engine(Parameterized):
    cylinders = IntParameter(4)
    displacement = FloatParameter(2, unit="liter")
    current_rpm = FloatParameter(0, unit="rpm")
    turbo_charged = BoolParameter(False)


@attr.s
class Transmission(Parameterized):
    manual = BoolParameter(False)
    num_gears = IntParameter(5)
    current_gear = IntParameter(1)

    def initialize(self):
        super().initialize()
        # Add private attributes in initialize()
        self._is_broken = True

    @property
    def has_clutch(self):
        return self.manual

    def shift_to(self, gear):
        if gear not in range(self.num_gears + 1):
            # 0 is reverse
            raise ValueError(f"Cannot shift into gear {gear}")
        if abs(gear - self.current_gear) > 1:
            raise ValueError("Cannot skip gears")
        self.current_gear = gear


@attr.s
class Car(Parameterized):
    VALID_CHASSIS = ["sedan", "coupe", "hatchback", "suv"]
    chassis = StringParameter("sedan", validator=attr.validators.in_(VALID_CHASSIS))
    num_doors = IntParameter(4, validator=attr.validators.in_([2, 4]))
    miles_per_gallon = FloatParameter(30, unit="mpg")
    engine = attr.ib(factory=lambda: Engine("engine"))
    transmission = attr.ib(factory=lambda: Transmission("transmission"))


class TestSerialization(unittest.TestCase):
    def test_to_from_dict(self):
        car = Car("test")
        other_car = Car.from_dict(car.as_dict())
        self.assertEqual(car, other_car)

    def test_to_from_json_str(self):
        car = Car("test")
        json_str = car.to_json(dumps=True)
        other_car = Car.from_json(json_str=json_str)
        self.assertEqual(car, other_car)

    def test_to_from_json_file(self):
        json_path = "__test_to_from_json_file.json"
        car = Car("test")
        car.to_json(json_path=json_path)
        other_car = Car.from_json(json_path=json_path)
        os.remove(json_path)
        self.assertEqual(car, other_car)


class TestParameters(unittest.TestCase):
    def test_string_parameter(self):
        @attr.s
        class Test(object):
            param = StringParameter("")

        inst = Test(param=1.0)
        self.assertEqual(inst.param, str(1.0))

    def test_bool_parameter(self):
        @attr.s
        class Test(object):
            param = BoolParameter(True)

        inst = Test(param="")
        self.assertFalse(inst.param)

    def test_int_parameter(self):
        @attr.s
        class Test(object):
            param = IntParameter(0)

        inst = Test(param=1.0)
        self.assertIsInstance(inst.param, int)

    def test_float_parameter(self):
        @attr.s
        class Test(object):
            param = FloatParameter(0)

        inst = Test(param="inf")
        self.assertIsInstance(inst.param, float)
        self.assertTrue(np.isinf(inst.param))

    def test_nanosecond_parameter(self):
        @attr.s
        class Test(object):
            param = NanosecondParameter(20, base=FloatParameter)

        inst = Test(param="inf")
        self.assertIsInstance(inst.param, float)
        self.assertTrue(np.isinf(inst.param))
        self.assertEqual(attr.fields(Test).param.metadata["unit"], "ns")

        @attr.s
        class Test(object):
            param = NanosecondParameter(0.5, base=IntParameter)

        inst = Test()
        self.assertIsInstance(inst.param, int)
        self.assertEqual(inst.param, int(0.5))
        self.assertEqual(attr.fields(Test).param.metadata["unit"], "ns")

    def test_gigahertz_parameter(self):
        @attr.s
        class Test(object):
            param = GigahertzParameter(20, base=FloatParameter)

        inst = Test(param="inf")
        self.assertIsInstance(inst.param, float)
        self.assertTrue(np.isinf(inst.param))
        self.assertEqual(attr.fields(Test).param.metadata["unit"], "GHz")

        @attr.s
        class Test(object):
            param = GigahertzParameter(0.5, base=IntParameter)

        inst = Test()
        self.assertIsInstance(inst.param, int)
        self.assertEqual(inst.param, int(0.5))
        self.assertEqual(attr.fields(Test).param.metadata["unit"], "GHz")

    def test_radian_parameter(self):
        @attr.s
        class Test(object):
            param = RadianParameter(np.pi, base=FloatParameter)

        inst = Test(param="0")
        self.assertIsInstance(inst.param, float)
        self.assertEqual(inst.param, 0)
        self.assertEqual(attr.fields(Test).param.metadata["unit"], "radian")

        @attr.s
        class Test(object):
            param = RadianParameter(0.5, base=IntParameter)

        inst = Test()
        self.assertIsInstance(inst.param, int)
        self.assertEqual(inst.param, int(0.5))
        self.assertEqual(attr.fields(Test).param.metadata["unit"], "radian")


if __name__ == "__main__":
    unittest.main()
