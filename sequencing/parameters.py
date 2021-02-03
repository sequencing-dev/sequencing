# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import json
import logging
import importlib
from functools import reduce
from contextlib import contextmanager
import attr


@attr.s
class Parameterized(object):
    """A serializable object with parameters.

    Parameterized objects must have a name and can have any number of
    parameters, which can be created using the functions defined in
    ``sequencing.parameters``, or by using `attrs
    <https://www.attrs.org/en/stable/index.html>`__ directly
    via ``attr.ib()``.

    Parameterized offers the following convenient features:

    * Recursive ``get()`` and ``set()`` methods for getting
      and setting attributes of nested ``Parameterized`` objects.
    * Methods for converting a ``Parameterized`` object into a Python dict,
      and creating a new Parameterized object from a dict.
    * Methods for serializing a ``Parameterized`` object to json
      and creating a new ``Parameterized`` object from json.

    Supported parameter types include:

    * ``StringParameter``
    * ``BoolParameter``
    * ``IntParameter``
    * ``FloatParameter``
    * ``NanosecondParameter``
    * ``GigahertzParameter``
    * ``RadianParameter``
    * ``DictParameter``
    * ``ListParameter``

    **Notes:**

    * Subclasses of ``Parameterized`` must be decorated with ``@attr.s``
    * Subclasses of ``Parameterized`` can define an ``initialize()`` method,
      which takes no arguments. It will be called on instantiation after the
      attrs-generated ``__init__`` method (see `__attrs_post_init__
      <https://www.attrs.org/en/stable/init.html#post-init-hook>`_
      for more details). If defined, the subclass' ``initialize()`` method
      should always call ``super().initialize()`` to ensure that
      the superclass is correctly initialized.

    """

    name = attr.ib(type=str)
    cls = attr.ib(type=str, default="")
    logger = logging.getLogger("Parameterized")

    def initialize(self):
        """Called after the attrs-generated __init__ method.

        Can be specialized to set private attributes or perform other setup tasks.
        """
        pass

    def __attrs_post_init__(self):
        # store the class name so instances can be serialized
        # and de-serialized
        self.cls = ".".join([self.__module__, self.__class__.__name__])
        self.initialize()

    def get_param(self, address, *args, delimiter="."):
        """Recursively "get" a single attribute of nested
        ``Parameterized`` objects.

        Args:
            address (str): ``delimiter``-delimited string specifying the
                attribute to fetch, e.g. ``instance.param.sub_param``.
            delimiter (optional, str): String used to split ``address``.
                Default: ``'.'``.

        Returns:
            object: Attribute specified by ``address``.
        """
        # https://stackoverflow.com/questions/31174295/
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return reduce(_getattr, [self] + address.split(delimiter))

    def set_param(self, address, value, delimiter="."):
        """Recursively "set" a single attribute of nested
        ``Parameterized`` objects.

        Args:
            address (str): ``delimiter``-delimited string specifying the
                attribute to fetch, e.g. ``instance.param.sub_param``.
            value (object): Value to assign to the attribute
                specified by ``address``.
            delimiter (optional, str): String used to split ``address``.
                Default: ".".
        """
        # https://stackoverflow.com/questions/31174295/
        pre, _, post = address.rpartition(delimiter)
        return setattr(self.get_param(pre) if pre else self, post, value)

    def set(self, **kwargs):
        """Recursively "set" attributes of nested ``Parameterized`` objects.

        Attributes must be specified as keyword arguments:
        ``attr_address=value``, where ``attr_address`` is a
        ``delimiter``-delimited string specifying the attribute to fetch,
        e.g. ``instance__param__sub_param=value``. The default ``delimiter``
        is ``"__"``, i.e. two underscores. This can be overridden by passing in
        ``delimiter`` as a keyword argument.
        """
        delimiter = kwargs.pop("delimiter", "__")
        for name, value in kwargs.items():
            self.set_param(name, value, delimiter=delimiter)

    def get(self, *addresses, delimiter="."):
        """Recursively "get" attributes of nested ``Parameterized`` objects.

        Args:
            *names (tuple[str]): Names of the attributes whose values should be
                returned.
            delimiter (optional, str): Delimiter for the attribute addresses.
                Default: "."

        Returns:
            dict[str, object]: A dictionary of (attr_address, attr_value).
        """
        params = {}
        for addr in addresses:
            params[addr] = self.get_param(addr, delimiter=delimiter)
        return params

    @contextmanager
    def temporarily_set(self, **kwargs):
        """A context mangaer that temporarily sets parameter values,
        then reverts them to the old values.

        Delimiter for ``get()`` and ``set()`` can be chosen using keyword
        argument ``delimiter="{whatever}"``.
        The default is two underscores, ``__``.
        """
        delimiter = kwargs.pop("delimiter", "__")
        old_params = self.get(*list(kwargs), delimiter=delimiter)
        try:
            set_kwargs = kwargs.copy()
            set_kwargs["delimiter"] = delimiter
            self.set(**set_kwargs)
            yield
        finally:
            set_kwargs = old_params.copy()
            set_kwargs["delimiter"] = delimiter
            self.set(**set_kwargs)

    def as_dict(self, json_friendly=True):
        """Returns a dictionary representation of the object
        and all of its parameters.

        Args:
            json_friendly (optional, bool): Whether to return
                a JSON-friendly dictionary. Default:True.

        Returns:
            dict: Dictionary representation of the Parameterized object.
        """
        return attr.asdict(self, retain_collection_types=True)

    def to_json(self, dumps=False, json_path=None):
        """Serialize object to json.

        Args:
            dumps (optional, bool): If True, returns the json string
                instead of writing to file. Default: False.
            json_path (optional, str): Path to write json file to.
                Default: ``{self.name}.json``.

        Returns:
            str or None: json string if ``dumps`` is True,
            else writes json to file and returns None.
        """
        d = self.as_dict(json_friendly=True)
        if dumps:
            if json_path is not None:
                raise ValueError("If dumps is True, json_path must be None.")
            return json.dumps(d, indent=2, sort_keys=True)
        if json_path is None:
            json_path = f"{self.name}.json"
        if not json_path.endswith(".json"):
            json_path = json_path + ".json"
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d):
        """Creates a new instance from a dict
        like that returned by ``self.as_dict()``.

        Args:
            d (dict): Dict from which to create the Parameterized object.

        Returns:
            Parameterized: Instance of ``Parameterized``
            whose parameters have been populated from ``d``.
        """
        fields_dict = attr.fields_dict(cls)
        kwargs = {}
        for name, value in d.items():
            if name not in fields_dict:
                # not a Parameter or Parameterized object
                continue
            if isinstance(value, (list, tuple)):
                # value is a list potentially containing both
                # Parameterized objects and
                # non-Parameterized objects
                vals = []
                for val in value:
                    if isinstance(val, dict) and "cls" in val:
                        # val is a Parameterized object
                        mod_name, cls_name = val["cls"].rsplit(".", 1)
                        module = importlib.import_module(mod_name)
                        other_cls = getattr(module, cls_name)
                        vals.append(other_cls.from_dict(val))
                    else:
                        # val is not Parameterized
                        vals.append(val)
                kwargs[name] = vals
            elif isinstance(value, dict):
                # value must be either a Parameterized object itself,
                # or it comes from a DictParameter.
                if "cls" in value:
                    # It's a Parameterized object itself,
                    # so ook up the correct class and instantiate it
                    mod_name, cls_name = value["cls"].rsplit(".", 1)
                    module = importlib.import_module(mod_name)
                    other_cls = getattr(module, cls_name)
                    kwargs[name] = other_cls.from_dict(value)
                else:
                    # It's a dict potentially containing some
                    # Parameterized objects and some non-Parameterized objects
                    val_dict = {}
                    for key, val in value.items():
                        if isinstance(val, dict) and "cls" in val:
                            # val is a Parameterized object
                            mod_name, cls_name = val["cls"].rsplit(".", 1)
                            module = importlib.import_module(mod_name)
                            other_cls = getattr(module, cls_name)
                            val_dict[key] = other_cls.from_dict(val)
                        else:
                            # val is not a Parameterized object
                            val_dict[key] = val
                    kwargs[name] = val_dict
            else:
                # value is not Parameterized, and it is not a list or dict,
                # so it must be a simple scalar parameter.
                kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_path=None, json_str=None):
        """Creates a new instance from a JSON file or string
        like that returned by ``self.to_json()``.

        Args:
            json_path (optional, str): Path to JSON file from which
                to load parameters. Required if ``json_str`` is ``None``.
                Default: None.
            json_str (optional, str): JSON string like that returned by
                ``self.to_json(dumps=True)``. Required if ``json_path``
                is ``None`` Default: None.

        Returns:
            Parameterized: Instance of ``Parameterized``
            whose parameters have been populated from the JSON data.
        """
        if json_str is not None:
            if json_path is not None:
                raise ValueError("Must provide either json_path or json_str, not both.")
            d = json.loads(json_str)
        else:
            if json_path is None:
                raise ValueError("Must provide either json_path or json_str.")
            if not json_path.endswith(".json"):
                json_path = json_path + ".json"
            with open(json_path, "r") as f:
                d = json.load(f)
        return cls.from_dict(d)


def StringParameter(default, **kwargs):
    """Adds a string parameter.

    Args:
        default (str): Default value.
    """
    return attr.ib(default=default, converter=str, **kwargs)


def BoolParameter(default, **kwargs):
    """Adds a boolean parameter.

    Args:
        default (bool): Default value.
    """
    return attr.ib(default=default, converter=bool, **kwargs)


def IntParameter(default, unit=None, **kwargs):
    """Adds an integer parameter.

    Args:
        default (int): Default value.
        unit (optional, str): Unit to record in metadata.
            Default: None.
    """
    if unit is not None:
        kwargs["metadata"] = dict(unit=str(unit))
    return attr.ib(default=default, converter=int, **kwargs)


def FloatParameter(default, unit=None, **kwargs):
    """Adds a float parameter.

    Args:
        default (float): Default value.
        unit (optional, str): Unit to record in metadata.
            Default: None.
    """
    if unit is not None:
        kwargs["metadata"] = dict(unit=str(unit))
    return attr.ib(default=default, converter=float, **kwargs)


def NanosecondParameter(default, base=IntParameter, **kwargs):
    """Adds a nanosecond parameter.

    Args:
        default (int or float): Default value.
        base (optional, type): IntParameter or FloatParameter.
            Default: IntParameter
    """
    return base(default, unit="ns", **kwargs)


def GigahertzParameter(default, base=FloatParameter, **kwargs):
    """Adds a GHz parameter.

    Args:
        default (int or float): Default value.
        base (optional, type): IntParameter or FloatParameter.
            Default: FloatParameter
    """
    return base(default, unit="GHz", **kwargs)


def RadianParameter(default, base=FloatParameter, **kwargs):
    """Add a radian parameter.

    Args:
        default (int or float): Default value.
        base (optional, type): IntParameter or FloatParameter.
            Default: FloatParameter
    """
    return base(default, unit="radian", **kwargs)


def DictParameter(default=None, factory=dict, **kwargs):
    """Adds a dict parameter:

    Args:
        default (optional, dict): Default value. Default: None.
        base (optional, callabe): Factory function, e.g. dict
            or collections.OrderedDict. Default: dict.
    """
    if default is not None:
        return attr.ib(default, **kwargs)
    return attr.ib(factory=factory, **kwargs)


def ListParameter(default=None, factory=list, **kwargs):
    """Adds a list parameter.

    Args:
        default (optional, list): Default value. Default: None.
        base (optional, callabe): Factory function, e.g. list
            or tuple. Default: list.
    """
    if default is not None:
        return attr.ib(default, **kwargs)
    return attr.ib(factory=factory, **kwargs)
