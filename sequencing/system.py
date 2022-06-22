# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import re
import json
from functools import reduce
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import qutip
import attr

from .parameters import Parameterized, ListParameter, DictParameter
from .modes import Mode, sort_modes


class CouplingTerm(object):
    """An object representing a coupling between multiple ``Modes``,
    given by a Hamiltonian term of the form ``strength * product(operators)``.
    If the keyword argument ``add_hc`` is provided and is True,
    then the Hamiltonian term takes the form
    ``strength * (product(operators) + product(operators).dag())``.

    Args:
        terms (list[tuple[Mode, str]]): List of tuples of ``(mode, expr)``,
            which defines the coupling. Each ``expr`` is a string containing an
            algebraic expression involving the Mode's operators.
            See :func:`Mode.operator_expr` for more details.
            The resulting operators are given by ``mode.operator_expr(expr)``.
        strength (optional, float): Coefficient parameterizing the
            strength of the coupling. Strength should be given in
            units of 2 * pi * GHz. Default: 1.
        add_hc (optional, bool): Whether to add the Hermitian conjugate
            of the product of the operators. Default: False.
    """

    def __init__(self, *terms, strength=1, add_hc=False):
        if len(terms) == 1:
            # terms = ([(mode_1, expr_1), ..., (mode_n, expr_n)], )
            # so we want terms[0]
            if not isinstance(terms[0], (list, tuple)):
                raise TypeError(
                    f"Expected a list of (Mode, str), but got {type(terms)}."
                )
            terms = terms[0]
        elif len(terms) == 4 and all(isinstance(item, (Mode, str)) for item in terms):
            # This is the old two-mode only syntax
            mode1, op1_expr, mode2, op2_expr = terms
            terms = [(mode1, op1_expr), (mode2, op2_expr)]
        for mode, expr in terms:
            if not isinstance(mode, Mode):
                raise TypeError(f"Expected instance of Mode, but got {type(mode)}.")
            if not isinstance(expr, str):
                raise TypeError(f"Expected instance of str, but got {type(expr)}.")
        self.terms = list(terms)
        self.strength = float(strength)
        self.add_hc = bool(add_hc)

    @property
    def operators(self):
        return [mode.operator_expr(expr) for mode, expr in self.terms]

    def H(self, strength=None, add_hc=None):
        """Returns the operator representing the coupling term.

        Args:
            strength (optional, float): Coefficient parameterizing the
                strength of the coupling. Strength should be given in units
                of 2 * pi * GHz. Defaults to self.strength.
            add_hc (optional, bool): Whether to add the Hermitian conjugate
                of product of self.operators. Defaults to self.add_hc.

        Returns:
            ``qutip.Qobj``: Operator representing the coupling term.
        """
        if strength is None:
            strength = self.strength
        if add_hc is None:
            add_hc = self.add_hc
        op = reduce(lambda a, b: a * b, self.operators)
        if add_hc:
            op = op + op.dag()
        return strength * op

    def __repr__(self):
        strings = [
            f"{type(self).__name__}([",
            ", ".join([f"({mode.name}, '{expr}')" for mode, expr in self.terms]),
            f"], strength={self.strength:.3e}",
            f", add_hc={self.add_hc})",
        ]
        return "".join(strings)


@attr.s
class System(Parameterized):
    """A collection of ``Modes`` that can be coupled together.

    Attributes:
        modes (list[Mode]): List of all Modes in the system.
        coupling_terms (dict[frozenset[str], list[CouplingTerm]]):
            A dictionary of CouplingTerm objects specifying all
            interactions in the system.
        cross_kerrs (dict[frozenset[str], float]): A dictionary
            of cross-Kerr values in units of GHz.
    """

    modes = ListParameter()
    cross_kerrs = DictParameter()

    order_modes = True

    def initialize(self):
        super().initialize()
        if self.order_modes:
            self.modes = sort_modes(self.modes)
        self._dt = 1
        self.active_modes = self.modes
        self.coupling_terms = defaultdict(list)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        for mode in self.modes:
            mode.dt = dt

    def __getattribute__(self, name):
        # Access modes like system.qubit.
        # System.modes can be changed at any time, so we cannot
        # just setattr(self, name, mode) in initialize().
        try:
            for mode in object.__getattribute__(self, "modes"):
                if mode.name == name:
                    return mode
        except AttributeError:
            pass
        return object.__getattribute__(self, name)

    def get_mode(self, mode):
        """Fetch a mode by name.

        Args:
            mode (str or Mode): Name of Mode to fetch, or the Mode itself.

        Returns:
            Mode: The requested ``Mode``.
        """
        if isinstance(mode, Mode):
            mode = mode.name
        if mode not in [m.name for m in self.modes]:
            raise ValueError(f"{mode} is not a mode of {self.name}.")
        return getattr(self, mode)

    @property
    def levels(self):
        """Dictionary of (mode_name, number_of_levels)"""
        return {mode.name: mode.levels for mode in self.modes}

    @property
    def active_modes(self):
        """List of the modes currently being used."""
        for mode in self._active_modes:
            if mode not in self.modes:
                raise ValueError(
                    f"{mode.name} is not in {self.name}.modes. "
                    f"This likely happened because {self.name}.modes "
                    f"was changed after {self.name} was created."
                    f"Please set {self.name}.active_modes to be "
                    f"a subset of {self.name}.modes."
                )
        return self._active_modes

    @active_modes.setter
    def active_modes(self, modes):
        if isinstance(modes[0], str):
            modes = [getattr(self, m) for m in modes]
        if self.order_modes:
            modes = sort_modes(modes)
        if not all(mode in self.modes for mode in modes):
            raise ValueError("active_modes must be a subset of the system's modes.")
        self._active_modes = modes
        for mode in self._active_modes:
            mode.space = self._active_modes

    @contextmanager
    def use_modes(self, modes):
        """A context manager that temporarily sets ``self.active_modes``
        to ``modes``, then reverts ``self.active_modes``
        to its previous value.

        Args:
            modes (list[Mode]): List of ``Modes`` to temporarily
                assign to ``self.active_modes``.
        """
        if isinstance(modes, (str, Mode)):
            modes = [modes]
        old_modes = self.active_modes
        try:
            self.active_modes = modes
            yield
        finally:
            self.active_modes = old_modes

    @staticmethod
    def tensor(*args):
        """Calculates the tensor product of input operators."""
        return qutip.tensor(*args)

    def I(self, modes=None):  # noqa: E741, E743
        """Identity operator.

        Args:
            modes (optional, list[Mode]): List of Modes to use in
                constructing H0. If None, will use self.active_modes.
                Default: None.

        Returns:
            ``qutip.Qobj``: Identity operator on the Hilbert space
            defined by ``modes``.
        """
        if modes is None:
            modes = self.active_modes
        elif not all(mode in self.modes for mode in modes):
            raise ValueError("modes must be a subset of the system's modes.")
        return self.tensor(*[qutip.qeye(mode.levels) for mode in modes])

    eye = I

    def fock(self, *args, **kwargs):
        """Returns a product state in the Fock basis. States can be
        specified either positionally or as keyword arguments.

        Args:
            *args (tuple): Fock states of Modes in the order of self.modes.
            **kwargs (dict): Fock states of Modes specified as keyword
                arguments, mode_name=n.

        Returns:
            ``qutip.Qobj``: The requested product state.
        """
        if args:
            if kwargs:
                raise ValueError(
                    "If positional arguments are provided, "
                    "no keyword arguments are allowed."
                )
            if len(args) != len(self.active_modes):
                raise ValueError(
                    "The number of positional argument must match "
                    "the number of active modes."
                )
            states = [
                qutip.fock(mode.levels, val)
                for mode, val in zip(self.active_modes, args)
            ]
        else:
            states = [
                qutip.fock(mode.levels, kwargs.get(mode.name, 0))
                for mode in self.active_modes
            ]
        return self.tensor(*states)

    def fock_dm(self, *args, **kwargs):
        """Returns a product state in the Fock basis, as a density matrix.

        States can be specified either positionally or as keyword arguments.

        Args:
            *args (tuple): Fock states of Modes in the order of self.modes.
            **kwargs (dict): Fock states of Modes specified as keyword
                arguments, mode_name=n.

        Returns:
            ``qutip.Qobj``: The requested product state, as a density matrix.
        """
        ket = self.fock(*args, **kwargs)
        return qutip.ket2dm(ket)

    basis = fock

    def ground_state(self):
        """Returns the ground state of the system.

        Returns:
            ``qutip.Qobj``: The system's ground state.
        """
        return self.fock()

    def logical_basis(self, *args, **kwargs):
        """Returns a product state in the basis spanned by the logical states
        of all modes. Logical states can be specified either positionally
        or as keyword arguments.

        Args:
            *args (tuple): Logical states of Modes in the order of self.modes.
            **kwargs (dict): Logical states of Modes specified as keyword
                arguments, mode_name=n.

        Returns:
            ``qutip.Qobj``: The requested product state.
        """
        if args:
            if kwargs:
                raise ValueError(
                    "If positional arguments are provided, "
                    "no keyword arguments are allowed."
                )
            if len(args) != len(self.active_modes):
                raise ValueError(
                    "The number of positional argument must match "
                    "the number of active modes."
                )
            states = [
                mode.logical_states(full_space=False)[val]
                for mode, val in zip(self.active_modes, args)
            ]
        else:
            states = [
                mode.logical_states(full_space=False)[kwargs.get(mode.name, 0)]
                for mode in self.active_modes
            ]
        return self.tensor(*states)

    def set_cross_kerr(self, mode1, mode2, chi=0):
        """Set the cross-Kerr (in GHz) between two modes. Note that the order
        of mode1 and mode2 doesn't matter.

        Args:
            mode1 (Mode or str): Instance of Mode or
                the name of a member of ``self.modes``.
            mode2 (Mode or str): Instance of Mode or
                the name of a member of ``self.modes``.
            chi (optional, float): Cross-Kerr between mode0 and mode1 in GHz.
                Default: 0.
        """
        if isinstance(mode1, str):
            mode1 = self.get_mode(mode1)
        if isinstance(mode2, str):
            mode2 = self.get_mode(mode2)
        if mode1 is mode2:
            raise ValueError("If mode1 is mode2, then it's not a cross-Kerr.")
        key = frozenset([mode1.name, mode2.name])
        # Replace this cross-Kerr if it already exists
        if key in self.coupling_terms:
            for i, coupling_term in enumerate(self.coupling_terms[key][:]):
                subterms = coupling_term.terms
                if (
                    subterms[0][0] is mode1
                    and subterms[0][1] == "n"
                    and subterms[1][0] is mode2
                    and subterms[1][1] == "n"
                ) or (
                    subterms[0][0] is mode2
                    and subterms[0][1] == "n"
                    and subterms[1][0] is mode1
                    and subterms[1][1] == "n"
                ):
                    _ = self.coupling_terms[key].pop(i)
        self.coupling_terms[key].append(
            CouplingTerm([(mode1, "n"), (mode2, "n")], strength=2 * np.pi * chi)
        )
        self.cross_kerrs[key] = chi

    def couplings(self, modes=None, clean=True):
        """Returns all of the static coupling terms in the Hamiltonian.

        Args:
            modes (optional, list[Mode]): List of Modes to use in
                constructing H0. If None, will use self.active_modes.
                Default: None.
            clean (optional, bool): Only keep operators with nonzero elements.
                Default: True.

        Returns:
            list[qutip.Qobj]: List of static coupling terms.
        """
        if modes is None:
            modes = self.active_modes
        mode_names = set(mode.name for mode in modes)
        coupling_terms = []
        with self.use_modes(modes):
            for names, terms in self.coupling_terms.items():
                if set(names).issubset(mode_names):
                    coupling_terms.extend([term.H() for term in terms])
        if clean:
            return [term for term in coupling_terms if term.data.nnz]
        return coupling_terms

    def H0(self, modes=None, clean=True):
        """Returns the static Hamiltonian consisting of all
        self-Kerrs and inter-mode couplings.

        Args:
            modes (optional, list[Mode]): List of Modes to use in
                constructing H0. If None, will use self.active_modes.
                Default: None.
            clean (optional, bool): Only keep operators with nonzero elements.
                Default: True.

        Returns:
            list[qutip.Qobj]: Static Hamiltonian in list form.
        """
        if modes is None:
            modes = self.active_modes
        detunings = [mode.detuning for mode in modes]
        self_kerrs = [mode.self_kerr for mode in modes]
        couplings = self.couplings(modes=modes, clean=clean)
        H0 = detunings + self_kerrs + couplings
        if clean:
            return [H for H in H0 if H.data.nnz]
        return H0

    def c_ops(self, modes=None, clean=True):
        """Returns a list of collapse operators corresponding to
        loss (decay/excitation) and dephasing of all modes.

        Args:
            modes (optional, list[Mode]): List of Modes to use in
                constructing H0. If None, will use self.active_modes.
                Default: None.
            clean (optional, bool): Only keep operators with nonzero elements.
                Default: True.

        Returns:
            list[qutip.Qobj]: List of all collapse operators.
        """
        if modes is None:
            modes = self.active_modes
        decay = [mode.decay for mode in modes]
        excitation = [mode.excitation for mode in modes]
        dephasing = [mode.dephasing for mode in modes]
        c_ops = decay + excitation + dephasing
        if clean:
            return [c for c in c_ops if c.data.nnz]
        return c_ops

    def as_dict(self, json_friendly=False):
        """Overrides Parameterized.as_dict() in order to deal with cross_kerrs.

        Args:
            json_friendly (optional, bool): Whether to return
                a JSON-friendly dictionary. Default:True.

        Returns:
            dict: Dictionary representation of the System object.
        """
        d = super().as_dict(json_friendly=json_friendly)
        cross_kerrs = d.pop("cross_kerrs")
        d["cross_kerrs"] = {}
        if json_friendly:
            # turn frozenset({mode0, mode1}) into '{mode0, mode1}' for json
            for key, val in cross_kerrs.items():
                new_key = "{" + ", ".join(key) + "}"
                d["cross_kerrs"][new_key] = val
        else:
            for key, val in cross_kerrs.items():
                d["cross_kerrs"][key] = val
        return d

    @classmethod
    def from_json(cls, json_path=None, json_str=None):
        """Overrides Parameterized.from_json()
        in order to deal with cross_kerrs.

        Args:
            json_path (optional, str): Path to JSON file from which
                to load parameters. Required if ``json_str`` is ``None``.
                Default: None.
            json_str (optional, str): JSON string like that returned by
                ``self.to_json(dumps=True)``. Required if ``json_path``
                is ``None`` Default: None.

        Returns:
            System: Instance of ``System``
            whose parameters have been populated from the JSON data.
        """

        def json_decode(obj):
            # decode CrossKerr
            for key in list(obj):
                if re.match(r"\{(.*?)\}", key):
                    # turn '{mode0, mode1}' into frozenset({mode0, mode1})
                    k = key.replace("{", "").replace("}", "").split(", ")
                    new_key = frozenset(k)
                    obj[new_key] = obj.pop(key)
            return obj

        if json_str is not None:
            if json_path is not None:
                raise ValueError(
                    "You must provide either json_path " "or json_str, not both."
                )
            d = json.loads(json_str, object_hook=json_decode)
        else:
            if json_path is None:
                raise ValueError("You must provide either json_path or json_str.")
            if not json_path.endswith(".json"):
                json_path = json_path + ".json"
            with open(json_path, "r") as f:
                d = json.load(f, object_hook=json_decode)

        return cls.from_dict(d)
