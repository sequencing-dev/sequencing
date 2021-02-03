# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import inspect
from functools import wraps
from collections import namedtuple

import qutip


def ket2dm(obj):
    """Converts a ``qutip.Qobj`` from a ket
    to a density matrix.
    """
    if obj.isket:
        return qutip.ket2dm(obj)
    return obj


def ops2dms(ops):
    """Converts a ``qutip.Qobj`` or sequence of
    ``qutip.Qobj`` from kets to density matrices.
    """
    if isinstance(ops, qutip.Qobj):
        ops = [ops]
    return list(map(ket2dm, ops))


class ValidatedList(object):
    """A list-like container which is enforced to
    only accept values of the types listed in the class attribute
    ``VALID_TYPES``.

    Args:
        iterable (optional, iterable): Iterable of initial values
            with which to populate the list. Default: None.
    """

    VALID_TYPES = None

    def __init__(self, iterable=None):
        self._items = []
        if iterable is not None:
            self.extend(iterable)

    def _validate(self, item):
        """Enforces that item is an instance of
        one of the types in VALID_TYPES.

        Returns:
            object: The item, if it is a valid type.
        """
        valid_types = self.VALID_TYPES
        if valid_types is not None:
            if not isinstance(item, valid_types):
                raise TypeError(
                    f"{type(self).__name__} expected instance of ["
                    + ", ".join(t.__name__ for t in valid_types)
                    + f"] , but got {type(item)}."
                )
        return item

    def __repr__(self):
        return f"{type(self).__name__}([{','.join(map(repr, self))}])"

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for i in self._items:
            yield i

    def __getitem__(self, item):
        return self._items[item]

    def append(self, item):
        """Add an item to the end of the ValidatedList."""
        item = self._validate(item)
        self._items.append(item)

    def extend(self, iterable):
        """Extend the ValidatedList by appending all the items
        from the iterable.
        """
        iterable = [self._validate(item) for item in iterable]
        self._items.extend(iterable)

    def insert(self, i, item):
        """Insert an item at a given position in the ValidatedList."""
        item = self._validate(item)
        self._items.insert(i, item)

    def pop(self, i=-1):
        """Remove the item at the given position
        in the ValidatedList, and return it.
        """
        return self._items.pop(i)

    def clear(self):
        """Remove all items from the ValidatedList."""
        self._items.clear()


HTerm = namedtuple("HTerm", ["H", "coeffs", "args", "kwargs"], defaults=[1, None, None])
"""A ``namedtuple`` specifying a time-depdendent
Hamiltonian term.

Args:
    H (qutip.Qobj): The Hamiltonian operator.
    coeffs (optional, number, array-like, or callable):
        Time-dependent coefficients for the given operator.
        You can specify single int/float/complex for constant
        coefficients, or provide a function that takes time
        as its first argument and returns coefficiencts.
        Default: None.
    args (optional, iterable): Positional arguments passed
        to coeffs if coeffs is callable. Default: None.
    kwargs (optional, dict): Keyword arguments passed to coeffs
        if coeffs is a function. Default: None.
"""


CTerm = namedtuple(
    "CTerm", ["op", "coeffs", "args", "kwargs"], defaults=[1, None, None]
)
"""A ``namedtuple`` specifying a time-depdendent
collapse operator.

Args:
    op (qutip.Qobj): The collapse operator.
    coeffs (optional, number, array-like, or callable):
        Time-dependenct oefficients for the given operator.
        You can specify single int/float/complex for constant
        coefficients, or provide a function that takes time
        as its first argument and returns coefficiencts.
        Default: None.
    args (optional, iterable): Positional arguments passed
        to coeffs if coeffs is callable. Default: None.
    kwargs (optional, dict): Keyword arguments passed to coeffs
        if coeffs is a function. Default: None.
"""


Operation = namedtuple("Operation", ["duration", "terms"])
"""A ``namedtuple`` specifying a set of
HTerms that are applied simultaneously.

Args:
    duration (int): The number of time points in
        each of the terms, i.e. the duration of the
        operation.
    terms (dict): Dict of (hamiltonian_channel_name, HTerm)
"""


class SyncOperation(object):
    """When inserted into a PulseSequence, ensures that the
    Hamiltonian channels all align up to this point.

    This means that all operations which follow the sync will be
    executed after all those before the sync. Sequences are constructed
    in terms of blocks of operations separated by syncs.
    Within a block, channels are made to have equal duration by padding
    shorter channels to the maximum channel length.
    """

    pass


class DelayOperation(object):
    """When inserted into a PulseSequence, adds a global delay to the sequence,
    delaying all channels by the same amount.

    Args:
        length (int): Length of the delay.
        sync_before (optional, bool): Whether to insert a sync() before
            the delay. Default: True.
        sync_after (optional, bool): Whether to insert a sync() after
            the delay. Default: True.
    """

    def __init__(self, length, sync_before=True, sync_after=True):
        self.length = length
        self.sync_before = sync_before
        self.sync_after = sync_after


class DelayChannelsOperation(object):
    """When inserted into a PulseSequence, adds a delay of
    duration ``length`` to only the channels specified in ``channels``.

    Args:
        channels (str | list | dict): Either the name of a single channel,
            a list of channel names, or a dict of the form
            {channel_name: H_op}, or a dict of the form
            {channel_name: (H_op, C_op)}. One of the latter two is required
            if the channels are not yet defined in seq.channels
            (i.e. if no previous Operations have involved these channels).
        length (int): Duration of the delay.
    """

    def __init__(self, channels, length):
        self.channels = channels
        self.length = length


def get_sequence(system=None, t0=0):
    """Returns the global PulseSequence.

    Args:
        system (optional, System): If system is not None,
            the global PulseSequence is reset. Default: None.

    Returns:
        PulseSequence: The glovbal PulseSequence.
    """
    from .main import _global_pulse_sequence

    if system is not None:
        _global_pulse_sequence.set_system(system, t0=0)
    return _global_pulse_sequence


def capture_operation(func):
    """A decorator used to wrap functions that return an ``Operation``,
    which captures the ``Operation`` and adds it to the
    global ``PulseSequence``.

    If a function that is decorated with ``@capture_operation`` is called with
    the keyword argument ``capture=False``, the ``Operation`` returned by the
    wrapped function will not be captured and added to the global
    ``PulseSequence``, but rather returned as normal.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        sequence = get_sequence()
        if sequence.system is not None:
            # inject the current sequence time into kwargs
            params = inspect.signature(func).parameters
            if (
                "t0" in params
                and params["t0"].default is None
                and kwargs.get("t0", None) is None
            ):
                kwargs["t0"] = sequence.t0
        capture = kwargs.pop("capture", True)
        result = func(*args, **kwargs)
        if capture and isinstance(result, Operation):
            if sequence.system is None:
                raise Exception(
                    "The global PulseSequence is not"
                    "currently associated with a System."
                )
            sequence.append(result)
            return None
        return result

    return wrapped_func


def sync(seq=None):
    """Ensure that the Hamiltonian channels all align up to this point.

    This means that all operations which follow the sync will be
    executed after all those before the sync. Sequences are constructed
    in terms of blocks of operations separated by sync()s.
    Within a block, channels are made to have equal duration by padding
    shorter channels to the maximum channel length.

    Args:
        seq (optional, CompiledPulseSequence): CompiledPulseSequence on
            which to apply the sync. If None, a SyncOperation is appended
            to the global PulseSequence. Default: None.
    """
    if seq is None:
        get_sequence().append(SyncOperation())
    elif seq.system is not None:
        seq.sync()


def delay(length, sync_before=True, sync_after=True, seq=None):
    """Adds a global delay to the sequence,
    delaying all channels by the same amount.

    Args:
        length (int): Length of the delay.
        sync_before (optional, bool): Whether to insert a sync() before
            the delay. Default: True.
        sync_after (optional, bool): Whether to insert a sync() after
            the delay. Default: True.
        seq (optional, CompiledPulseSequence): CompiledPulseSequence on which
            to apply the delay. If None, a DelayOperation is appended to
            the global CompiledPulseSequence. Default: None.
    """
    if seq is None:
        get_sequence().append(DelayOperation(length))
    elif seq.system is not None:
        seq.delay(length, sync_before=sync_before, sync_after=sync_after)


def delay_channels(channels, length, seq=None):
    """Adds a delay of duration `length` to only the channels
    specified in `channels`.

    Args:
        channels (str | list | dict): Either the name of a single channel,
            a list of channel names, or a dict of the form
            {channel_name: H_op}, or a dict of the form
            {channel_name: (H_op, C_op)}. One of the latter two is required
            if the channels are not yet defined in seq.channels
            (i.e. if no previous Operations have involved these channels).
        length (int): Duration of the delay.
        seq (optional, CompiledPulseSequence): CompiledPulseSequence on which
            to delay channels. If None, a DelayChannelsOperation is appended
            to the global CompiledPulseSequence. Default: None.
    """
    if seq is None:
        get_sequence().append(DelayChannelsOperation(channels, length))
    elif seq.system is not None:
        if isinstance(channels, str):
            channels = [channels]
        if isinstance(channels, dict):
            for name, val in channels.items():
                if not isinstance(val, (list, tuple)):
                    val = [val, None]
                H, C = val
                if name not in seq.channels:
                    seq.add_channel(name, H=H, C_op=C, time_dependent=True)
        elif not isinstance(channels, (list, tuple)):
            raise TypeError(
                "Channels must be either a sequence of channel names "
                "or a dict like {name: operator}."
            )
        seq.hc.delay_channels(list(channels), length)
