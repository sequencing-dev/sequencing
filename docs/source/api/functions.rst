.. sequencing

.. _api-functions:

.. figure:: ../images/sequencing-logo.*
   :alt: seQuencing logo

*********
Functions
*********

.. _api-functions-parameters:

Parameters
----------

StringParameter
***************
.. autofunction:: sequencing.parameters.StringParameter

BoolParameter
*************
.. autofunction:: sequencing.parameters.BoolParameter

IntParameter
************
.. autofunction:: sequencing.parameters.IntParameter

FloatParameter
**************
.. autofunction:: sequencing.parameters.FloatParameter

NanosecondParameter
*******************
.. autofunction:: sequencing.parameters.NanosecondParameter

GigahertzParameter
******************
.. autofunction:: sequencing.parameters.GigahertzParameter

RadianParameter
***************
.. autofunction:: sequencing.parameters.RadianParameter

DictParameter
*************
.. autofunction:: sequencing.parameters.DictParameter

ListParameter
*************
.. autofunction:: sequencing.parameters.ListParameter

-------------------------------------------------

.. _api-functions-modes:

Modes
-----

sort_modes
**********
.. autofunction:: sequencing.modes.sort_modes

-------------------------------------------------

.. _api-functions-pulses:

Pulses
------

array_pulse
***********
.. autofunction:: sequencing.pulses.array_pulse

pulse_factory
*************
.. autofunction:: sequencing.pulses.pulse_factory

-------------------------------------------------

.. _api-functions-sequencing:

Sequencing
----------

ket2dm
******
.. autofunction:: sequencing.sequencing.ket2dm

ops2dms
*******
.. autofunction:: sequencing.sequencing.ops2dms

get_sequence
************
.. autofunction:: sequencing.sequencing.get_sequence

sync
****
.. autofunction:: sequencing.sequencing.sync

.. seealso::
    :class:`sequencing.sequencing.SyncOperation`

delay
*****
.. autofunction:: sequencing.sequencing.delay

.. seealso::
    :class:`sequencing.sequencing.DelayOperation`

delay_channels
**************
.. autofunction:: sequencing.sequencing.delay_channels

.. seealso::
    :class:`sequencing.sequencing.DelayChannelsOperation`

@capture_operation
******************
.. autodecorator:: sequencing.sequencing.capture_operation

-------------------------------------------------

.. _api-functions-gates:

Gates
-----

.. autodecorator:: sequencing.gates.single_qubit_gate

.. autodecorator:: sequencing.gates.pulsed_gate_exists

Single-qubit gates
******************

.. autofunction:: sequencing.gates.rx

.. autofunction:: sequencing.gates.ry

.. autofunction:: sequencing.gates.rz

.. autofunction:: sequencing.gates.x

.. autofunction:: sequencing.gates.y

.. autofunction:: sequencing.gates.z

.. autofunction:: sequencing.gates.h

.. autofunction:: sequencing.gates.r

Two-qubit gates
***************

.. autofunction:: sequencing.gates.cu

    .. seealso::
        :func:`sequencing.gates.twoqubit.CUGate`

.. autofunction:: sequencing.gates.cx

    .. seealso::
        :func:`sequencing.gates.CXGate`

.. autofunction:: sequencing.gates.cy

    .. seealso::
        :func:`sequencing.gates.CYGate`

.. autofunction:: sequencing.gates.cz

    .. seealso::
        :func:`sequencing.gates.CZGate`

.. autofunction:: sequencing.gates.cphase

    .. seealso::
        :func:`sequencing.gates.CPhaseGate`

.. autofunction:: sequencing.gates.swap

    .. seealso::
        :func:`sequencing.gates.SWAPGate`

.. autofunction:: sequencing.gates.swapphi

    .. seealso::
        :func:`sequencing.gates.SWAPphiGate`

.. autofunction:: sequencing.gates.iswap

    .. seealso::
        :func:`sequencing.gates.iSWAPGate`

.. autofunction:: sequencing.gates.eswap

    .. seealso::
        :func:`sequencing.gates.eSWAPGate`

.. autofunction:: sequencing.gates.sqrtswap

    .. seealso::
        :func:`sequencing.gates.SqrtSWAPGate`

.. autofunction:: sequencing.gates.sqrtiswap

    .. seealso::
        :func:`sequencing.gates.SqrtiSWAPGate`


-------------------------------------------------

.. _api-functions-qasm:

QASM
----

parse_qasm_gate
***************
.. autofunction:: sequencing.qasm.parse_qasm_gate

-------------------------------------------------

.. _api-functions-calibration:

Calibration
-----------

tune_rabi
*********
.. autofunction:: sequencing.calibration.tune_rabi

tune_drag
*********
.. autofunction:: sequencing.calibration.tune_drag

tune_displacement
*****************
.. autofunction:: sequencing.calibration.tune_displacement
