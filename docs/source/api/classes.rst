.. sequencing

.. _api-classes:

.. figure:: ../images/sequencing-logo.*
   :alt: seQuencing logo

*******
Classes
*******

.. _api-classes-parameters:

Parameters
----------

Parameterized
*************
.. autoclass:: sequencing.parameters.Parameterized
    :members:

-------------------------------------------------

.. _api-classes-pulses:

Pulses
------

Pulse
*****
.. autoclass:: sequencing.pulses.Pulse
    :members:
    :show-inheritance:

ConstantPulse
*************
.. autoclass:: sequencing.pulses.ConstantPulse
    :members:
    :show-inheritance:

SmoothedConstantPulse
*********************
.. autoclass:: sequencing.pulses.SmoothedConstantPulse
    :members:
    :show-inheritance:

GaussianPulse
*************
.. autoclass:: sequencing.pulses.GaussianPulse
    :members:
    :show-inheritance:

SechPulse
*********
.. autoclass:: sequencing.pulses.SechPulse
    :members:
    :show-inheritance:

SlepianPulse
************
.. autoclass:: sequencing.pulses.SlepianPulse
    :members:
    :show-inheritance:

-------------------------------------------------

.. _api-classes-modes:

Modes
-----

Mode
****
.. autoclass:: sequencing.modes.Mode
    :members:
    :show-inheritance:

PulseMode
*********
.. autoclass:: sequencing.modes.PulseMode
    :members:
    :show-inheritance:

Qubit
*****
.. autoclass:: sequencing.modes.Qubit
    :members:
    :show-inheritance:

Transmon
********
.. autoclass:: sequencing.modes.Transmon
    :members:
    :show-inheritance:

Cavity
******
.. autoclass:: sequencing.modes.Cavity
    :members:
    :show-inheritance:

-------------------------------------------------

.. _api-classes-system:

System
------

CouplingTerm
************
.. autoclass:: sequencing.system.CouplingTerm
    :members:

System
******
.. autoclass:: sequencing.system.System
    :members:
    :show-inheritance:

-------------------------------------------------

.. _api-classes-sequencing:

Sequencing
----------

HamiltonianChannels
*******************
.. autoclass:: sequencing.sequencing.HamiltonianChannels
    :members:

CompiledPulseSequence
*********************
.. autoclass:: sequencing.sequencing.CompiledPulseSequence
    :members:

PulseSequence
*********************
.. autoclass:: sequencing.sequencing.PulseSequence
    :members:
    :show-inheritance:
    :inherited-members:

Sequence
********
.. autoclass:: sequencing.sequencing.main.Sequence
    :members:
    :show-inheritance:
    :inherited-members:

SequenceResult
**************
.. autoclass:: sequencing.sequencing.SequenceResult
    :members:

HTerm
*****
.. autoclass:: sequencing.sequencing.HTerm
    :members:

CTerm
*****
.. autoclass:: sequencing.sequencing.CTerm
    :members:

Operation
*********
.. autoclass:: sequencing.sequencing.Operation
    :members:

SyncOperation
*************
.. autoclass:: sequencing.sequencing.SyncOperation
    :members:

.. seealso::
    :func:`sequencing.sequencing.sync`

DelayOperation
**************
.. autoclass:: sequencing.sequencing.DelayOperation
    :members:

.. seealso::
    :func:`sequencing.sequencing.delay`

DelayChannelsOperation
**********************
.. autoclass:: sequencing.sequencing.DelayChannelsOperation
    :members:

.. seealso::
    :func:`sequencing.sequencing.delay_channels`

ValidatedList
*************
.. autoclass:: sequencing.sequencing.common.ValidatedList
    :members:

-------------------------------------------------

.. _api-classes-gates:

Gates
-----

TwoQubitGate
************
.. autoclass:: sequencing.gates.twoqubit.TwoQubitGate
    :members:

ControlledTwoQubitGate
**********************
.. autoclass:: sequencing.gates.twoqubit.ControlledTwoQubitGate
    :members:
    :show-inheritance:

CUGate
******
.. autoclass:: sequencing.gates.twoqubit.CUGate
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.cu`

CXGate
******
.. autoclass:: sequencing.gates.CXGate
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.cx`

CYGate
******
.. autoclass:: sequencing.gates.CYGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.cy`

CZGate
******
.. autoclass:: sequencing.gates.CZGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.cz`

CPhaseGate
**********
.. autoclass:: sequencing.gates.CPhaseGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.cphase`

SWAPGate
********
.. autoclass:: sequencing.gates.SWAPGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.swap`

SWAPphiGate
***********
.. autoclass:: sequencing.gates.SWAPphiGate
    :members:

    .. seealso::
        :func:`sequencing.gates.swapphi`

iSWAPGate
*********
.. autoclass:: sequencing.gates.iSWAPGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.iswap`

eSWAPGate
*********
.. autoclass:: sequencing.gates.eSWAPGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.eswap`

SqrtSWAPGate
************
.. autoclass:: sequencing.gates.SqrtSWAPGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.sqrtswap`

SqrtiSWAPGate
*************
.. autoclass:: sequencing.gates.SqrtiSWAPGate
    :members:
    :show-inheritance:

    .. seealso::
        :func:`sequencing.gates.sqrtiswap`

-------------------------------------------------

.. _api-classes-qasm:

QASM
----

QasmSequence
************
.. autoclass:: sequencing.qasm.QasmSequence
    :members:
    :show-inheritance:
    :inherited-members:

-------------------------------------------------

.. _api-classes-benchmarking:

Benchmarking
------------

Benchmark
*********
.. autoclass:: sequencing.benchmarking.Benchmark
    :members:
