# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

"""
Here we adopt the convention that modes are ordered as
    |m_n, m_n-1, ..., m_1, m_0>.
Furthermore, the ordering is decided primarily by mode Hilbert space size:
modes with more levels go on the right.
Among modes with the same number of levels, the ordering is decided
alphanumerically from right to left. For example,
assuming all cavities have the same number of levels
and all qubits have the same, smaller, number of levels:
    |qubit1, qubit0, cavity2, cavity1, cavity0>

For the motivation behind this convention,
see https://arxiv.org/pdf/1711.02086.pdf.
"""

from .modes import Mode, Transmon, Cavity, sort_modes
from .system import System, CouplingTerm
from .benchmarking import Benchmark
from .sequencing import (
    Sequence,
    PulseSequence,
    get_sequence,
    capture_operation,
    sync,
    delay,
    delay_channels,
    HTerm,
    CTerm,
    Operation,
    ket2dm,
    ops2dms,
    tqdm,
    SequenceResult,
)
from .qasm import QasmSequence
from .version import __version__, __version_info__
