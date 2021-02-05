# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import os
import pytest
import warnings
import matplotlib


TESTDIR = os.path.join(
    os.path.pardir, os.path.dirname(os.path.abspath(__file__)), "test"
)


def run():
    # We want to temporarily use a non-GUI backend to avoid
    # spamming the user's screen with a bunch of plots.
    # Matplotlib helpfully raises a UserWarning when
    # using a non-GUI backend...
    with warnings.catch_warnings():
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Matplotlib is currently using agg"
        )
        # We also want to ignore DeprecationWarnings becuase
        # they are not directly relevant to the user.
        pytest.main(["-v", TESTDIR, "-W ignore::DeprecationWarning"])
        matplotlib.pyplot.close("all")
        matplotlib.use(old_backend)


if __name__ == "__main__":
    run()
