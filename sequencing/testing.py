# This file is part of sequencing.
#
#    Copyright (c) 2021, The Sequencing Authors.
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.

import os
import pytest

TESTDIR = os.path.join(
    os.path.pardir, os.path.dirname(os.path.abspath(__file__)), "test"
)


def run():
    pytest.main(["-v", TESTDIR])


if __name__ == "__main__":
    run()
