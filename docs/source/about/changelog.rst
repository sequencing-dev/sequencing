.. sequencing

.. figure:: ../images/sequencing-logo.*
   :alt: seQuencing logo

**********
Change Log
**********

View release history on `PyPI <https://pypi.org/project/sequencing/#history>`_
or `GitHub <https://github.com/sequencing-dev/sequencing/releases>`_.

Version 1.2.0
-------------

Release date: 2022-08-30.

**New Features**
    - Add support for time step ``dt != 1 ns`` (`#21 <https://github.com/sequencing-dev/sequencing/pull/21>`_).


Version 1.1.4
-------------

Release date: 2021-06-22.

**Bug fixes**
    - Fix a bug that prevented sequences with dynamic collapse operators (``CTerms``) from compiling.
    - Add a unit test for the above feature.
  
Version 1.1.3
-------------

Release date: 2021-02-05.

**Bug fixes**
    - Suppress plots and DeprecationWarnings in ``sequencing.testing``.

Version 1.1.2
-------------

Release date: 2021-02-04.

**Bug fixes**
    - Use ``tempfile`` for tests that require file IO, removing the possibility of tests failing due to file permission errors.

Version 1.1.1 (initial release)
-------------------------------

Release date: 2021-02-03.

