.. sequencing

.. figure:: images/sequencing-logo.*
   :alt: seQuencing logo

************
Installation
************

This section of the documentation describes the recommended method for installing and verifying ``sequencing``.

We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``sequencing``. Here we call the environment ``sequencing-env``, but you may
call it whatever you like. ``sequencing`` is compatible with Python 3.7, 3.8, and 3.9.

.. code-block:: bash

    (base) conda create -n sequencing-env python=<3.7, 3.8, or 3.9>
    (base) conda activate sequencing-env
    (sequencing-env)

.. important::
    
    **Note for Windows users:**
      `QuTip: Installation on MS Windows <http://qutip.org/docs/latest/installation.html#installation-on-ms-windows>`_: 
      The only supported installation configuration is using the Conda environment with Python 3.5+ and Visual Studio 2015.

Installing with pip
===================

.. code-block:: bash

    (sequencing-env) pip install sequencing


Installing from source
======================

Alternatively, you can install ``sequencing`` from
`GitHub <https://github.com/sequencing-dev/sequencing>`_ using the following commands:

.. code-block:: bash

    (sequencing-env) git clone https://github.com/sequencing-dev/sequencing.git
    (sequencing-env) pip install -e .

Verifying the installation
==========================

To verify your installation by running the ``sequencing`` test suite,
execute the following commands in a Python session:

.. code-block:: python

    >>> import sequencing.testing as st
    >>> st.run()

If you prefer, you can also run the ``sequencing`` tests in a single line:

.. code-block:: bash

    (sequencing-env) python -m sequencing.testing