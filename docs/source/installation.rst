.. sequencing

.. figure:: images/sequencing-logo.*
   :alt: seQuencing logo

************
Installation
************

We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``sequencing``. ``sequencing`` is compatible with, and tested on, Python 3.7, 3.8, and 3.9.

.. code-block:: bash

    conda create -n <env-name> python=<3.7, 3.8, or 3.9>
    conda activate <env-name>

.. important::
    
    **Note for Windows users:**
      `QuTip: Installation on MS Windows <http://qutip.org/docs/latest/installation.html#installation-on-ms-windows>`_: 
      The only supported installation configuration is using the Conda environment with Python 3.5+ and Visual Studio 2015.

Installing with pip
===================

.. code-block:: bash

    pip install sequencing


Installing from source
======================

Alternatively, you can install ``sequencing`` from
`GitHub <https://github.com/sequencing-dev/sequencing>`_ using the following commands:

.. code-block:: bash

    git clone https://github.com/sequencing-dev/sequencing.git
    pip install -e .

Verifying the installation
==========================

To verify your installation by running the ``sequencing`` test suite,
execute the following commands in a Python session:

.. code-block:: python

    >>> import sequencing.testing as st
    >>> st.run()

If you prefer, you can also run the ``sequencing`` tests in a single line:

.. code-block:: bash

    python -m sequencing.testing