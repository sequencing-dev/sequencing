.. sequencing documentation master file, created by
   sphinx-quickstart on Tue Dec 29 12:45:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: images/sequencing-logo.*
   :alt: seQuencing logo

**********
seQuencing
**********

`SeQuencing <https://github.com/sequencing-dev/sequencing>`_ is an open-source Python package
for simulating realistic quantum control sequences using
`QuTiP <http://qutip.org/docs/latest/index.html>`_, the Quantum Toolbox in Python.

Built for researchers and quantum engineers, ``sequencing`` provides an intuitive framework
for constructing models of quantum systems composed of many modes and generating complex time-dependent control Hamiltonians
for `simulations of quantum dynamics <http://qutip.org/docs/latest/guide/dynamics/dynamics-master.html>`_.


.. image:: https://img.shields.io/pypi/v/sequencing
   :alt: PyPI

.. image:: https://img.shields.io/github/workflow/status/sequencing-dev/sequencing/lint-and-test/main
   :alt: GitHub Workflow Status (branch)

.. image:: https://readthedocs.org/projects/sequencing/badge/?version=latest
   :target: https://sequencing.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/sequencing-dev/sequencing/branch/main/graph/badge.svg?token=LLABAKBJ0C
   :target: https://codecov.io/gh/sequencing-dev/sequencing

.. image:: https://img.shields.io/github/license/sequencing-dev/sequencing
   :alt: GitHub

.. image:: https://zenodo.org/badge/334427937.svg
   :target: https://zenodo.org/badge/latestdoi/334427937

-----------------------------------------------------------

Motivation
----------

The general problem ``sequencing`` is designed to set up and solve is the time evolution of the state :math:`|\psi(t)\rangle` or density matrix :math:`\rho(t)` of a system composed of :math:`n` oscillators or "modes"---each with its own nonlinearity, dimension, coherence properties, and interactions with other modes---acted upon by realistic time-dependent controls.

The typical time-dependent Hamiltonian constructed using ``sequencing`` has the following form (taking :math:`\hbar=1`):

.. math::
   \hat{H}(t) &= \sum_{i=1}^n \hat{H}_{\mathrm{mode, }i} + \sum_{i\neq j}\hat{H}_{\mathrm{int, }ij} + \sum_{k}\hat{H}_{\mathrm{control, }k}(t)\\
   &= \sum_{i=1}^n \delta_i\hat{a}_i^\dagger\hat{a}_i + \frac{K}{2}(\hat{a}_i^\dagger)^2(\hat{a}_i)^2\\
   &+ \sum_{i\neq j}\chi_{ij}\hat{a}_i^\dagger\hat{a}_i\hat{a}_j^\dagger\hat{a}_j\\
   &+ \sum_{\{\hat{A}_k\}}c_{\hat{A}_k}(t)\hat{A}_k

- :math:`\hat{a}_i` is the mode annihilation (lowering) operator.
- :math:`\delta_i` is the detuning of the mode :math:`i` relative to the chosen rotating frame.
- :math:`K_i` is the Kerr nonlinearity (or self-Kerr) of mode :math:`i`.
- :math:`\chi_{ij}` is the cross-Kerr or dispersive shift between mode :math:`i` and mode :math:`j`.
- :math:`\{\hat{A}_k\}` are a set of Hermitian control operators with complex time-dependent coefficients :math:`c_{\hat{A}_k}(t)`, where each :math:`\hat{A}_k` may be composed of operators acting on one or more modes.

.. note::
   - Although only the first-order Kerr nonlinearity :math:`(\hat{a}_i^\dagger)^2(\hat{a}_i)^2` is included by default, higher-order nonlinearities can easily be added.
   - Although the default interaction term is that of a dispersive coupling, :math:`\hat{H}_{\mathrm{int, }ij}=\chi_{ij}\hat{a}_i^\dagger\hat{a}_i\hat{a}_j^\dagger\hat{a}_j`, any type of two-mode interaction can be included (see :class:`sequencing.system.CouplingTerm`).

The finite coherence of each mode can also be included in the form of Lindblad collapse operators (see the `Lindblad master equation <http://qutip.org/docs/latest/guide/dynamics/dynamics-master.html#the-lindblad-master-equation>`_ section of the QuTiP documentation). The default collapse operators implemented in ``sequencing`` are:

- :math:`\hat{C}_{\uparrow,i} = \sqrt{\gamma_{\uparrow,i}}\,\hat{a}_i^\dagger`, where :math:`\gamma_{\uparrow,i} = p_{\mathrm{therm, }i}T_{1,i}^{-1}` is the mode's energy excitation rate, computed as thermal population divided by :math:`T_1`.
- :math:`\hat{C}_{\downarrow,i} = \sqrt{\gamma_{\downarrow,i}}\,\hat{a}_i`, where :math:`\gamma_{\downarrow,i} = T_{1,i}^{-1} - \gamma_{\uparrow,i}` is the mode's energy decay rate.
- :math:`\hat{C}_{\phi,i} = \sqrt{2\gamma_{\phi,i}}\,\hat{a}_i^\dagger\hat{a}_i`, where :math:`\gamma_{\phi,i} = T_{\phi,i}^{-1} = T_{2,i}^{-1} - (2T_{1,i})^{-1}` is the mode's pure dephasing rate.

Only collapse operators with nonzero coefficients are included in simulations. By default, each mode is set to have ideal coherence properties (:math:`p_{\mathrm{therm, }i}=0`, :math:`T_{1,i}=T_{2,i}=\infty`), meaning that no collapse operators are used and the evolution is unitary.

-----------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation.rst
   notebooks/introduction.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Learn seQuencing

   tutorials/tutorials.rst

.. toctree::
   :maxdepth: 2
   :caption: About seQuencing

   about/license.rst
   about/authors.rst
   about/contributing.rst
   about/acknowledging.rst
   about/changelog.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/classes.rst
   api/functions.rst
