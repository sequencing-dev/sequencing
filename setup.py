"""# seQuencing

``sequencing`` is a Python package for simulating realistic quantum control sequences using
[QuTiP](http://qutip.org/docs/latest/index.html). Built for researchers and quantum engineers,
``sequencing`` provides an intuitive framework for constructing models of quantum systems
composed of many modes and generating complex time-dependent control Hamiltonians
for [master equation simulations](http://qutip.org/docs/latest/guide/dynamics/dynamics-master.html).
"""

import os
from setuptools import setup, find_packages

DESCRIPTION = "seQuencing: simulate realistic quantum control sequences using QuTiP"
LONG_DESCRIPTION = __doc__

NAME = "sequencing"
AUTHOR = "Logan Bishop-Van Horn"
AUTHOR_EMAIL = "logan.bvh@gmail.com"
URL = "https://github.com/sequencing-dev/sequencing"
LICENSE = "BSD"
PYTHON_VERSION = ">=3.7"

INSTALL_REQUIRES = [
    "qutip>=4.5",
    "numpy>=1.16",
    "colorednoise>=1.1.1",
    "attrs>=20",
    "matplotlib",
    "scipy",
    "jupyter",
    "tqdm",
    "lmfit",
    "pytest",
    "pytest-cov",
]

EXTRAS_REQUIRE = {
    "docs": [
        "sphinx",
        "sphinx_rtd_theme",
        "nbsphinx",
    ],
}

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
"""

CLASSIFIERS = [line for line in CLASSIFIERS.splitlines() if line]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
KEYWORDS = "quantum pulse sequence"

exec(open("sequencing/version.py").read())

setup(
    name=NAME,
    version=__version__,  # noqa: F821
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    python_requires=PYTHON_VERSION,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
