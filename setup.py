import os
from setuptools import setup, find_packages

short_description = (
    "seQuencing: simulate realistic quantum control sequences using QuTiP"
)
readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
if os.path.isfile(readme):
    description = open(readme).read()
else:
    description = short_description


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

exec(open("sequencing/version.py").read())

setup(
    name="sequencing",
    version=__version__,  # noqa: F821
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude=["test.*", "test"]),
    long_description=description,
    description=short_description,
    include_package_data=True,
    python_requires=PYTHON_VERSION,
    install_requires=INSTALL_REQUIRES,
)
