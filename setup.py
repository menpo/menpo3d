"""Only the Cython extension is built here; everything else is in pyproject.toml."""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extension = Extension(
    "menpo3d.rasterize.tripixel",
    sources=["menpo3d/rasterize/tripixel.pyx"],
    include_dirs=[np.get_include()],
    language="c++",
)
if sys.platform in ("linux", "darwin"):
    extension.extra_compile_args.append("-Wno-unused-function")
if sys.platform == "darwin":
    extension.extra_link_args.append("-headerpad_max_install_names")

setup(ext_modules=cythonize([extension], quiet=True))
