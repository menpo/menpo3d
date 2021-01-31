import os
import platform
import site

from setuptools import Extension, find_packages, setup

SYS_PLATFORM = platform.system().lower()
IS_LINUX = "linux" in SYS_PLATFORM
IS_OSX = "darwin" == SYS_PLATFORM
IS_WIN = "windows" == SYS_PLATFORM

# Get Numpy include path without importing it
NUMPY_INC_PATHS = [
    os.path.join(r, "numpy", "core", "include")
    for r in site.getsitepackages()
    if os.path.isdir(os.path.join(r, "numpy", "core", "include"))
]
if len(NUMPY_INC_PATHS) == 0:
    try:
        import numpy as np
    except ImportError:
        raise ValueError(
            "Could not find numpy include dir and numpy not installed before build - "
            "cannot proceed with compilation of cython modules."
        )
    else:
        # just ask numpy for it's include dir
        NUMPY_INC_PATHS = [np.get_include()]

elif len(NUMPY_INC_PATHS) > 1:
    print(
        "Found {} numpy include dirs: "
        "{}".format(len(NUMPY_INC_PATHS), ", ".join(NUMPY_INC_PATHS))
    )
    print("Taking first (highest precedence on path): {}".format(NUMPY_INC_PATHS[0]))
NUMPY_INC_PATH = NUMPY_INC_PATHS[0]


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


# ---- C/C++ EXTENSIONS ---- #
# Stolen (and modified) from the Cython documentation:
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html
def no_cythonize(extensions, **_ignore):
    import os.path as op

    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
                if not op.exists(sfile):
                    raise ValueError(
                        "Cannot find pre-compiled source file "
                        "({}) - please install Cython".format(sfile)
                    )
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


def build_extension_from_pyx(pyx_path, extra_sources_paths=None):
    if extra_sources_paths is None:
        extra_sources_paths = []
    extra_sources_paths.insert(0, pyx_path)
    ext = Extension(
        name=pyx_path[:-4].replace("/", "."),
        sources=extra_sources_paths,
        include_dirs=[NUMPY_INC_PATH],
        language="c++",
    )
    if IS_LINUX or IS_OSX:
        ext.extra_compile_args.append("-Wno-unused-function")
    if IS_OSX:
        ext.extra_link_args.append("-headerpad_max_install_names")
    return ext


try:
    from Cython.Build import cythonize
except ImportError:
    import warnings

    cythonize = no_cythonize
    warnings.warn(
        "Unable to import Cython - attempting to build using the "
        "pre-compiled C++ files."
    )

cython_modules = [
    build_extension_from_pyx("menpo3d/rasterize/tripixel.pyx"),
]
cython_exts = cythonize(cython_modules, quiet=True)

version, cmdclass = get_version_and_cmdclass("menpo3d")

install_requires = ["menpo>=0.9.0,<0.12.0", "mayavi>=4.7.0", "moderngl>=5.6.*,<6.0"]

setup(
    name="menpo3d",
    version=version,
    cmdclass=cmdclass,
    description="Menpo library providing tools for 3D Computer Vision research",
    author="James Booth",
    author_email="james.booth08@imperial.ac.uk",
    packages=find_packages(),
    package_data={
        "menpo3d": [
            "data/*",
            "rasterize/shaders/per_vertex/*",
            "rasterize/shaders/texture/*",
        ]
    },
    install_requires=install_requires,
    tests_require=["pytest>=5.0"],
    ext_modules=cython_exts,
)
