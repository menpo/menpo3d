import os
import platform
import site
from setuptools import setup, find_packages, Extension
import versioneer


SYS_PLATFORM = platform.system().lower()
IS_LINUX = 'linux' in SYS_PLATFORM
IS_OSX = 'darwin' == SYS_PLATFORM
IS_WIN = 'windows' == SYS_PLATFORM
# Get Numpy include path without importing it
NUMPY_INC_PATHS = [os.path.join(r, 'numpy', 'core', 'include')
                   for r in site.getsitepackages() if
                   os.path.isdir(os.path.join(r, 'numpy', 'core', 'include'))]
if len(NUMPY_INC_PATHS) == 0:
    raise ValueError("Could not find numpy include dir - cannot proceed with "
                     "compilation of cython modules.")
elif len(NUMPY_INC_PATHS) > 1:
    print("Found {} numpy include dirs: "
          "{}".format(len(NUMPY_INC_PATHS), ', '.join(NUMPY_INC_PATHS)))
    print("Taking first (highest precedence on path): {}".format(
        NUMPY_INC_PATHS[0]))
NUMPY_INC_PATH = NUMPY_INC_PATHS[0]


# ---- C/C++ EXTENSIONS ---- #
# Stolen (and modified) from the Cython documentation:
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html
def no_cythonize(extensions, **_ignore):
    import os.path as op
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
                if not op.exists(sfile):
                    raise ValueError('Cannot find pre-compiled source file '
                                     '({}) - please install Cython'.format(sfile))
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


def build_extension_from_pyx(pyx_path, extra_sources_paths=None):
    if extra_sources_paths is None:
        extra_sources_paths = []
    extra_sources_paths.insert(0, pyx_path)
    ext = Extension(name=pyx_path[:-4].replace('/', '.'),
                    sources=extra_sources_paths,
                    include_dirs=[NUMPY_INC_PATH],
                    language='c++')
    if IS_LINUX or IS_OSX:
        ext.extra_compile_args.append('-Wno-unused-function')
    return ext

try:
    from Cython.Build import cythonize
except ImportError:
    import warnings
    cythonize = no_cythonize
    warnings.warn('Unable to import Cython - attempting to build using the '
                  'pre-compiled C++ files.')


cython_modules = [
    build_extension_from_pyx('menpo3d/rasterize/tripixel.pyx'),
]
cython_exts = cythonize(cython_modules, quiet=True)

install_requires = ['menpo>=0.8,<0.9',
                    'mayavi>=4.5.0']

setup(name='menpo3d',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Menpo library providing tools for 3D Computer Vision research',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      package_data={'menpo3d': ['data/*']},
      install_requires=install_requires,
      tests_require=['nose', 'mock'],
      ext_modules=cython_exts
      )
