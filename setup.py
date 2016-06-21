from setuptools import setup, find_packages
import versioneer

install_requires = ['menpo>=0.7,<0.8',
                    'cyrasterize>=0.2.4,<0.3',
                    'mayavi>=4.4.4']

setup(name='menpo3d',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Menpo library providing tools for 3D Computer Vision research',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      package_data={'menpo3d': ['data/*']},
      install_requires=install_requires,
      tests_require=['nose', 'mock']
)
