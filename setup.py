from setuptools import setup, find_packages
import sys
import versioneer

project_name = 'menpo3d'

# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.
versioneer.VCS = 'git'
versioneer.versionfile_source = '{}/_version.py'.format(project_name)
versioneer.versionfile_build = '{}/_version.py'.format(project_name)
versioneer.tag_prefix = 'v'  # tags are like v1.2.0
versioneer.parentdir_prefix = project_name + '-'  # dirname like 'menpo-v1.2.0'

install_requires = ['menpo==0.4.0a3',
                    'cyassimp==0.2.0',
                    'cyrasterize==0.2.2']

# These dependencies currently don't work on Python 3
if sys.version_info.major == 2:
    install_requires.append('mayavi==4.3.1')
    install_requires.append('menpo-pyvrml97==2.3.0a4')

setup(name=project_name,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='MenpoKit providing tools for 3D Computer Vision research',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      package_data={'menpo3d': ['data/*']},
      install_requires=install_requires,
      tests_require=['nose==1.3.4', 'mock==1.0.1']
)
