<p align="center">
  <img src="menpo3d-logo.png" alt="menpo" width="30%"></center>
  <br><br>
  <a href="https://github.com/menpo/menpo3d/blob/master/LICENSE.txt"><img src="http://img.shields.io/badge/License-BSD-green.svg" alt="BSD License"/></a>
  <img src="https://img.shields.io/badge/Python-3.6-green.svg" alt="Python 3.6 Support"/>
  <img src="https://img.shields.io/badge/Python-3.7-green.svg" alt="Python 3.7 Support"/>
</p>


menpo3d - Tools for manipulating meshes
=======================================
A library inside the [Menpo Project](http://www.menpo.org/) that makes manipulating 3D mesh data a
simple task. In particular, this project provides the ability to import,
visualize and rasterize 3D meshes. Although 3D meshes can be created within
the main Menpo project, this package adds the real functionality for working
with 3D data.

Installation
------------
Here in the Menpo team, we are firm believers in making installation as simple
as possible. Unfortunately, we are a complex project that relies on satisfying
a number of complex 3rd party library dependencies. The default Python packing
environment does not make this an easy task. Therefore, we evangelise the use
of the conda ecosystem, provided by
[Anaconda](https://store.continuum.io/cshop/anaconda/). In order to make things
as simple as possible, we suggest that you use conda too! To try and persuade
you, go to the [Menpo website](http://www.menpo.io/installation/) to find
installation instructions for all major platforms.

Visualizing 3D objects
----------------------

menpo3d adds support for viewing 3D objects through
[Mayavi](http://code.enthought.com/projects/mayavi/), which is based on VTK.
One of the main reasons menpo3d is a seperate project to the menpo core
library is to isolate the more complex dependencies that this brings to the
project. 3D visualization is not yet supported in the browser, so we rely on
platform-specific viewing mechanisms like QT or WX.

In order to view 3D items you will need to first use the `%matplotlib qt`
IPython magic command to set up QT for rendering (you do this instead of
`%matplotlib inline` which is what is needed for rendering directly
in Jupyter/Ipython notebooks). As a complete example, to view a
mesh in IPython you would run something like:
```python
import menpo3d
mesh = menpo3d.io.import_builtin_asset('james.obj')
```
```python
%matplotlib qt
mesh.view()
```
