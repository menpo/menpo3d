<p align="center">
  <img src="menpo3d-logo.png" alt="menpo" width="30%"></center>
  <br><br>
  <a href="https://github.com/menpo/menpo3d/blob/master/LICENSE.txt"><img src="http://img.shields.io/badge/License-BSD-green.svg" alt="BSD License"/></a>
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
```
pip install menpo3d
conda install -c conda-forge menpo3d
```
`menpo3d.correspond.solve_pnp` needs OpenCV (`pip install "menpo3d[opencv]"`).

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

mesh = menpo3d.io.import_builtin_asset("james.obj")
```
```python
%matplotlib qt
mesh.view()
```


Releasing
---------
Releases are made from git tags: tag `vX.Y.Z` (a signed, annotated tag) and push it. The
`Release` workflow builds the wheel and the sdist and publishes them to PyPI with trusted
publishing after the `pypi` environment is approved; conda-forge's bot then updates the feedstock.
A manual run of the workflow with `publish: testpypi` is a dry run on TestPyPI.
