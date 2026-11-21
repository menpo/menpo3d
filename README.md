<p align="center">
  <img src="menpo3d-logo.png" alt="menpo" width="30%"></center>
  <br><br>
  <a href="https://github.com/menpo/menpo3d/blob/master/LICENSE.txt"><img src="http://img.shields.io/badge/License-BSD-green.svg" alt="BSD License"/></a>
  <a href="https://menpo3d.readthedocs.io/en/stable/"><img src="https://readthedocs.org/projects/menpo3d/badge/?version=stable" alt="Documentation"/></a>
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
`menpo3d.correspond.solve_pnp` needs OpenCV (`pip install "menpo3d[opencv]"`). Importing and
exporting meshes, closest point queries, rasterization and offline rendering need nothing extra.
The interactive viewers need one of these extras:

| Extra | Gives you |
|---|---|
| `menpo3d[viz]` | everything below, including H.264 |
| `menpo3d[viz_notebook]` | the notebook widget and the browser window (JPEG frames) |
| `menpo3d[viz_h264]` | H.264 video for the streamed viewers |
| `menpo3d[viz_native_window]` | the native window |

Visualizing 3D objects
----------------------
menpo3d draws 3D objects with a moderngl renderer (OpenGL, no Qt or Mayavi). `mesh.view()` and the
other `view` methods work in a Jupyter notebook as an interactive widget, in a native window, in a
browser window (over SSH too) and offline to a PNG. In a notebook, with `menpo3d[viz_notebook]` installed, there is nothing else to set up (no `%matplotlib` magic):
```python
import menpo3d

mesh = menpo3d.io.import_builtin_asset("james.obj")
mesh.view()  # drag to rotate, shift-drag to pan, wheel to zoom
```
```python
renderer = mesh.view()
renderer.save_figure("james.png")  # offline, needs no extra
renderer.figure.show()  # a native window (or a browser window)
pose = renderer.figure.get_pose()  # the camera as you left it with the mouse
```
See the [documentation](https://menpo3d.readthedocs.io/en/stable/) for the full viewer API.


Releasing
---------
Releases are made from git tags: tag `vX.Y.Z` (a signed, annotated tag) and push it. The
`Release` workflow builds the wheel and the sdist and publishes them to PyPI with trusted
publishing after the `pypi` environment is approved; conda-forge's bot then updates the feedstock.
A manual run of the workflow with `publish: testpypi` is a dry run on TestPyPI.
