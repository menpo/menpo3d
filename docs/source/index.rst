Menpo3D Documentation
=====================

Menpo3D is a library inside the `Menpo Project <https://www.menpo.org/>`_ that
makes manipulating 3D mesh data a simple task. In particular, it provides the
ability to import, visualize and rasterize 3D meshes. The 3D meshes
themselves (:map:`TriMesh`, :map:`ColouredTriMesh` and :map:`TexturedTriMesh`)
live in `Menpo <https://menpo.readthedocs.io/en/stable/>`_; Menpo3D adds the
functionality for working with them.

.. code-block:: python

    import menpo3d.io as m3io

    # the bunny and James meshes ship with the package
    mesh = m3io.import_builtin_asset.james_obj()
    print(mesh)

Installation
------------

.. code-block:: bash

    pip install menpo3d
    conda install -c conda-forge menpo3d

Menpo3D is pure Python and supports CPython 3.10 and later.
:func:`~menpo3d.correspond.solve_pnp` needs OpenCV
(``pip install "menpo3d[opencv]"``).

Visualizing 3D objects
----------------------

Menpo3D views 3D objects with `Mayavi <https://docs.enthought.com/mayavi/mayavi/>`_,
which is based on VTK and is an optional dependency: ``pip install "menpo3d[mayavi]"``.
To view meshes in a Jupyter notebook through the
`menpowidgets <https://menpowidgets.readthedocs.io/>`_ widgets, either use a
Qt window (``%matplotlib qt``) or let Mayavi render images into the notebook
with ``mlab.init_notebook("png")``.

API Documentation
-----------------

.. toctree::
  :maxdepth: 2

  api/barycentric
  api/camera
  api/closest_point
  api/correspond
  api/extractimage
  api/io
  api/math
  api/morphablemodel
  api/rasterize
  api/result
  api/unwrap
  api/visualize
