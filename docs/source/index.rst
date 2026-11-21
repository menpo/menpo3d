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

``mesh.view()``, ``landmarks.view()`` and the menpowidgets 3D widgets draw with
one OpenGL renderer (moderngl). The same renderer, with the same camera, serves
three uses, so they cannot disagree about what a scene looks like:

* **Offline**: ``renderer.save_figure("out.png", size=(800, 600))`` renders on
  any machine with OpenGL (a GPU, or Mesa's software renderer), without a
  window.
* **In a notebook**: a figure displays as an interactive widget (drag to
  rotate, shift- or right-drag to pan, wheel to zoom, click to pick a
  triangle). The frames are rendered on the machine running the kernel and
  streamed to the browser as H.264, so it works over SSH too. A lossless still
  of the last view is stored in the notebook output.
* **In a window**: ``figure.show()`` (or ``menpo3d.visualize.viewmoderngl.show()``)
  opens a standalone viewer window and returns when it is closed. It is a real
  window (moderngl-window on GLFW, ``pip install "menpo3d[viz_native_window]"``) when there
  is a display, and a browser app window otherwise (``window="browser"`` forces
  that; it works over SSH with a port forward).

.. code-block:: python

    renderer = mesh.view(mesh_type="surface", colour="r", new_figure=True)
    renderer.figure.show()                      # place the mesh interactively
    pose = renderer.figure.get_pose()           # azimuth, elevation, distance, target, fov
    renderer.figure.set_pose(**{**pose, "azimuth": 0.5})
    renderer.save_figure("view.png", size=(1024, 768), overwrite=True)

Scalar values can be shown with a colour bar and a statistics text, which are
drawn into every render (stream, native window and ``save_figure``)::

    from menpo3d.visualize.viewer.colormap import scalars_to_colours, statistics_lines

    colours, lo, hi = scalars_to_colours(errors, "hot_r", vmin=0, vmax=2)
    renderer = ColouredTriMeshViewer3d(None, True, mesh.points, mesh.trilist, colours).render()
    renderer.add_colorbar("hot_r", lo, hi, title="error (mm)")
    renderer.add_text("\n".join(statistics_lines(errors)), "top-left")

``SurfaceViewer3d(None, True, values, mask).render(colorbar=True)`` draws a 2D
array as a height field.

The interactive parts need ``pip install "menpo3d[viz]"`` (everything), or just
``"menpo3d[viz_notebook]"`` for the notebook widget and the browser window and
``"menpo3d[viz_native_window]"`` for the native window; offline rendering needs
none of them. Without ``"menpo3d[viz_h264]"`` (PyAV, which bundles ffmpeg with
x264) frames are streamed as JPEG; with it they are H.264, which is several
times smaller on slow links. If PyAV is missing, the
ffmpeg build lacks the encoder, or the browser has no WebCodecs (Firefox,
Safari), the viewer falls back to JPEG by itself. To reach a remote machine, forward the viewer's port or call
``menpo3d.visualize.viewer.hub.configure(host="0.0.0.0", public_url=...)``
before the first viewer is created.

The viewers that used Mayavi are still available: set
``MENPO3D_VIEWER=mayavi`` before importing ``menpo3d.visualize`` (install with
``pip install "menpo3d[mayavi]"``).

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
