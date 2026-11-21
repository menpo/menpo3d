.. _changelog:

Changelog
#########

For releases before 0.10.0 see the `GitHub releases <https://github.com/menpo/menpo3d/releases>`_.

0.10.1 (2026/11/21)
-------------------

**Fixed.** ``mesh.view_landmarks()`` (and ``view_landmarks()`` on a point cloud) raised
``ValueError: Invalid RGBA argument: None`` unless a ``marker_colour`` was given. menpo passes
``None`` for a colour that was not chosen, and the 3D viewers did not accept it. ``None`` now
means the viewer's own default: red for markers, lines and meshes, black for the point numbers
and the normals. A colour that is given is used as before. The same code was in 0.10.0's
predecessors (the Mayavi viewers), so older versions are probably affected too.

Github Pull Requests
....................
- `#97`_ Draw with the default colour when menpo passes None, so that view_landmarks works (@patricksnape)

.. _#97: https://github.com/menpo/menpo3d/pull/97

0.10.0 (2026/11/21)
-------------------

menpo3d no longer depends on VTK or Mayavi. The 3D viewers are rewritten on
moderngl, and the mesh importers, closest point queries and occlusion are pure
NumPy and SciPy.

**New viewers.** ``mesh.view()``, ``landmarks.view()`` and the other ``view``
methods keep their API (the same viewer classes, constructor arguments and
``render()`` options, ``figure_id``/``new_figure``, ``save_figure``,
``renderer_settings``) but draw with one moderngl renderer, so every output
shows the same pixels:

- offline, to a PNG (``save_figure``), with no extra dependencies;
- in a Jupyter notebook as an interactive widget (rotate, pan, zoom, click to
  pick a triangle), with H.264 video while the camera moves and a lossless
  still when it stops;
- in a native window (moderngl-window on GLFW), falling back to a browser
  window when there is no display;
- in a browser window, which also works over SSH with a forwarded port.

The streaming part is the separate
`scene-streamer <https://github.com/menpo/scene-streamer>`_ package. There is a
screen-space colour bar and statistics text, a height-field
``SurfaceViewer3d`` (the old ``mlab.surf``), ``Figure.reset_camera``, and a
camera pose that can be read and set (``figure.get_pose()``,
``figure.set_pose(**pose)``) so a view placed with the mouse can be reused for
offline rendering.

**Breaking: Mayavi and VTK are gone.**

- ``menpo3d[mayavi]`` and the ``vtk`` extra no longer exist, and
  ``MENPO3D_VIEWER=mayavi`` is ignored with a warning. ``%matplotlib qt`` is no
  longer needed.
- The viewers come from new extras: ``menpo3d[viz]`` (everything, including
  H.264), ``menpo3d[viz_notebook]`` (notebook widget and browser window, JPEG
  frames), ``menpo3d[viz_h264]`` (H.264 video, PyAV) and
  ``menpo3d[viz_native_window]``. Offline rendering needs none of them.
- ``menpo3d.vtkutils`` (``trimesh_to_vtk``, ``trimesh_from_vtk``,
  ``VTKClosestPointLocator``) is removed. Use
  ``menpo3d.closest_point.ClosestPointLocator``, an exact NumPy/SciPy closest
  point on a mesh that ``snap_pointcloud_to_surface``,
  ``barycentric_coordinates_of_pointcloud`` and NICP now use.

**Importers.** OBJ, PLY, STL and VRML 2.0 are read without VTK. They reproduce
what the VTK importers returned (point order, duplicated points, unused
vertices, triangle order and which diagonal a quad is split on), because
existing datasets are built against those point lists. Polygons are
retriangulated like ``vtkTriangleFilter`` did, and a warning is raised every
time. Differences you may notice:

- malformed input that VTK accepted now raises ``ValueError`` (out-of-range or
  negative face indices, a PLY without ``end_header``, a truncated ASCII PLY);
  binary STL is read to the end and must be whole facets;
- an OBJ with materials but no texture coordinates, given a texture, is a
  ``TriMesh``; VTK gave a ``TexturedTriMesh`` with all ``-1`` coordinates;
- a byte order mark in an OBJ is skipped (VTK lost the first vertex); VRML
  numbers such as ``.5`` are read correctly (VTK read them as 0);
- VRML primitives (``Box``, ``Cone``, ``Cylinder``, ``Sphere``),
  ``ElevationGrid`` and ``Extrusion`` are not supported.

**Fixes.** ``per_vertex_occlusion_accurate`` reused the first ray's hit for
every vertex; it now casts every ray. The PLY exporter no longer segfaults on
float64 texture coordinates. ``scipy`` is now declared as a dependency.

Github Pull Requests
....................
- `#92`_ Move closest point, occlusion and PLY export off VTK (@patricksnape)
- `#93`_ Import OBJ, PLY, STL and VRML without VTK (same topology), make VTK and Mayavi optional (@patricksnape)
- `#94`_ moderngl viewer: notebook widget, browser and native window, replacing Mayavi rendering (@patricksnape)
- `#95`_ Remove the Mayavi viewers (menpo3d 0.10) (@patricksnape)

.. _#92: https://github.com/menpo/menpo3d/pull/92
.. _#93: https://github.com/menpo/menpo3d/pull/93
.. _#94: https://github.com/menpo/menpo3d/pull/94
.. _#95: https://github.com/menpo/menpo3d/pull/95
