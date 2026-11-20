"""Which 3D viewer backs ``.view()``: moderngl by default (works offline, in notebooks and over SSH), Mayavi on request.

Set ``MENPO3D_VIEWER=mayavi`` (before importing menpo3d.visualize) to get the Mayavi viewers back.
"""

import os

if os.environ.get("MENPO3D_VIEWER", "moderngl").lower() == "mayavi":
    from .viewmayavi import (
        MayaviTriMeshViewer3d as TriMeshViewer3d,
        MayaviPointGraphViewer3d as PointGraphViewer3d,
        MayaviTexturedTriMeshViewer3d as TexturedTriMeshViewer3d,
        MayaviLandmarkViewer3d as LandmarkViewer3d,
        MayaviVectorViewer3d as VectorViewer3d,
        MayaviColouredTriMeshViewer3d as ColouredTriMeshViewer3d,
        MayaviRenderer as Renderer3d,
    )
else:
    from .viewmoderngl import (
        ModernGLTriMeshViewer3d as TriMeshViewer3d,
        ModernGLPointGraphViewer3d as PointGraphViewer3d,
        ModernGLTexturedTriMeshViewer3d as TexturedTriMeshViewer3d,
        ModernGLLandmarkViewer3d as LandmarkViewer3d,
        ModernGLVectorViewer3d as VectorViewer3d,
        ModernGLColouredTriMeshViewer3d as ColouredTriMeshViewer3d,
        ModernGLRenderer as Renderer3d,
    )
