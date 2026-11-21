"""The 3D viewers behind ``.view()``: moderngl, which works offline, in notebooks, in a window and over SSH."""

import os
import warnings

from .viewmoderngl import (
    ModernGLColouredTriMeshViewer3d as ColouredTriMeshViewer3d,
    ModernGLLandmarkViewer3d as LandmarkViewer3d,
    ModernGLPointGraphViewer3d as PointGraphViewer3d,
    ModernGLRenderer as Renderer3d,
    ModernGLSurfaceViewer3d as SurfaceViewer3d,
    ModernGLTexturedTriMeshViewer3d as TexturedTriMeshViewer3d,
    ModernGLTriMeshViewer3d as TriMeshViewer3d,
    ModernGLVectorViewer3d as VectorViewer3d,
)

if os.environ.get("MENPO3D_VIEWER", "").lower() == "mayavi":
    warnings.warn(
        "MENPO3D_VIEWER=mayavi is ignored: the Mayavi viewers were removed in menpo3d 0.10, "
        "the moderngl viewers have the same API.",
        stacklevel=2,
    )
