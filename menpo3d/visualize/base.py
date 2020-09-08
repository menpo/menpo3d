from .viewmayavi import (
    MayaviTriMeshViewer3d, MayaviPointGraphViewer3d,
    MayaviTexturedTriMeshViewer3d, MayaviLandmarkViewer3d,
    MayaviVectorViewer3d, MayaviColouredTriMeshViewer3d)

from .viewitkwidgets import (ItkwidgetsTriMeshViewer3d,
                             ItkwidgetsPointGraphViewer3d)

PointGraphViewer3d = MayaviPointGraphViewer3d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
ColouredTriMeshViewer3d = MayaviColouredTriMeshViewer3d
LandmarkViewer3d = MayaviLandmarkViewer3d
VectorViewer3d = MayaviVectorViewer3d

TriMeshInlineViewer3d = ItkwidgetsTriMeshViewer3d
PointGraphInlineViewer3d = ItkwidgetsPointGraphViewer3d
