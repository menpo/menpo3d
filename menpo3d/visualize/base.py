from .viewmayavi import (
    MayaviTriMeshViewer3d, MayaviPointGraphViewer3d,
    MayaviTexturedTriMeshViewer3d, MayaviLandmarkViewer3d,
    MayaviVectorViewer3d, MayaviColouredTriMeshViewer3d, MayaviHeatmapViewer3d)

# from .viewitkwidgets import (ItkwidgetsTriMeshViewer3d,
#                             ItkwidgetsPointGraphViewer3d)

from .viewk3dwidgets import (K3dwidgetsTriMeshViewer3d,
                             K3dwidgetsPointGraphViewer3d,
                             K3dwidgetsVectorViewer3d,
                             K3dwidgetsLandmarkViewer3d,
                             K3dwidgetsTexturedTriMeshViewer3d,
                             K3dwidgetsHeatmapViewer3d,
                             K3dwidgetsPCAModelViewer3d)

PointGraphViewer3d = MayaviPointGraphViewer3d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
ColouredTriMeshViewer3d = MayaviColouredTriMeshViewer3d
LandmarkViewer3d = MayaviLandmarkViewer3d
VectorViewer3d = MayaviVectorViewer3d
HeatmapViewer3d = MayaviHeatmapViewer3d

TriMeshInlineViewer3d = K3dwidgetsTriMeshViewer3d
TexturedTriMeshInlineViewer3d = K3dwidgetsTexturedTriMeshViewer3d
LandmarkInlineViewer3d = K3dwidgetsLandmarkViewer3d
PointGraphInlineViewer3d = K3dwidgetsPointGraphViewer3d
VectorInlineViewer3d = K3dwidgetsVectorViewer3d
HeatmapInlineViewer3d = K3dwidgetsHeatmapViewer3d
PCAModelInlineViewer3d = K3dwidgetsPCAModelViewer3d
