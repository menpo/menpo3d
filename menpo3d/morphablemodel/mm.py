import numpy as np
import menpo.io as mio
from menpo.shape import ColouredTriMesh


class Menpo3DMM:

    # TODO
    def __init__(self, shape_model=None, texture_model=None):
        self.shape = shape_model
        self.texture = texture_model

    @classmethod
    def init_from_path(cls, shape_path, texture_path):
        shape = mio.import_pickle(shape_path)
        texture = mio.import_pickle(texture_path)
        return Menpo3DMM(shape, texture)

    def instance(self, tex_scale, alpha=None, beta=None):
        if not alpha:
            alpha = np.zeros(len(self.shape.eigenvalues))
        if not beta:
            beta = np.zeros(len(self.texture.eigenvalues))

        # Generate instance
        shape = self.shape.instance(alpha, normalized_weights=True)
        texture = self.texture.instance(beta, normalized_weights=True)

        trimesh = ColouredTriMesh(shape.points, trilist=shape.trilist,
                                  colours=tex_scale.apply(texture.reshape([-1, 3])))

        return trimesh