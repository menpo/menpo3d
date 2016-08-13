import numpy as np
import menpo.io as mio
from menpo.shape import ColouredTriMesh


class Menpo3DMM:

    # TODO
    def __init__(self, shape_model=None, texture_model=None):
        self.shape_model = shape_model
        self.texture_model = texture_model

    @classmethod
    def init_from_pickle(cls, shape_path, texture_path):
        shape = mio.import_pickle(shape_path)
        texture = mio.import_pickle(texture_path)
        return Menpo3DMM(shape, texture)

    def instance(self, tex_scale, alpha=None, beta=None):
        if alpha is None:
            alpha = np.zeros(len(self.shape_model.eigenvalues))
        if beta is None:
            beta = np.zeros(len(self.texture_model.eigenvalues))

        # Generate instance
        shape = self.shape_model.instance(alpha, normalized_weights=True)
        texture = self.texture_model.instance(beta, normalized_weights=True)

        trimesh = ColouredTriMesh(shape.points, trilist=shape.trilist,
                                  colours=tex_scale.apply(texture.reshape([-1, 3])))

        return trimesh