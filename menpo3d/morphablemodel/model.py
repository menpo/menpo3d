import scipy.io as spio
import numpy as np
import menpo.io as mio


class Model:
    r"""
        Class for loading and managing a 3D Morphable Model.

        Parameters
        ----------
        shape_mean :
            The mean of the shape vectors.
        shape_pc :
            The principal components of the shape vectors.
        shape_ev :
            The eigenvalues of the shape vectors.
        texture_mean:
            The mean of the texture vectors.
        texture_pc:
            The principal components of the texture vectors.
        texture_ev:
            The eigenvalues of the texture vectors.
        triangle_array:
            The array containing all model triangles.
        seg_bin :
        seg_mm :
        seg_mb :
        ...
    """
    def __init__(self, shape_mean=None, shape_pc=None, shape_ev=None,
                 texture_mean=None, texture_pc=None, texture_ev=None,
                 triangle_array=None, seg_bin=None, seg_mm=None, seg_mb=None):
        # Assign attributes

        self.shape_mean = shape_mean    # shape mean
        self.shape_pc = shape_pc    # shape principal components
        self.shape_ev = shape_ev    # shape eigenvalues

        self.texture_mean = texture_mean    # texture mean
        self.texture_pc = texture_pc    # texture principal components
        self.texture_ev = texture_ev     # texture eigenvalues

        self.triangle_array = np.transpose(triangle_array)  # triangle array

        self.seg_bin = seg_bin
        self.seg_mm = seg_mm
        self.seg_mb = seg_mb

    @classmethod
    def init_from_basel(cls, basel_pf):
        # Loading the model
        data = spio.loadmat(basel_pf)["model"]
        # Shape
        shape_mean = data["shapeMean"][0, 0]  # mean
        shape_pc = data["shapePC"][0, 0]  # principal components
        shape_ev = data["shapeEV"][0, 0]  # eigenvalues

        # Texture
        texture_mean = data["textureMean"][0, 0]  # mean
        texture_pc = data["texturePC"][0, 0]  # principal components
        texture_ev = data["textureEV"][0, 0]  # eigenvalues

        # Triangle array
        triangle_array = np.transpose(data["triangleArray"])  # triangle array

        # Segmentation
        seg_bin = data["segBin"]
        seg_mm = data["segMM"]
        seg_mb = data["segMB"]

        return Model(shape_mean, shape_pc, shape_ev,
                     texture_mean, texture_pc, texture_ev,
                     triangle_array, seg_bin, seg_mm, seg_mb)

    def change_model(self):
        return self

    def save_model(self, pf):
        # Save the model as a pickle file
        mio.export_pickle(self, pf, overwrite=True)

    @property
    def _str_title(self):
        return '3D Morphable Model'


