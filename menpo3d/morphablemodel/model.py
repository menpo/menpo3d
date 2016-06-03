from scipy.io import loadmat
import numpy as np
import menpo.io as mio


class Model:

    def __init__(self, shape_mean=None, shape_pc=None, shape_ev=None,
                 texture_mean=None, texture_pc=None, texture_ev=None,
                 triangle_array=None, seg_bin=None, seg_mm=None, seg_mb=None):

        """
            Initializes the model from shape, texture, triangle array and segmentation components
        """

        # Shape
        self.shape_mean = shape_mean    # mean
        self.shape_pc = shape_pc    # principal components
        self.shape_ev = shape_ev    # eigenvalues

        # Texture
        self.texture_mean = texture_mean    # mean
        self.texture_pc = texture_pc    # principal components
        self.texture_ev = texture_ev     # eigenvalues

        # Triangle array
        self.triangle_array = np.transpose(triangle_array)  # triangle array

        # Segmentation
        self.seg_bin = seg_bin
        self.seg_mm = seg_mm
        self.seg_mb = seg_mb

    @classmethod
    def init_from_basel(cls, basel_pf):
        # Loading the model
        data = loadmat(basel_pf)["model"]
        # Shape
        shape_mean = data["shapeMean"]  # mean
        shape_pc = data["shapePC"]  # principal components
        shape_ev = data["shapeEV"]  # eigenvalues

        # Texture
        texture_mean = data["textureMean"]  # mean
        texture_pc = data["texturePC"]  # principal components
        texture_ev = data["textureEV"]  # eigenvalues

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

    def save_data(self, pf):

        """
            Saves the data as a pickle file
        """
        mio.export_pickle(self, pf, overwrite=True)

