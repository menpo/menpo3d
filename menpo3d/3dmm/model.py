from scipy.io import loadmat
import numpy as np
import menpo.io as mio


class Model:

    def __init__(self, pf):

        """
            Loads the data from the model specified in the pathfile  and initialises the model
        """
        # Loading the model
        data = loadmat(pf)["model"]

        # Shape
        self.shape_mean = data["shapeMean"]    # mean
        self.shape_pc = data["shapePC"]    # principal components
        self.shape_ev = data["shapeEV"]    # eigenvalues

        # Texture
        self.texture_mean = data["textureMean"]    # mean
        self.texture_pc = data["texturePC"]    # principal components
        self.texture_ev = data["textureEV"]     # eigenvalues

        # Triangle array
        self.triangle_array = np.transpose(data["triangleArray"])  # triangle array

        # Segmentation
        self.seg_bin = data["segBin"]
        self.seg_mm = data["segMM"]
        self.seg_mb = data["segMB"]

    def save_data(self, pf):

        """
            Saves the data as a pickle file
        """
        mio.export_pickle(self, pf, overwrite=True)

m = Model("model.mat")
