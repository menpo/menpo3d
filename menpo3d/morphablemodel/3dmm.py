import numpy as np
import menpo.io as mio
from scipy.io import loadmat
from model import Model
from fitter import MorphableModelFitter


class Menpo3DMM:
    # TODO
    def __init__(self, image, model, fitting, anchors_pf, epsilon):
        self.image = image
        self.model = model
        self.anchors_pf = anchors_pf
        self.fitting = fitting
        self.epsilon = epsilon

    # TODO
    def run(self):
        # Runs the fitting using the 3D Morphable Model
        # model = Model()
        # self.model = model.init_from_basel(model_pf)

        fitter = MorphableModelFitter(self.model)
        fitter.fit(self.image, self.anchors_pf)

if __name__ == '__main__':
    menpo3d = Menpo3DMM(0, 0.1, 0, "bale.mat", 0)
    menpo3d.run()

















