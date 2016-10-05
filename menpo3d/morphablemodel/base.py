import numpy as np
from menpo.shape import ColouredTriMesh
from menpo.transform import UniformScale


class ColouredMorphableModel(object):
    
    def __init__(self, shape_model=None, texture_model=None, landmarks=None):
        self.shape_model = shape_model
        self.texture_model = texture_model
        self.landmarks = landmarks

    def __str__(self):
        return ('ColouredMorphableModel\n\n' +
                'Shape Model \n----------- \n' + self.shape_model.__str__() + '\n' +
                'Colour Model \n------------- \n' + self.texture_model.__str__() + '\n' +
                'Landmarks \n--------- \n' + self.landmarks.__str__())

    def instance(self, model_type='bfm', alpha=None, beta=None, landmark_group='ibug68'):
        if alpha is None:
            alpha = np.zeros(len(self.shape_model.eigenvalues))
        if beta is None:
            beta = np.zeros(len(self.texture_model.eigenvalues))

        # Generate instance
        shape = self.shape_model.instance(alpha, normalized_weights=True)
        texture = self.texture_model.instance(beta, normalized_weights=True)
        
        if model_type == 'bfm':
            tex_scale = UniformScale(1. / 255, 3)
            lms_scale = UniformScale(1e-5, 3)
            texture = tex_scale.apply(texture.reshape([-1, 3]))
            landmarks = lms_scale.apply(self.landmarks)
        elif model_type == 'lsfm':
            pass

        trimesh = ColouredTriMesh(shape.points, trilist=shape.trilist,
                                  colours=texture)
        
        trimesh.landmarks[landmark_group] = landmarks

        return trimesh
