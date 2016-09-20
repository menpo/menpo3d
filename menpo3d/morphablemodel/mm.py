import numpy as np
import menpo.io as mio
import menpo3d.io as m3dio
from menpo.shape import ColouredTriMesh
from menpo.transform import UniformScale


class Menpo3DMM:
    
    def __init__(self, shape_model=None, texture_model=None, landmarks=None):
        self.shape_model = shape_model
        self.texture_model = texture_model
        self.landmarks = landmarks

    @classmethod
    def init_from_file(cls, shape_path, texture_path, landmarks_path=None):
        shape = mio.import_pickle(shape_path)
        texture = mio.import_pickle(texture_path)
        if landmarks_path is not None:
            landmarks = m3dio.import_landmark_file(landmarks_path).lms
        else:
            landmarks = None
        return Menpo3DMM(shape, texture, landmarks)
    
    def __str__(self):
        str_out = '3D MORPHABLE MODEL \n\n' + \
                  'Shape Model \n----------- \n' + self.shape_model.__str__() + '\n' + \
                  'Texture Model \n------------- \n' + self.texture_model.__str__() + '\n' + \
                  'Landmarks \n--------- \n' + self.landmarks.__str__()
        return str_out

    def instance(self, model_type = 'bfm', alpha=None, beta=None, landmark_group = 'ibug68'):
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