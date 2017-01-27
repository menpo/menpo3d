from scipy.io import loadmat
from menpo.model import PCAModel
from menpo.shape import TriMesh


def lsfm_model_importer(path, **kwargs):
    m = loadmat(str(path))
    mean = TriMesh(m['mean'].reshape([-1, 3]), trilist=m['trilist'])
    return PCAModel.init_from_components(m['components'].T, m['eigenvalues'],
                                         mean, m['n_training_samples'], True)
