import os
from pathlib import Path


def menpo3d_src_dir_path():
    r"""The path to the top of the menpo3d Python package.

    Useful for locating where the data folder is stored.

    Returns
    -------
    path : str
        The full path to the top of the Menpo3d package
    """
    return Path(os.path.abspath(__file__)).parent
