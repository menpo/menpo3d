from menpo.io.input.base import (glob_with_suffix, _import_glob_lazy_list,
                                 _import, _import_object_attach_landmarks)
from menpo3d.base import menpo3d_src_dir_path


def same_name(path):
    r"""
    Menpo3d's default landmark resolver. Returns all landmarks found to have
    the same stem as the asset.
    """
    # pattern finding all landmarks with the same stem
    pattern = path.with_suffix('.*')
    # find all the landmarks we can with this name. Key is ext (without '.')
    return {p.suffix[1:].upper(): p for p in landmark_file_paths(pattern)}


def data_dir_path():
    r"""A path to the Menpo3d built in ./data folder on this machine.

    Returns
    -------
    `pathlib.Path`
        The path to the local Menpo3d ./data folder

    """
    return menpo3d_src_dir_path() / 'data'


def ls_builtin_assets():
    r"""List all the builtin asset examples provided in Menpo3d.

    Returns
    -------
    list of strings
        Filenames of all assets in the data directory shipped with Menpo3d

    """
    return [p.name for p in data_dir_path().glob('*')]


def data_path_to(asset_filename):
    r"""
    The path to a builtin asset in the ./data folder on this machine.

    Parameters
    ----------
    asset_filename : `str`
        The filename (with extension) of a file builtin to Menpo3d. The full
        set of allowed names is given by :func:`ls_builtin_assets()`

    Returns
    -------
    data_path : `pathlib.Path`
        The path to a given asset in the ./data folder

    Raises
    ------
    ValueError
        If the asset_filename doesn't exist in the `data` folder.
    """
    asset_path = data_dir_path() / asset_filename
    if not asset_path.is_file():
        raise ValueError("{} is not a builtin asset: {}".format(
            asset_filename, ls_builtin_assets()))
    return asset_path


def _import_builtin_asset(asset_name, **kwargs):
    r"""Single builtin asset (mesh or landmark) importer.

    Imports the relevant builtin asset from the ./data directory that
    ships with Menpo3d.

    Parameters
    ----------
    asset_name : `str`
        The filename of a builtin asset (see :map:`ls_builtin_assets`
        for allowed values)

    Returns
    -------
    asset
        An instantiated :map:`LandmarkGroup` or :map:`TriMesh` asset.
    """
    asset_path = data_path_to(asset_name)
    # Import could be either a mesh or a set of landmarks, so we try
    # importing them both separately.
    try:
        return _import(asset_path, mesh_types,
                       landmark_ext_map=mesh_landmark_types,
                       landmark_attach_func=_import_object_attach_landmarks,
                       importer_kwargs=kwargs)
    except ValueError:
        return _import(asset_path, mesh_landmark_types,
                       importer_kwargs=kwargs)


def import_mesh(filepath, landmark_resolver=same_name, texture=True):
    r"""Single mesh (and associated landmarks and texture) importer.

    Iff an mesh file is found at `filepath`, returns a :map:`TriMesh`
    representing it. Landmark files sharing the same filename
    will be imported and attached too. If texture coordinates and a suitable
    texture are found the object returned will be a :map:`TexturedTriMesh`.

    Parameters
    ----------
    filepath : `str`
        A relative or absolute filepath to an image file.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        mesh. The function should take one argument (the mesh itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the mesh file.
    texture : `bool`, optional
        If ``False``, don't search for textures.

    Returns
    -------
    trimesh : :map:`TriMesh`
        An instantiated :map:`TriMesh` (or subclass thereof)
    """
    kwargs = {'texture': texture}
    return _import(filepath, mesh_types,
                   landmark_resolver=landmark_resolver,
                   landmark_ext_map=mesh_landmark_types,
                   landmark_attach_func=_import_object_attach_landmarks,
                   importer_kwargs=kwargs)


def import_meshes(pattern, max_meshes=None, shuffle=False,
                  landmark_resolver=same_name, textures=True,
                  as_generator=False, verbose=False):
    r"""Multiple mesh importer.

    Makes it's best effort to import and attach relevant related
    information such as landmarks. It searches the directory for files that
    begin with the same filename and end in a supported extension.

    If texture coordinates and a suitable texture are found the object
    returned will be a :map:`TexturedTriMesh`.

    Note that this is a function returns a :map:`LazyList`. Therefore, the
    function will return immediately and indexing into the returned list
    will load the landmarks at run time. If all meshes should be loaded, then
    simply wrap the returned :map:`LazyList` in a Python `list`.

    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for textures and meshes.
    max_meshes : positive `int`, optional
        If not ``None``, only import the first ``max_meshes`` meshes found.
        Else, import all.
    shuffle : `bool`, optional
        If ``True``, the order of the returned meshes will be randomised. If
        ``False``, the order of the returned meshes will be alphanumerically
        ordered.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        mesh. The function should take one argument (the mesh itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the mesh file.
    textures : `bool`, optional
        If ``False``, don't search for textures.
    as_generator : `bool`, optional
        If ``True``, the function returns a generator and assets will be yielded
        one after another when the generator is iterated over.
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Returns
    -------
    lazy_list : :map:`LazyList` or generator of Python objects
        A :map:`LazyList` or generator yielding Python objects inside the
        pickle files found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no meshes are found at the provided glob.
    """
    kwargs = {'texture': textures}
    return _import_glob_lazy_list(
        pattern, mesh_types, max_assets=max_meshes, shuffle=shuffle,
        landmark_resolver=landmark_resolver,
        landmark_ext_map=mesh_landmark_types,
        importer_kwargs=kwargs, as_generator=as_generator,
        landmark_attach_func=_import_object_attach_landmarks, verbose=verbose)


def import_landmark_file(filepath, landmark_resolver=same_name):
    r"""Single landmark group importer.

    Iff an landmark file is found at `filepath`, returns a :map:`LandmarkGroup`
    representing it.

    Parameters
    ----------
    filepath : `str`
        A relative or absolute filepath to an landmark file.

    Returns
    -------
    :map:`LandmarkGroup`
        The :map:`LandmarkGroup` that the file format represents.

    """
    return _import(filepath, mesh_landmark_types,
                   landmark_resolver=landmark_resolver)


def import_landmark_files(pattern, max_landmarks=None, shuffle=False,
                          as_generator=False, verbose=False):
    r"""Multiple landmark file importer.

    Note that this is a function returns a :map:`LazyList`. Therefore, the
    function will return immediately and indexing into the returned list
    will load the landmarks at run time. If all landmarks should be loaded, then
    simply wrap the returned :map:`LazyList` in a Python `list`.

    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for landmarks.
    max_landmarks : positive `int`, optional
        If not ``None``, only import the first ``max_landmarks`` found.
        Else, import all.
    shuffle : `bool`, optional
        If ``True``, the order of the returned landmark files will be
        randomised. If ``False``, the order of the returned landmark files will
        be alphanumerically ordered.
    as_generator : `bool`, optional
        If ``True``, the function returns a generator and assets will be yielded
        one after another when the generator is iterated over.
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Returns
    ------
    lazy_list : :map:`LazyList` or generator of Python objects
        A :map:`LazyList` or generator yielding Python objects inside the
        pickle files found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no landmarks are found at the provided glob.

    """
    return _import_glob_lazy_list(pattern, mesh_landmark_types,
                                  max_assets=max_landmarks,
                                  shuffle=shuffle, as_generator=as_generator,
                                  verbose=verbose)


def mesh_paths(pattern):
    r"""
    Return mesh filepaths that Menpo3d can import that match the glob pattern.
    """
    return glob_with_suffix(pattern, mesh_types)


def landmark_file_paths(pattern):
    r"""
    Return landmark file filepaths that Menpo3d can import that match the glob
    pattern.
    """
    return glob_with_suffix(pattern, mesh_landmark_types)


def import_builtin(x):

    def execute():
        return _import_builtin_asset(x)

    return execute


class BuiltinAssets(object):

    def __call__(self, asset_name):
        return _import_builtin_asset(asset_name)

import_builtin_asset = BuiltinAssets()

for asset in ls_builtin_assets():
    setattr(import_builtin_asset, asset.replace('.', '_'), import_builtin(asset))


# Avoid circular imports
from .extensions import mesh_types, mesh_landmark_types
