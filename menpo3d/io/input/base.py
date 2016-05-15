import warnings
from functools import partial

from menpo.io.input.base import (glob_with_suffix, _import_glob_lazy_list,
                                 _import, _import_object_attach_landmarks,
                                 _data_dir_path, _data_path_to,
                                 _ls_builtin_assets, BuiltinAssets,
                                 _import_builtin_asset)
from menpo.io.input import same_name, image_paths
from menpo3d.base import menpo3d_src_dir_path
from .extensions import mesh_types, mesh_landmark_types


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


def same_name_texture(path, paths_callable=image_paths):
    r"""
    Default image texture resolver. Returns **the lexicographically
    sorted first** texture found to have the same stem as the asset. A warning
    is raised if more than one texture is found.
    """
    # pattern finding all landmarks with the same stem
    pattern = path.with_suffix('.*')
    texture_paths = sorted(paths_callable(pattern))
    if len(texture_paths) > 1:
        warnings.warn('More than one texture found for file, returning '
                      'only the first.')
    if not texture_paths:
        return None
    return texture_paths[0]


same_name_landmark = partial(same_name, paths_callable=landmark_file_paths)


menpo3d_data_dir_path = partial(_data_dir_path, menpo3d_src_dir_path)
menpo3d_data_dir_path.__doc__ = _data_dir_path.__doc__

menpo3d_ls_builtin_assets = partial(_ls_builtin_assets, menpo3d_data_dir_path)
menpo3d_ls_builtin_assets.__doc__ = _ls_builtin_assets.__doc__

menpo3d_data_path_to = partial(_data_path_to, menpo3d_data_dir_path,
                               menpo3d_ls_builtin_assets)
menpo3d_data_path_to.__doc__ = _data_path_to.__doc__

_menpo3d_import_builtin_asset = partial(_import_builtin_asset,
                                        menpo3d_data_path_to,
                                        mesh_types, mesh_landmark_types,
                                        texture_resolver=same_name_texture)
_menpo3d_import_builtin_asset.__doc__ = _import_builtin_asset.__doc__


def import_mesh(filepath, landmark_resolver=same_name_landmark,
                texture_resolver=same_name_texture):
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
    kwargs = {'texture_resolver': texture_resolver}
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


import_builtin_asset = BuiltinAssets(_menpo3d_import_builtin_asset)

for asset in menpo3d_ls_builtin_assets():
    setattr(import_builtin_asset, asset.replace('.', '_'),
            partial(_menpo3d_import_builtin_asset, asset))
