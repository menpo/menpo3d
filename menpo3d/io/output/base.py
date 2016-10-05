from pathlib import Path

from menpo.compatibility import basestring
from menpo.io.output.base import (_export, _export_paths_only,
                                  _normalize_extension,
                                  _validate_and_get_export_func,
                                  _enforce_only_paths_supported)
from menpo.io.output.extensions import image_types


def export_landmark_file(landmark_group, fp, extension=None,
                         overwrite=False):
    r"""
    Exports a given landmark group. The ``fp`` argument can be either
    or a `str` or any Python type that acts like a file. If a file is provided,
    the ``extension`` kwarg **must** be provided. If no
    ``extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix in string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to export.
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    extension : `str` or None, optional
        The extension to use, this must match the file path if the file
        path is a string. Determines the type of exporter that is used.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``fp`` is a `str` and the ``extension`` is not ``None``
        and the two extensions do not match
    ValueError
        ``fp`` is a `file`-like object and ``extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    from .extensions import landmark_types

    _export(landmark_group, fp, landmark_types, extension,
            overwrite)


def export_mesh(mesh, fp, extension=None, overwrite=False, **kwargs):
    r"""
    Exports a given mesh. The ``fp`` argument can be either
    a `str` or any Python type that acts like a file. If a file is provided,
    the ``extension`` kwarg **must** be provided. If no
    ``extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix of string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    mesh : :map:`PointCloud`
        The mesh to export.
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    extension : `str` or None, optional
        The extension to use, this must match the file path if the file
        path is a string. Determines the type of exporter that is used.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.
    **kwargs : `dict`
        Keyword arguments to be passed through to exporter function.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``fp`` is a `str` and the ``extension`` is not ``None``
        and the two extensions do not match
    ValueError
        ``fp`` is a `file`-like object and ``extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    from .extensions import mesh_types, mesh_types_paths_only
    if isinstance(fp, basestring):
        fp = Path(fp)
    if isinstance(fp, Path):
        _, extension = _validate_and_get_export_func(
            fp, mesh_types, extension, overwrite, return_extension=True)
    if extension in mesh_types_paths_only:
        _export_paths_only(mesh, fp, mesh_types, extension, overwrite,
                           exporter_kwargs=kwargs)
    else:
        _export(mesh, fp, mesh_types, extension, overwrite,
                exporter_kwargs=kwargs)


def export_textured_mesh(mesh, file_path, extension=None,
                         texture_extension='.jpg', overwrite=False, **kwargs):
    r"""
    Exports a given textured mesh. The ``filepath`` argument must be a string
    containing the filepath to write the mesh out to. Unlike the other export
    methods, this cannot take a file-like object because two files are written.

    If no ``extension`` is provided then the export type is calculated based on
    the filepath extension.

    The exported texture is always placed in the same directory as the exported
    mesh and is given the same base name.

    Parameters
    ----------
    mesh : :map:`PointCloud`
        The mesh to export.
    file_path : `pathlib.Path` or `str`
        The path to save the object at/into.
    extension : `str` or None, optional
        The extension to use for the exported mesh, this must match the file
        path if the file path is a string. Determines the type of exporter that
        is used for the mesh.
    texture_extension : `str`, optional
        Determines the type of exporter that is used for the texture.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.
    **kwargs : `dict`
        Keyword arguments to be passed through to exporter function.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``fp`` is a `str` and the ``extension`` is not ``None``
        and the two extensions do not match
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    ValueError
        The mesh is not textured.
    """
    from menpo.shape import TexturedTriMesh
    if not isinstance(mesh, TexturedTriMesh):
        raise ValueError('Must supply a textured mesh.')

    file_path = _enforce_only_paths_supported(file_path, 'textured mesh')
    export_mesh(mesh, file_path, extension=extension, overwrite=overwrite,
                exporter_kwargs=kwargs)

    # Put the image next to the mesh
    texture_extension = _normalize_extension(texture_extension)
    image_output_path = Path(file_path).with_suffix(texture_extension)
    _export(mesh.texture, image_output_path,
            image_types, texture_extension, overwrite)
