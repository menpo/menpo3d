import warnings
from collections import namedtuple
import os.path as path
import json
from pathlib import Path

import numpy as np
from menpo.io.input.base import Importer, import_image
from menpo.io.exceptions import MeshImportError
from menpo.shape.mesh import ColouredTriMesh, TexturedTriMesh, TriMesh


# This formalises the return type of a mesh importer (before building)
# However, at the moment there is a disconnect between this and the
# Assimp type, and at some point they should become the same object
MeshInfo = namedtuple('MeshInfo', ['points', 'trilist', 'tcoords',
                                   'colour_per_vertex'])


def files_with_matching_stem(filepath):
    r"""
    Given a filepath, find all the files that share the same stem.

    Can be used to find all landmark files for a given image for instance.

    Parameters
    ----------
    filepath : `pathlib.Path`
        The filepath to be matched against

    Yields
    ------
    path : `pathlib.Path`
        A list of absolute filepaths to files that share the same stem
        as filepath.

    """
    return filepath.parent.glob('{}.*'.format(filepath.stem))


def filter_extensions(filepaths, extensions_map):
    r"""
    Given a set of filepaths, filter the files who's extensions are in the
    given map. This is used to find images and landmarks from a given basename.

    Parameters
    ----------
    filepaths : list of strings
        A list of absolute filepaths
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.

    Returns
    -------
    basenames : list of strings
        A list of basenames
    """
    extensions = extensions_map.keys()
    return [f.name for f in filepaths if f.suffix in extensions]


def find_alternative_files(filepath, extensions_map):
    r"""
    Given a filepath, search for files with the same basename that match
    a given extension type, eg images. If more than one file is found, an error
    is printed and the first such basename is returned.

    Parameters
    ----------
    filepath : string
        An absolute filepath
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.

    Returns
    -------
    base_name : string
        The basename of the file that was found eg `mesh.bmp`. Only **one**
        file is ever returned. If more than one is found, the first is taken.

    Raises
    ------
    ImportError
        If no alternative file is found
    """
    filepath = Path(filepath)
    try:
        all_paths = files_with_matching_stem(filepath)
        base_names = filter_extensions(all_paths, extensions_map)
        if len(base_names) > 1:
            print("Warning: More than one file was found: "
                  "{}. Taking the first by default".format(base_names))
        return base_names[0]
    except Exception as e:
        raise ImportError("Failed to find a file for {} from types {}. "
                          "Reason: {}".format(filepath, extensions_map, e))


class MeshImporter(Importer):
    r"""
    Abstract base class for importing meshes. Searches in the directory
    specified by filepath for landmarks and textures with the same basename as
    the mesh. If found, they are automatically attached. If a texture is found
    then a :map:`TexturedTriMesh` is built, if colour information is found a
    :map:`ColouredTriMesh` is built, and if neither is found a :map:`Trimesh`
    is built. Note that this behavior can be overridden if desired.

    Parameters
    ----------
    filepath : `str`
        Absolute filepath of the mesh.
    texture: 'bool', optional
        If ``False`` even if a texture exists a normal :map:`TriMesh` is
        produced.
    """

    def __init__(self, filepath, texture=True):
        super(MeshImporter, self).__init__(filepath)
        self.meshes = []
        self.import_textures = texture
        self.attempted_texture_search = False
        self.relative_texture_path = None

    def _search_for_texture(self):
        r"""
        Tries to find a texture with the same name as the mesh.

        Returns
        --------
        relative_texture_path : `str`
            The relative path to the texture or ``None`` if one can't be found
        """
        # Stop searching every single time we access the property
        self.attempted_texture_search = True
        # This import is here to avoid circular dependencies
        from menpo.io.input.extensions import image_types
        try:
            return find_alternative_files(self.filepath, image_types)
        except ImportError:
            return None

    @property
    def texture_path(self):
        """
        Get the absolute path to the texture. Returns None if one can't be
        found. Makes it's best effort to find an appropriate texture by
        searching for textures with the same name as the mesh. Will only
        search for the path the first time ``texture_path`` is invoked.

        Sets the :attr:`relative_texture_path`.

        Returns
        -------
        texture_path : `str`
            Absolute filepath to the texture
        """
        # Try find a texture path if we can
        if (self.relative_texture_path is None and not
                self.attempted_texture_search):
            self.relative_texture_path = self._search_for_texture()
        try:
            texture_path = path.join(self.folder, self.relative_texture_path)
        # AttributeError POSIX, TypeError Windows
        except (AttributeError, TypeError):
            return None
        if not path.isfile(texture_path):
            texture_path = None
        return texture_path

    def _parse_format(self):
        r"""
        Abstract method that handles actually building a mesh. This involves
        reading the mesh from disk and doing any necessary parsing.

        Should set the `self.meshes` attribute. Each mesh in `self.meshes`
        is expected to be an object with attributes:

        ======== ==========================
        name     type
        ======== ==========================
        points   double ndarray
        trilist  int ndarray
        tcoords  double ndarray (optional)
        ======== ==========================

        May also set the :attr:`relative_texture_path` if it is specified by
        the format.
        """
        pass

    def build(self):
        r"""
        Overrides the :meth:`build <menpo.io.base.Importer.build>` method.

        Parse the format as defined by :meth:`_parse_format` and then search
        for valid textures and landmarks that may have been defined by the
        format.

        Build the appropriate type of mesh defined by parsing the format. May
        or may not be textured.

        Returns
        -------
        meshes : list of :class:`menpo.shape.mesh.textured.TexturedTriMesh` or :class:`menpo.shape.mesh.base.Trimesh`
            If more than one mesh, returns a list of meshes. If only one
            mesh, returns the single mesh.
        """
        #
        self._parse_format()

        # Only want to create textured meshes if there is a texture path
        # and import_textures is True
        textured = self.import_textures and self.texture_path is not None

        meshes = []
        for mesh in self.meshes:
            if textured:
                new_mesh = TexturedTriMesh(mesh.points,
                                           mesh.tcoords,
                                           import_image(self.texture_path),
                                           trilist=mesh.trilist)
            elif mesh.colour_per_vertex is not None:
                new_mesh = ColouredTriMesh(mesh.points,
                                           colours=mesh.colour_per_vertex,
                                           trilist=mesh.trilist)
            else:
                new_mesh = TriMesh(mesh.points, trilist=mesh.trilist)

            meshes.append(new_mesh)
        if len(meshes) == 1:
            return meshes[0]
        else:
            return meshes


class AssimpImporter(MeshImporter):
    """
    Uses assimp to import meshes. The assimp importing is wrapped via cython,

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """
    def __init__(self, filepath, texture=True):
        from cyassimp import AIImporter  # expensive import
        MeshImporter.__init__(self, filepath, texture=texture)
        self.ai_importer = AIImporter(filepath)

    def _parse_format(self):
        r"""
        Use assimp to build the mesh and get the relative texture path.
        """
        self.ai_importer.build_scene()
        self.meshes = self.ai_importer.meshes
        self.relative_texture_path = self.ai_importer.assimp_texture_path


class WRLImporter(MeshImporter):
    """Allows importing VRML 2.0 meshes.

    Uses VTK and assumes that the first actor in the scene is the one
    that we want.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath, texture=True):
        # Setup class before super class call
        super(WRLImporter, self).__init__(filepath, texture=texture)

    def _parse_format(self):
        r"""
        Use VTK to parse the file and build a mesh object. A single shape per
        scene is assumed.

        Raises
        ------
        MeshImportError
            If no transform or shape is found in the scenegraph
        """
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        vrml_importer = vtk.vtkVRMLImporter()
        vrml_importer.SetFileName(self.filepath)
        vrml_importer.Update()

        # Get the first actor.
        actors = vrml_importer.GetRenderer().GetActors()
        actors.InitTraversal()
        mapper = actors.GetNextActor().GetMapper()
        mapper_dataset = mapper.GetInput()

        if actors.GetNextActor():
            # There was more than one actor!
            warnings.warn('More than one actor was detected in the scene. Only '
                          'single scene actors are currently supported.')

        # Get the Data
        polydata = vtk.vtkPolyData.SafeDownCast(mapper_dataset)
        polydata.Update()

        # We must have point data!
        points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

        # We may have connectivity data. And this connectivity data may
        # need coercing into pure triangles (since we only support triangular
        # meshes at the moment
        try:
            trilist = vtk_to_numpy(polydata.GetPolys().GetData())

            # 5 is the triangle type - if we have another type we need to
            # use a vtkTriangleFilter
            if polydata.GetNumberOfTypes() != 1 or polydata.GetCellType(0) != 5:
                warnings.warn('Non-triangular mesh connectivity was detected - '
                              'this is currently unsupported and thus the '
                              'connectivity is being coerced into a triangular '
                              'mesh. This may have unintended consequences.')
                t_filter = vtk.vtkTriangleFilter()
                t_filter.SetInput(polydata)
                t_filter.Update()
                trilist = vtk_to_numpy(t_filter.GetOutput().GetPolys().GetData())
        except Exception as e:
            pass

        # Three different outcomes - either we have a textured mesh, a coloured
        # mesh or just a plain mesh. Let's try each in turn.
        tcoords, colour_per_vertex = None, None
        # Textured
        try:
            tcoords = vtk_to_numpy(polydata.GetPointData().GetTCoords())
        except Exception as e:
            pass

        # Color-per-vertex
        if tcoords is None:
            try:
                colour_per_vertex = vtk_to_numpy(
                    mapper.GetLookupTable().GetTable()) / 255.
            except Exception as e:
                pass

        # Assumes a single mesh per file
        self.mesh = MeshInfo(points,
                             trilist.reshape([-1, 4])[:, 1:],
                             tcoords,
                             colour_per_vertex)
        self.meshes = [self.mesh]


class MJSONImporter(MeshImporter):
    """
    Import meshes that are in a simple JSON format.

    """

    def _parse_format(self):
        with open(self.filepath, 'rb') as f:
            mesh_json = json.load(f)
        mesh = MeshInfo(mesh_json['points'], mesh_json['trilist'],
                        mesh_json.get('tcoords'),
                        mesh_json.get('colour_per_vertex'))
        self.meshes = [mesh]
