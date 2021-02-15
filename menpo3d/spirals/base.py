def adj_trigs(list_meshes):
    r"""
    Return a list of adjancency lists and a list of faces per vertex list
    for each mesh in the list

    Parameters
    ----------
    list_meshes : list of TriMesh

    Returns
    -------
    Adj: list
         for each Trimesh in list_meshes, find its adjancency list
         and add it in this list
    Trigs: list
         for each Trimesh in list_meshes, for each vertex, find its faces
         (a list of (3,) lists) and add it in this list
    """

    Adj = []
    Trigs = []

    for mesh in list_meshes:
        Adj.append(mesh.as_pointgraph().get_adjacency_list())
        Trigs.append(mesh.list_faces_per_vertex())
    return Adj, Trigs

def generate_transform_matrices(reference_mesh, ds_factors, M=None):
    """Return the lists of upsampling and downsampling matrices
       for a list of meshes defined either by M list or computed using
       the ds_factors

       Parameters:
       --------------
       reference_mesh: TriMesh
       The template mesh
       ds_factors: list
       A list with integers representing the factors of vertices to be kept
       As our decimate method uses the vertices to be removed, we simply
       transformed them.
       M: list of TriMesh
       A list of decimated meshes. If None, used ds_factor to compute this
       list

       Returns:
       -------------
       M: list of TriMesh
       The list of the decimated matrices
       D: list of sparse arrays
       The list of downsampling sparce matrices
       A: list of sparse arrays
       The list of upsampling sparce matrices
    """
    ds_factors = list(map(lambda x: 1-1.0/x, ds_factors))
    D, U = [], []

    if M is None:
        M = []
        M.append(reference_mesh)
        for factor in ds_factors:
            M.append(M[-1].decimate(factor))

    current_mesh = reference_mesh
    for current_mesh, next_mesh in zip(M, M[1:]):
        U.append(current_mesh.upsampling_matrix(next_mesh))
        D.append(next_mesh.find_closest_vertices(current_mesh))

    return M, D, U
