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
