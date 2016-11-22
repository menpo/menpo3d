import numpy as np
import scipy.sparse as sp
from menpo.shape import TriMesh, PointCloud
from menpo.transform import Translation, UniformScale, AlignmentSimilarity
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator
from menpo3d.morphablemodel.shapemodel import ShapeModel


try:
    
    try:
        # First try the newer scikit-sparse namespace
        from sksparse.cholmod import cholesky_AAt
    except ImportError:
        # Fall back to the older scikits.sparse namespace
        from scikits.sparse.cholmod import cholesky_AAt

    # user has cholesky available - provide a fast solve
    def spsolve(sparse_X, dense_b):
        factor = cholesky_AAt(sparse_X.T)
        return factor(sparse_X.T.dot(dense_b)).toarray()

except ImportError:
    # fallback to (much slower) scipy solve
    from scipy.sparse.linalg import spsolve as scipy_spsolve

    def spsolve(sparse_X, dense_b):
        return scipy_spsolve(sparse_X.T.dot(sparse_X),
                             sparse_X.T.dot(dense_b)).toarray()


def node_arc_incidence_matrix(source):
    unique_edge_pairs = source.unique_edge_indices()
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    row = np.hstack((np.arange(m), np.arange(m)))
    col = unique_edge_pairs.T.ravel()
    data = np.hstack((-1 * np.ones(m), np.ones(m)))
    return sp.coo_matrix((data, (row, col))), unique_edge_pairs


def non_rigid_icp(source, target, eps=1e-3, stiffness_weights=None,
                  verbose=False, landmarks=None, lm_weight=None,
                  generate_instances=False, model_landmarks=None):
    # call the generator version of NICP, always returning a generator
    results = non_rigid_icp_generator(source, target, eps=eps,
                                      stiffness_weights=stiffness_weights,
                                      verbose=verbose, landmarks=landmarks,
                                      lm_weights=lm_weight,
                                      generate_instances=True,
                                      model_mean_landmarks=model_landmarks)
    if generate_instances:
        # the user wants to inspect results per-iteration - return the iterator
        # directly to them
        return results
    else:
        # the user is not interested in per-iteration results. Exhaust the
        # generator ourselves and return the last result only.
        while True:
            try:
                instance = next(results)
            except StopIteration:
                return instance[0]


def non_rigid_icp_generator(source, target, eps=1e-3,
                            stiffness_weights=None,
                            landmarks=None, lm_weights=None, verbose=False,
                            generate_instances=False,
                            model_mean_landmarks=None):
    r"""
    Deforms the source trimesh to align with to optimally the target.
    """
    # If the source is a PCA model, perform active NICP.
    active = hasattr(source, 'components')

    if active:
        if verbose:
            print("PCAModel provided as source - performing Active NICP")
        model = source
        model_mean = model.mean()

        if landmarks is not None:

            # user better have provided model landmarks!
            if model_mean_landmarks is None:
                raise ValueError(
                    'For Active NICP with landmarks the model_mean_landmarks '
                    'need to be provided.')

            shape_model = ShapeModel(model)
            source_lms = model_mean_landmarks
            target_lms = target.landmarks[landmarks].lms
            model_lms_index = model_mean.distance_to(source_lms).argmin(axis=0)
            shape_model_lms = shape_model.mask_points(model_lms_index)

            # Sim align the target lms to the mean before projecting
            target_lms_aligned = AlignmentSimilarity(target_lms,
                                                     source_lms).apply(
                target_lms)

            # project to learn the weights for the landmark model
            weights = shape_model_lms.project(target_lms_aligned,
                                              n_components=20)
            # use these weights on the dense shape model to generate an improved
            # instance

            source = model.instance(weights)
            # update the source landmarks (for the alignment below)
            source.landmarks[landmarks] = PointCloud(source.points[
                                                      model_lms_index])
        else:
            # Start from the mean of the model
            source = model_mean

    # Scale factors completely change the behavior of the algorithm - always
    # rescale the source down to a sensible size (so it fits inside box of
    # diagonal 1) and is centred on the origin. We'll undo this after the fit
    # so the user can use whatever scale they prefer.
    if landmarks is not None:
        if verbose:
            print("'{}' landmarks will be used as a landmark constraint.".format(landmarks))
            print("Performing similarity alignment using landmarks")
        lm_align = AlignmentSimilarity(source.landmarks[landmarks].lms,
                                       target.landmarks[landmarks].lms).as_non_alignment()
        source = lm_align.apply(source)

    tr = Translation(-1 * source.centre())
    sc = UniformScale(1.0 / np.sqrt(np.sum(source.range() ** 2)), 3)
    prepare = tr.compose_before(sc)

    source = prepare.apply(source)
    target = prepare.apply(target)

    # store how to undo the similarity transform
    restore = prepare.pseudoinverse()

    n_dims = source.n_dims
    # Homogeneous dimension (1 extra for translation effects)
    h_dims = n_dims + 1
    points, trilist = source.points, source.trilist
    n = points.shape[0]  # record number of points

    edge_tris = source.boundary_tri_index()

    M_s, unique_edge_pairs = node_arc_incidence_matrix(source)

    # weight matrix
    G = np.identity(n_dims + 1)

    M_kron_G_s = sp.kron(M_s, G)

    # build octree for finding closest points on target.
    target_vtk = trimesh_to_vtk(target)
    closest_points_on_target = VTKClosestPointLocator(target_vtk)

    # save out the target normals. We need them for the weight matrix.
    target_tri_normals = target.tri_normals()

    # init transformation
    X_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
    v_i = points

    if stiffness_weights is not None:
        stiffness = stiffness_weights
        if verbose:
            print('using user defined stiffness weights')
    else:
        # these values have been empirically found to perform well for well
        # rigidly aligned facial meshes
        stiffness = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
        if verbose:
            print('using default stiffness weights: {}'.format(stiffness))

    if lm_weights is not None:
        lm_weights = lm_weights
        if verbose:
            print('using user defined lm_weight values: {}'.format(lm_weights))
    else:
        # these values have been empirically found to perform well for well
        # rigidly aligned facial meshes
        lm_weights = [5, 2, .5, 0, 0, 0, 0, 0]
        if verbose:
            print('using default lm_weight values: {}'.format(lm_weights))

    # we need to prepare some indices for efficient construction of the D
    # sparse matrix.
    row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(),
                     np.arange(n)))

    x = np.arange(n * h_dims).reshape((n, h_dims))
    col = np.hstack((x[:, :n_dims].ravel(),
                     x[:, n_dims]))
    o = np.ones(n)

    if landmarks is not None:
        source_lm_index = source.distance_to(source.landmarks[landmarks].lms).argmin(axis=0)
        target_lms = target.landmarks[landmarks].lms
        U_L = target_lms.points
        n_landmarks = target_lms.n_points
        lm_mask = np.in1d(row, source_lm_index)
        col_lm = col[lm_mask]
        # pull out the rows for the lms - but the values are
        # all wrong! need to map them back to the order of the landmarks
        row_lm_to_fix = row[lm_mask]
        source_lm_index_l = list(source_lm_index)
        row_lm = np.array([source_lm_index_l.index(r) for r in row_lm_to_fix])

    for i, (alpha, beta) in enumerate(zip(stiffness, lm_weights), 1):
        alpha_is_per_vertex = isinstance(alpha, np.ndarray)
        if alpha_is_per_vertex:
            # stiffness is provided per-vertex
            alpha_per_edge = alpha[unique_edge_pairs].mean(axis=1)
            alpha_M_s = sp.diags(alpha_per_edge).dot(M_s)
            alpha_M_kron_G_s = sp.kron(alpha_M_s, G)
        else:
            # stiffness is global - just a scalar multiply. Note that here
            # we don't have to recalculate M_kron_G_s
            alpha_M_kron_G_s = alpha * M_kron_G_s

        if verbose:
            a_str = (alpha if not alpha_is_per_vertex
                     else 'min: {:.2f}, max: {:.2f}'.format(alpha.min(),
                                                            alpha.max()))
            i_str = '{}/{}: stiffness: {}'.format(i, len(stiffness), a_str)
            if landmarks is not None:
                i_str += '  lm_weight: {}'.format(beta)
            print(i_str)

        j = 0
        while True:  # iterate until convergence
            # find nearest neighbour and the normals
            U, tri_indices = closest_points_on_target(v_i)

            # ---- WEIGHTS ----
            # 1.  Edges
            # Are any of the corresponding tris on the edge of the target?
            # Where they are we return a false weight (we *don't* want to
            # include these points in the solve)
            w_i_e = np.in1d(tri_indices, edge_tris, invert=True)

            # 2. Normals
            # Calculate the normals of the current v_i
            v_i_tm = TriMesh(v_i, trilist=trilist)
            v_i_n = v_i_tm.vertex_normals()
            # Extract the corresponding normals from the target
            u_i_n = target_tri_normals[tri_indices]
            # If the dot of the normals is lt 0.9 don't contrib to deformation
            w_i_n = (u_i_n * v_i_n).sum(axis=1) > 0.9

            # 3. Self-intersection
            # This adds approximately 12% to the running cost and doesn't seem
            # to be very critical in helping mesh fitting performance so for
            # now it's removed. Revisit later.
            # # Build an intersector for the current deformed target
            # intersect = build_intersector(to_vtk(v_i_tm))
            # # budge the source points 1% closer to the target
            # source = v_i + ((U - v_i) * 0.5)
            # # if the vector from source to target intersects the deformed
            # # template we don't want to include it in the optimisation.
            # problematic = [i for i, (s, t) in enumerate(zip(source, U))
            #                if len(intersect(s, t)[0]) > 0]
            # print(len(problematic) * 1.0 / n)
            # w_i_i = np.ones(v_i_tm.n_points, dtype=np.bool)
            # w_i_i[problematic] = False

            # Form the overall w_i from the normals, edge case
            w_i = np.logical_and(w_i_n, w_i_e)
            # we could add self intersection at a later date too...
            # w_i = np.logical_and(np.logical_and(w_i_n, w_i_e), w_i_i)

            prop_w_i = (n - w_i.sum() * 1.0) / n
            prop_w_i_n = (n - w_i_n.sum() * 1.0) / n
            prop_w_i_e = (n - w_i_e.sum() * 1.0) / n
            j += 1

            # Build the sparse diagonal weight matrix
            W_s = sp.diags(w_i.astype(np.float)[None, :], [0])

            data = np.hstack((v_i.ravel(), o))
            D_s = sp.coo_matrix((data, (row, col)))

            to_stack_A = [alpha_M_kron_G_s, W_s.dot(D_s)]
            to_stack_B = [np.zeros((alpha_M_kron_G_s.shape[0], n_dims)),
                          U * w_i[:, None]]  # nullify nearest points by w_i

            if landmarks is not None:
                D_L = sp.coo_matrix((data[lm_mask], (row_lm, col_lm)),
                                    shape=(n_landmarks, D_s.shape[1]))
                to_stack_A.append(beta * D_L)
                to_stack_B.append(beta * U_L)

            A_s = sp.vstack(to_stack_A).tocsr()
            B_s = sp.vstack(to_stack_B).tocsr()
            X = spsolve(A_s, B_s)

            # deform template
            v_i_prev = v_i
            v_i = D_s.dot(X)
            delta_v_i = v_i - v_i_prev

            # project onto the shape model to restrict the basis
            if active:
                v_i = prepare.apply(model.reconstruct(
                    restore.apply(source.from_vector(v_i.ravel())))).points

            err = np.linalg.norm(X_prev - X, ord='fro')
            stop_criterion = err / np.sqrt(np.size(X_prev))

            if landmarks is not None:
                src_lms = v_i[source_lm_index]
                lm_err = np.sqrt((src_lms - U_L) ** 2).sum(axis=1).mean()

            if verbose:
                v_str = (' - {} stop crit: {:.3f}  '
                         'total: {:.0%}  norms: {:.0%}  '
                         'edges: {:.0%}'.format(j, stop_criterion,
                                                prop_w_i, prop_w_i_n,
                                                prop_w_i_e))
                if landmarks is not None:
                    v_str += '  lm_err: {:.4f}'.format(lm_err)

                print(v_str)

            X_prev = X

            # track the progress of the algorithm per-iteration
            info_dict = {
                'alpha': alpha,
                'iteration': j + 1,
                'prop_omitted': prop_w_i,
                'prop_omitted_norms': prop_w_i_n,
                'prop_omitted_edges': prop_w_i_e,
                'delta': err,
                'mask_normals': w_i_n,
                'mask_edges': w_i_e,
                'mask_all': w_i,
                'nearest_points': restore.apply(U),
                'deformation_per_step': delta_v_i
            }

            current_instance = source.copy()
            current_instance.points = v_i.copy()

            if landmarks:
                info_dict['beta'] = beta
                info_dict['lm_err'] = lm_err
                current_instance.landmarks[landmarks] = PointCloud(src_lms)

            yield restore.apply(current_instance), info_dict

            if stop_criterion < eps:
                break
