import numpy as np
import pytest

import menpo3d.io as m3io
from menpo.transform import Translation
from menpo3d.correspond import non_rigid_icp


def test_non_rigid_icp_moves_a_mesh_onto_a_translated_copy():
    # np.in1d, used to mask the landmarks and edges, is deprecated by NumPy 2
    source = m3io.import_builtin_asset.bunny_obj()
    target = Translation([0.01, 0.0, 0.0]).apply(source)
    result = non_rigid_icp(source, target, eps=1e-3)
    extent = np.ptp(target.points, axis=0).max()
    assert np.abs(result.points - target.points).max() < 0.05 * extent
    assert result.n_points == source.n_points


# ---- stdout_redirected: no closing of sys.stdout (it hung in a notebook)
import os
import sys

import scipy.sparse as sp
from menpo3d.correspond import nicp


def test_stdout_redirected_swallows_low_level_writes_and_never_closes_sys_stdout(
    capfd,
):
    stdout = sys.stdout
    with nicp.stdout_redirected():
        os.write(1, b"from C\n")  # what a C library does
        assert sys.stdout is stdout and not stdout.closed
    assert sys.stdout is stdout and not stdout.closed
    os.write(1, b"after, low level\n")
    out = capfd.readouterr().out
    assert "from C" not in out
    assert "after, low level" in out


def test_stdout_redirected_restores_stdout_when_the_block_raises(capfd):
    with pytest.raises(RuntimeError):
        with nicp.stdout_redirected():
            raise RuntimeError("boom")
    os.write(1, b"still works\n")
    assert "still works" in capfd.readouterr().out


def test_stdout_redirected_sends_the_output_to_a_file(tmp_path):
    target = tmp_path / "out.txt"
    with nicp.stdout_redirected(to=str(target)):
        os.write(1, b"into the file\n")
    assert "into the file" in target.read_text()


# ---- spsolve: the damping factor
def _rank_deficient_problem():
    # the second column is the first one: X^T X is singular
    rs = np.random.RandomState(0)
    col = rs.randn(20)
    X = sp.csr_matrix(np.column_stack([col, col, rs.randn(20)]))
    b = sp.csr_matrix(rs.randn(20, 2))
    return X, b


@pytest.fixture
def no_cholmod(monkeypatch):
    monkeypatch.setattr(nicp, "_cholesky_AAt", None)


def test_spsolve_with_damping_solves_the_regularised_normal_equations(no_cholmod):
    X, b = _rank_deficient_problem()
    with pytest.warns(UserWarning, match="suitesparse"):
        x = nicp.spsolve(X, b, damping=1e-3)
    dense_X, dense_b = X.toarray(), b.toarray()
    expected = np.linalg.solve(
        dense_X.T @ dense_X + 1e-3 * np.eye(3), dense_X.T @ dense_b
    )
    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-8)


def test_spsolve_without_damping_is_the_plain_normal_equations(no_cholmod):
    rs = np.random.RandomState(1)
    X, b = sp.csr_matrix(rs.randn(30, 4)), sp.csr_matrix(rs.randn(30, 2))
    with pytest.warns(UserWarning, match="suitesparse"):
        x = nicp.spsolve(X, b)
    dense_X, dense_b = X.toarray(), b.toarray()
    np.testing.assert_allclose(
        x, np.linalg.lstsq(dense_X, dense_b, rcond=None)[0], atol=1e-8
    )


def test_spsolve_gives_the_damping_to_cholmod_as_beta(monkeypatch):
    calls = {}

    class Factor:
        def __init__(self, A, beta):
            self.A, self.beta = A, beta

        def __call__(self, rhs):
            # solves (A A^T + beta I) x = rhs, as CHOLMOD does
            n = self.A.shape[0]
            normal = (self.A @ self.A.T + self.beta * sp.identity(n)).tocsc()
            return sp.csr_matrix(
                np.linalg.solve(normal.toarray(), np.asarray(rhs.todense()))
            )

    def fake_cholesky_AAt(A, beta=0):
        calls["beta"] = beta
        return Factor(A, beta)

    monkeypatch.setattr(nicp, "_cholesky_AAt", fake_cholesky_AAt)
    X, b = _rank_deficient_problem()
    x = nicp.spsolve(X, b, damping=1e-3)
    assert calls["beta"] == 1e-3
    dense_X, dense_b = X.toarray(), b.toarray()
    np.testing.assert_allclose(
        x,
        np.linalg.solve(dense_X.T @ dense_X + 1e-3 * np.eye(3), dense_X.T @ dense_b),
        rtol=1e-6,
        atol=1e-8,
    )


def test_non_rigid_icp_passes_the_damping_to_every_solve(monkeypatch, no_cholmod):
    seen = []
    real = nicp.spsolve
    monkeypatch.setattr(
        nicp,
        "spsolve",
        lambda X, b, damping=0.0: seen.append(damping) or real(X, b, damping),
    )
    source = m3io.import_builtin_asset.bunny_obj()
    target = Translation([0.01, 0.0, 0.0]).apply(source)
    with pytest.warns(UserWarning):
        nicp.non_rigid_icp(source, target, stiffness_weights=[10, 5], damping=1e-6)
    assert len(seen) >= 2 and set(seen) == {1e-6}
    seen.clear()
    with pytest.warns(UserWarning):
        nicp.non_rigid_icp(source, target, stiffness_weights=[10, 5])
    assert len(seen) >= 2 and set(seen) == {0.0}


def test_stdout_redirected_in_a_fresh_process_swallows_python_prints_too():
    import subprocess

    code = (
        "import sys\n"
        "from menpo3d.correspond import nicp\n"
        "with nicp.stdout_redirected():\n"
        "    print('hidden')\n"
        "    sys.stdout.write('hidden too\\n')\n"
        "print('shown')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "shown\n"


def test_flushing_the_c_library_never_raises(monkeypatch):
    # whatever the platform, a C library that can not be found or loaded is
    # nothing to flush, not an error (CDLL(None) is a TypeError on Windows)
    import ctypes

    def cannot_load(name):
        raise TypeError("LoadLibrary() argument 1 must be str, not None")

    monkeypatch.setattr(ctypes, "CDLL", cannot_load)
    nicp._flush_all_stdout()
    monkeypatch.setattr(
        ctypes, "CDLL", lambda name: (_ for _ in ()).throw(OSError(name))
    )
    nicp._flush_all_stdout()


# ---- the edge mask: which nearest triangles of the *target* are on its boundary
def _grid(n, z=0.0):
    # a flat n x n grid with its centre at the origin and a bounding box
    # diagonal of 1, so NICP's own normalisation (centre, scale to a diagonal of
    # 1) changes nothing and the points it sees are these ones
    from menpo.shape import TriMesh

    mesh = TriMesh.init_2d_grid((n, n))
    xy = mesh.points - mesh.points.mean(axis=0)
    xy = xy / np.sqrt((np.ptp(xy, axis=0) ** 2).sum())
    return TriMesh(np.column_stack([xy, np.full(len(xy), z)]), mesh.trilist, copy=False)


def test_the_edge_mask_is_about_the_boundary_triangles_of_the_target():
    from menpo3d.closest_point import ClosestPointLocator

    # a 7 x 7 source over a 5 x 5 target: different triangulations, so a mask
    # taken from the source's triangles would be wrong for the target's
    target = _grid(5)
    source = _grid(7, z=0.05)
    nearest_tri = ClosestPointLocator.from_trimesh(target)(source.points)[1]
    expected_edge = target.boundary_tri_index()[nearest_tri]
    assert expected_edge.any() and not expected_edge.all()  # a real test

    _, info = next(
        non_rigid_icp(source, target, stiffness_weights=[10], generate_instances=True)
    )
    np.testing.assert_array_equal(info["mask_edges"], ~expected_edge)
    assert info["prop_omitted_edges"] == pytest.approx(expected_edge.mean())
    # the edge constraint is not applied (it was found to be noisy): the
    # weights are the normals' only
    np.testing.assert_array_equal(info["mask_all"], info["mask_normals"])
