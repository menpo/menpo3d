from pathlib import Path
from glob import glob

SHADERS_DIR = Path(__file__).parents[1] / "shaders"


def test_n_expected_builtin_shaders():
    all_shaders = glob(str(SHADERS_DIR / "**/*.*"), recursive=True)
    assert len(all_shaders) == 4, all_shaders


def test_all_per_vertex_shaders_exist():
    per_vertex_shaders = SHADERS_DIR / "per_vertex"
    assert [p.name for p in sorted(per_vertex_shaders.glob("*"))] == [
        "passthrough.frag",
        "passthrough.vert",
    ]


def test_all_texture_shaders_exist():
    per_vertex_shaders = SHADERS_DIR / "texture"
    assert [p.name for p in sorted(per_vertex_shaders.glob("*"))] == [
        "passthrough.frag",
        "passthrough.vert",
    ]
