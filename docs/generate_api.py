"""Write docs/source/api/*.rst and docs/xref_map.py from the public API of menpo3d.

Run it from the repository root after adding or removing public classes or functions:

    python docs/generate_api.py

The cross-references of menpo's own classes (``:map:`TriMesh```) are taken from menpo's
docs/xref_map.py (a menpo checkout next to this one, or ``MENPO_DOCS`` pointing to it).
"""

import importlib
import inspect
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("ETS_TOOLKIT", "null")

DOCS = Path(__file__).parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

MODULES = [
    "barycentric",
    "camera",
    "correspond",
    "extractimage",
    "io",
    "math",
    "morphablemodel",
    "rasterize",
    "result",
    "unwrap",
    "visualize",
    "vtkutils",
]


def public_members(module):
    """The classes and functions of menpo3d that the module exposes, sorted."""
    classes, functions = [], []
    for name, value in vars(module).items():
        if name.startswith("_") or not getattr(value, "__module__", "").startswith("menpo3d"):
            continue
        if inspect.isclass(value):
            classes.append(name)
        elif inspect.isfunction(value):
            functions.append(name)
    return sorted(classes), sorted(functions)


def write_api_pages():
    members = {}
    (DOCS / "source" / "api").mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        module = importlib.import_module(f"menpo3d.{name}")
        classes, functions = public_members(module)
        members[name] = {n: "class" for n in classes} | {n: "function" for n in functions}
        title = f"menpo3d.{name}"
        lines = [title, "=" * len(title), "", f".. currentmodule:: {title}", ""]
        for cls in classes:
            lines += [
                cls,
                "-" * len(cls),
                f".. autoclass:: {cls}",
                "  :members:",
                "  :inherited-members:",
                "  :show-inheritance:",
                "",
            ]
        for func in functions:
            lines += [func, "-" * len(func), f".. autofunction:: {func}", ""]
        (DOCS / "source" / "api" / f"{name}.rst").write_text("\n".join(lines))
    return members


# classes of menpo that its own xref_map does not list
EXTRA_MENPO = {"LandmarkGroup": ("class", "menpo.landmark.LandmarkGroup")}


def menpo_xref_map():
    candidates = [os.environ.get("MENPO_DOCS"), ROOT.parent / "menpo" / "docs"]
    for path in candidates:
        if path and (Path(path) / "xref_map.py").exists():
            namespace = {}
            exec((Path(path) / "xref_map.py").read_text(), namespace)
            return {**namespace["xref_map"], **EXTRA_MENPO}
    raise SystemExit("menpo's docs/xref_map.py not found: set MENPO_DOCS")


def write_xref_map(members):
    used = set()
    for path in (ROOT / "menpo3d").rglob("*.py"):
        used |= set(re.findall(r":map:`([^`\n]+)`", path.read_text()))
    menpo = menpo_xref_map()
    # everything menpo documents (the methods that menpo3d's classes inherit use these too)
    xref, missing = {k: tuple(v) for k, v in menpo.items()}, []
    for name in sorted(used):
        short = name.split(".")[-1]
        own = [m for m in MODULES if short in members[m]]
        if own:  # menpo3d's own: the first of the documented modules that exposes it
            xref[short] = (members[own[0]][short], f"menpo3d.{own[0]}.{short}")
        elif short in menpo:
            xref[short] = tuple(menpo[short])
        else:
            missing.append(name)
    lines = ["xref_map = {"]
    lines += [f"    {k!r}: ({v[0]!r}, {v[1]!r})," for k, v in sorted(xref.items())]
    lines.append("}")
    (DOCS / "xref_map.py").write_text("\n".join(lines) + "\n")
    return missing


if __name__ == "__main__":
    api = write_api_pages()
    missing = write_xref_map(api)
    print(f"wrote {len(api)} API pages and the cross-reference map")
    if missing:
        print("no target for (add them to menpo's xref_map or fix the docstring):", missing)
