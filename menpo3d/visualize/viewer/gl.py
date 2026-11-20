"""moderngl renderer for a :class:`Scene`: shaded / textured / vertex coloured meshes, wireframe, points, line sets with a
width in pixels, sphere markers (ray-cast impostors, so any size and any count), arrow glyphs, alpha. One GL context
renders for the offline PNG, the notebook widget and the standalone window, so they cannot differ.

GPU objects are built lazily from the scene's NumPy data and freed when the last holder releases the scene.
"""

import numpy as np

_MESH_VS = """#version 330
uniform mat4 mvp; uniform mat3 nmat; uniform float psize;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in vec2 in_uv;
out vec3 n; out vec3 col; out vec2 uv;
void main(){ n = nmat*in_nrm; col = in_col; uv = in_uv; gl_Position = mvp*vec4(in_pos,1.0); gl_PointSize = psize; }"""

_MESH_FS = """#version 330
uniform int mode;   // 0 shaded, 1 flat (no lighting), 2 shaded + texture
uniform sampler2D tex; uniform float alpha; uniform float ambient; uniform float specular;
in vec3 n; in vec3 col; in vec2 uv; out vec4 frag;
void main(){
  vec3 base = mode == 2 ? texture(tex, vec2(uv.x, 1.0 - uv.y)).rgb : col;
  if (mode == 1) { frag = vec4(base, alpha); return; }
  vec3 nn = normalize(n), l = normalize(vec3(0.3, 0.5, 1.0));
  float d = abs(dot(nn, l));
  float s = specular * pow(abs(dot(nn, normalize(l + vec3(0, 0, 1)))), 40.0);
  frag = vec4(clamp(base * (0.35 + ambient + 0.70 * d) + vec3(s), 0.0, 1.0), alpha);
}"""

_LINE_VS = """#version 330
uniform mat4 mvp; uniform vec2 viewport; uniform float width; uniform float bias;
in vec2 corner; in vec3 p0; in vec3 p1; in vec3 c0; in vec3 c1; out vec3 vcol;
void main(){
  vcol = mix(c0, c1, corner.x);
  vec4 a = mvp*vec4(p0,1.0), b = mvp*vec4(p1,1.0);
  if (a.w < 1e-4 || b.w < 1e-4) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 sa = a.xy/a.w*viewport*0.5, sb = b.xy/b.w*viewport*0.5;
  vec2 d = sb - sa; float len = length(d);
  d = len > 1e-6 ? d/len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-d.y, d.x);
  vec4 p = mix(a, b, corner.x);
  vec2 off = (nrm*corner.y + d*(corner.x*2.0-1.0)*0.5) * width*0.5 / (viewport*0.5);
  gl_Position = vec4(p.xy + off*p.w, p.z - bias*p.w, p.w);
}"""

_LINE_FS = """#version 330
in vec3 vcol; uniform float alpha; out vec4 frag;
void main(){ frag = vec4(vcol, alpha); }"""

_SPHERE_VS = """#version 330
uniform mat4 view; uniform mat4 proj; uniform float radius; uniform float viewport_h; uniform float max_px;
in vec3 in_pos; out vec3 centre;
void main(){
  vec4 v = view*vec4(in_pos,1.0); centre = v.xyz;
  gl_Position = proj*v;
  gl_PointSize = clamp(2.0*radius*abs(proj[1][1])/max(gl_Position.w, 1e-6)*viewport_h*0.5, 1.0, max_px);
}"""

_SPHERE_FS = """#version 330
uniform mat4 proj; uniform float radius; uniform vec3 colour; uniform float alpha; uniform float ambient;
in vec3 centre; out vec4 frag;
void main(){
  vec2 c = gl_PointCoord*2.0 - 1.0; float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  vec3 nn = vec3(c.x, c.y, sqrt(1.0 - r2));   // the render is y-flipped for a top-down readback: view-up is +c.y
  vec4 clip = proj*vec4(centre + nn*radius, 1.0);
  gl_FragDepth = (clip.z/clip.w)*0.5 + 0.5;
  float d = abs(dot(nn, normalize(vec3(0.3, 0.5, 1.0))));
  frag = vec4(clamp(colour*(0.35 + ambient + 0.70*d) + vec3(0.4*pow(d, 6.0)), 0.0, 1.0), alpha);
}"""

_ARROW_VS = """#version 330
uniform mat4 mvp; uniform mat3 nmat; uniform float scale; uniform float thick;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_origin; in vec3 in_vec; out vec3 n;
void main(){
  float len = length(in_vec)*scale;
  if (len < 1e-12) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); n = vec3(0, 0, 1); return; }
  vec3 d = in_vec/length(in_vec);
  vec3 u = abs(d.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0);
  vec3 x = normalize(cross(u, d)), y = cross(d, x);
  vec3 world = in_origin + (x*in_pos.x*thick + y*in_pos.y*thick + d*in_pos.z)*len;
  n = nmat*(x*in_nrm.x + y*in_nrm.y + d*in_nrm.z);
  gl_Position = mvp*vec4(world, 1.0);
}"""

_ARROW_FS = """#version 330
uniform vec3 colour; uniform float alpha; in vec3 n; out vec4 frag;
void main(){ float d = abs(dot(normalize(n), normalize(vec3(0.3, 0.5, 1.0)))); frag = vec4(colour*(0.3 + 0.7*d), alpha); }"""


def _arrow_geometry(sides=8):
    """Unit arrow along +z: a thin shaft (radius 0.03 of the length) and a cone head (radius 0.09)."""
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    verts, norms, tris = [], [], []

    def ring(radius, z, nz):
        base = len(verts)
        for i in range(sides):
            verts.append((radius * c[i], radius * s[i], z))
            nv = np.array([c[i], s[i], nz])
            norms.append(nv / np.linalg.norm(nv))
        return base

    shaft_b, shaft_t = ring(0.03, 0.0, 0.0), ring(0.03, 0.75, 0.0)
    head_b, head_t = ring(0.09, 0.75, 0.5), ring(0.0, 1.0, 0.5)
    for lo, hi in ((shaft_b, shaft_t), (head_b, head_t)):
        for i in range(sides):
            j = (i + 1) % sides
            tris += [(lo + i, lo + j, hi + i), (lo + j, hi + j, hi + i)]
    return np.hstack([verts, norms]).astype("f4"), np.array(tris, "i4")


class GLRenderer:
    MAX_FBOS = 6

    def __init__(self, backend=None, samples=4):
        import moderngl

        from menpo3d._gl import create_context

        self.mgl = moderngl
        ctx = create_context(backend)
        self.ctx = ctx
        self.gl_info = ctx.info["GL_RENDERER"]
        self.samples = samples if samples and samples > 1 else 0
        ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.max_point = float(ctx.info.get("GL_POINT_SIZE_RANGE", (1, 64))[1])
        self.p_mesh = ctx.program(vertex_shader=_MESH_VS, fragment_shader=_MESH_FS)
        self.p_line = ctx.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        self.p_sphere = ctx.program(
            vertex_shader=_SPHERE_VS, fragment_shader=_SPHERE_FS
        )
        self.p_arrow = ctx.program(vertex_shader=_ARROW_VS, fragment_shader=_ARROW_FS)
        self.corner = ctx.buffer(
            np.array([[0, -1], [0, 1], [1, -1], [1, 1]], "f4").tobytes()
        )
        geo, idx = _arrow_geometry()
        self.arrow_vbo, self.arrow_ibo = (
            ctx.buffer(geo.tobytes()),
            ctx.buffer(idx.tobytes()),
        )
        self.arrow_n = len(idx) * 3
        self._fbos = {}
        self._gpu = {}  # scene id -> dict(version, drawables, objects, refs)

    # ---- GPU data for a scene
    def acquire(self, scene):
        entry = self._gpu.setdefault(
            scene.id, dict(version=None, drawables=[], objects=[], refs=0)
        )
        entry["refs"] += 1
        self._sync(scene, entry)

    def release(self, scene):
        entry = self._gpu.get(scene.id)
        if entry is None or entry["refs"] == 0:
            return
        entry["refs"] -= 1
        if entry["refs"] == 0:
            self._free(entry)
            del self._gpu[scene.id]

    def on_gpu(self):
        return {sid: e["refs"] for sid, e in self._gpu.items()}

    def _free(self, entry):
        for obj in entry["objects"]:
            obj.release()
        entry["objects"], entry["drawables"] = [], []

    def _sync(self, scene, entry):
        if entry["version"] == scene.version:
            return
        self._free(entry)
        items = scene.items
        for item in items:
            self._build(item, entry)
        entry["version"] = scene.version
        entry["items"] = items

    def _buf(self, entry, data):
        b = self.ctx.buffer(data)
        entry["objects"].append(b)
        return b

    def _build(self, item, entry):
        ctx, kind = self.ctx, item["kind"]
        keep = lambda o: (entry["objects"].append(o), o)[1]  # noqa: E731
        if kind == "mesh":
            pts, n = item["points"], len(item["points"])
            tri = item["trilist"]
            if len(tri) and item["mode"] in ("surface",):
                normals = _vertex_normals(pts, tri)
            else:
                normals = np.tile([0, 0, 1.0], (n, 1)).astype("f4")
            col = (
                item["colours"]
                if item["colours"] is not None
                else np.tile(item["colour"], (n, 1)).astype("f4")
            )
            uv = (
                item["tcoords"]
                if item["tcoords"] is not None
                else np.zeros((n, 2), "f4")
            )
            vbo = self._buf(
                entry, np.hstack([pts, normals, col, uv]).astype("f4").tobytes()
            )
            layout = (vbo, "3f 3f 3f 2f", "in_pos", "in_nrm", "in_col", "in_uv")
            tex = None
            if item["texture"] is not None and item["tcoords"] is not None:
                h, w, _ = item["texture"].shape
                tex = keep(ctx.texture((w, h), 3, item["texture"].tobytes()))
                tex.build_mipmaps()
                tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            mode = item["mode"]
            if mode == "surface" and len(tri):
                ibo = self._buf(entry, tri.astype("i4").tobytes())
                vao = keep(ctx.vertex_array(self.p_mesh, [layout], ibo))
                entry["drawables"].append(("surface", item, vao, tex))
            if mode in ("wireframe", "fancymesh") and len(tri):
                edges = np.unique(
                    np.sort(
                        np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]),
                        axis=1,
                    ),
                    axis=0,
                )
                self._build_lines(
                    entry,
                    pts,
                    edges,
                    _vertex_colours(item),
                    item["line_width"],
                    item["alpha"],
                )
            if mode == "points":
                vao = keep(ctx.vertex_array(self.p_mesh, [layout]))
                entry["drawables"].append(("points", item, vao, None))
            if mode == "fancymesh":
                size = item["marker_size"] or _auto_marker_size(pts)
                self._build_markers(
                    entry, pts, size, item["colour"], "sphere", item["alpha"]
                )
        elif kind == "lines":
            self._build_lines(
                entry,
                item["points"],
                item["edges"],
                np.asarray(item["colour"], "f4"),
                item["width"],
                item["alpha"],
            )
        elif kind == "markers":
            self._build_markers(
                entry,
                item["points"],
                item["size"],
                item["colour"],
                item["style"],
                item["alpha"],
            )
        elif kind == "vectors":
            if len(item["points"]):
                inst = self._buf(
                    entry,
                    np.hstack([item["points"], item["vectors"]]).astype("f4").tobytes(),
                )
                vao = keep(
                    ctx.vertex_array(
                        self.p_arrow,
                        [
                            (self.arrow_vbo, "3f 3f", "in_pos", "in_nrm"),
                            (inst, "3f 3f/i", "in_origin", "in_vec"),
                        ],
                        self.arrow_ibo,
                    )
                )
                entry["drawables"].append(("arrows", item, vao, len(item["points"])))

    def _build_lines(self, entry, pts, edges, colour, width, alpha):
        """colour: one RGB triple or an (n, 3) array of per-vertex colours (interpolated along each edge)."""
        if not len(edges):
            return
        col = np.tile(colour, (len(pts), 1)) if np.ndim(colour) == 1 else colour
        seg = np.hstack(
            [pts[edges[:, 0]], pts[edges[:, 1]], col[edges[:, 0]], col[edges[:, 1]]]
        ).astype("f4")
        inst = self._buf(entry, seg.tobytes())
        vao = self.ctx.vertex_array(
            self.p_line,
            [
                (self.corner, "2f", "corner"),
                (inst, "3f 3f 3f 3f/i", "p0", "p1", "c0", "c1"),
            ],
        )
        entry["objects"].append(vao)
        entry["drawables"].append(
            ("lines", dict(width=width, alpha=alpha), vao, len(seg))
        )

    def _build_markers(self, entry, pts, size, colour, style, alpha):
        vbo = self._buf(entry, np.ascontiguousarray(pts, "f4").tobytes())
        vao = self.ctx.vertex_array(self.p_sphere, [(vbo, "3f", "in_pos")])
        entry["objects"].append(vao)
        entry["drawables"].append(
            (
                "spheres",
                dict(colour=colour, size=size, style=style, alpha=alpha),
                vao,
                len(pts),
            )
        )

    # ---- framebuffers
    def _fbo(self, w, h):
        key = (w, h)
        fbo = self._fbos.pop(key, None)
        if fbo is None:
            while len(self._fbos) >= self.MAX_FBOS:
                self._drop_fbo(next(iter(self._fbos)))
            ctx = self.ctx
            s = self.samples
            msaa = (
                ctx.framebuffer(
                    ctx.renderbuffer((w, h), samples=s),
                    ctx.depth_renderbuffer((w, h), samples=s),
                )
                if s
                else None
            )
            flat = ctx.framebuffer(
                ctx.renderbuffer((w, h)), ctx.depth_renderbuffer((w, h))
            )
            fbo = (msaa, flat)
        self._fbos[key] = fbo
        return fbo

    def _drop_fbo(self, key):
        for fb in self._fbos.pop(key):
            if fb is not None:
                for a in (*fb.color_attachments, fb.depth_attachment):
                    a.release()
                fb.release()

    def release_framebuffers(self):
        while self._fbos:
            self._drop_fbo(next(iter(self._fbos)))

    # ---- drawing
    def render(self, scene, camera, width, height, style=None):
        """Returns (rgb top-down uint8 (h, w, 3), mvp as a row-major standard projection matrix for client overlays)."""
        style = style or {}
        held = scene.id in self._gpu
        if not held:
            self.acquire(scene)
        try:
            entry = self._gpu[scene.id]
            self._sync(scene, entry)
            return self._draw(scene, entry, camera, width, height, style)
        finally:
            if not held:
                self.release(scene)

    def _draw(self, scene, entry, camera, width, height, style):
        ctx = self.ctx
        msaa, flat = self._fbo(width, height)
        target = msaa or flat
        target.use()
        view = camera.view_matrix()
        proj = camera.projection_matrix(width / height)
        proj_flipped = proj.copy()
        proj_flipped[1] *= -1  # render upside down so the readback is already top-down
        mvp = proj_flipped @ view
        nmat = view[:3, :3]
        bg = scene.background
        target.clear(*bg, 1.0)
        mat = lambda m: m.T.astype("f4").tobytes()  # noqa: E731
        wireframe_override = style.get("wireframe")

        def passes(transparent):
            for drawable in entry["drawables"]:
                kind, item, vao, extra = drawable
                alpha = item["alpha"]
                if (alpha < 1.0) != transparent:
                    continue
                if kind == "surface":
                    p = self.p_mesh
                    p["mvp"].write(mat(mvp))
                    p["nmat"].write(nmat.T.astype("f4").tobytes())
                    p["alpha"].value, p["ambient"].value, p["specular"].value = (
                        alpha,
                        item["ambient"],
                        item["specular"],
                    )
                    p["psize"].value = 1.0
                    p["mode"].value = 2 if extra is not None else 0
                    if extra is not None:
                        extra.use(0)
                        p["tex"].value = 0
                    ctx.polygon_offset = 1.0, 1.0
                    vao.render()
                    ctx.polygon_offset = 0.0, 0.0
                elif kind == "points":
                    p = self.p_mesh
                    p["mvp"].write(mat(mvp))
                    p["nmat"].write(nmat.T.astype("f4").tobytes())
                    p["alpha"].value, p["mode"].value = alpha, 1
                    p["psize"].value = max(1.0, item["line_width"] * 2)
                    p["ambient"].value = p["specular"].value = 0.0
                    vao.render(self.mgl.POINTS, vertices=len(item["points"]))
                elif kind == "lines":
                    p = self.p_line
                    p["mvp"].write(mat(mvp))
                    p["viewport"].value = (width, height)
                    p["width"].value, p["bias"].value = max(item["width"], 1.0), 2e-5
                    p["alpha"].value = alpha
                    vao.render(self.mgl.TRIANGLE_STRIP, instances=extra)
                elif kind == "spheres":
                    if item["style"] == "point":
                        continue_flat = True
                    p = self.p_sphere
                    p["view"].write(mat(view))
                    p["proj"].write(mat(proj_flipped))
                    p["radius"].value = item["size"] / 2
                    p["viewport_h"].value, p["max_px"].value = (
                        float(height),
                        min(self.max_point, 2048.0),
                    )
                    p["colour"].value, p["alpha"].value, p["ambient"].value = (
                        item["colour"],
                        alpha,
                        0.0,
                    )
                    vao.render(self.mgl.POINTS, vertices=extra)
                elif kind == "arrows":
                    p = self.p_arrow
                    p["mvp"].write(mat(mvp))
                    p["nmat"].write(nmat.T.astype("f4").tobytes())
                    p["scale"].value, p["thick"].value = (
                        item["scale"],
                        1.0 + 0.15 * max(item["width"] - 2, 0),
                    )
                    p["colour"].value, p["alpha"].value = item["colour"], alpha
                    vao.render(instances=extra)

        passes(False)
        target.depth_mask = False
        passes(True)
        target.depth_mask = True
        if msaa:
            ctx.copy_framebuffer(flat, msaa)
        rgb = np.frombuffer(flat.read(components=3), np.uint8).reshape(height, width, 3)
        std = proj @ view  # standard orientation, for the browser's labels and picking
        return rgb, std

    # ---- picking: ray against the scene's triangles, no extra pass
    def pick(self, scene, camera, width, height, x, y):
        std = camera.projection_matrix(width / height) @ camera.view_matrix()
        inv = np.linalg.inv(std)
        ndc = np.array([2 * x / width - 1, 1 - 2 * y / height])
        near, far = (inv @ np.array([*ndc, z, 1.0]) for z in (-1.0, 1.0))
        near, far = near[:3] / near[3], far[:3] / far[3]
        d = (far - near) / np.linalg.norm(far - near)
        best = None
        for index, item in enumerate(scene.items):
            if (
                item["kind"] != "mesh"
                or not len(item["trilist"])
                or item["mode"] == "points"
            ):
                continue
            p, t = item["points"].astype(float), item["trilist"]
            a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
            e1, e2 = b - a, c - a
            h = np.cross(d, e2)
            det = (e1 * h).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = 1.0 / det
                s = near - a
                u = f * (s * h).sum(1)
                q = np.cross(s, e1)
                v = f * (d * q).sum(1)
                dist = f * (e2 * q).sum(1)
            hit = (
                (np.abs(det) > 1e-14) & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 0)
            )
            if not hit.any():
                continue
            i = int(np.flatnonzero(hit)[np.argmin(dist[hit])])
            if best is None or dist[i] < best["distance"]:
                bary = np.array([1 - u[i] - v[i], u[i], v[i]])
                point = near + d * dist[i]
                best = dict(
                    item=index,
                    tri=i,
                    vertex=int(t[i][np.argmax(bary)]),
                    point=point.tolist(),
                    distance=float(dist[i]),
                )
        return best


def _vertex_colours(item):
    """What a wireframe is drawn with: per-vertex colours, the texture sampled at each vertex, or the flat colour."""
    if item["colours"] is not None:
        return item["colours"]
    if item["texture"] is not None and item["tcoords"] is not None:
        tex, uv = item["texture"], item["tcoords"]
        h, w, _ = tex.shape
        x = np.clip(np.rint(uv[:, 0] * (w - 1)), 0, w - 1).astype(int)
        y = np.clip(np.rint((1 - uv[:, 1]) * (h - 1)), 0, h - 1).astype(int)
        return (tex[y, x] / 255.0).astype("f4")
    return item["colour"]


def _vertex_normals(p, t):
    p, t = p.astype(float), t.astype(int)
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    n = np.zeros_like(p)
    for i in range(3):
        np.add.at(n, t[:, i], fn)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(
        "f4"
    )


def _auto_marker_size(points):
    """Mayavi's automatic scale factor for a point set (what menpo's viewers fall back on)."""
    lo, hi = points.min(0), points.max(0)
    d = np.sqrt(((hi - lo) ** 2).sum() / (4 * len(points) ** 0.33))
    return 1.0 if d == 0 else 0.1 * float(d)
