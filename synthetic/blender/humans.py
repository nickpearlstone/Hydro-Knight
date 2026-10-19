"""Procedural swimmers, vinyl tubes and deck clutter for the wave-pool renders (no downloaded assets).

Bodies grow from a stick skeleton with Blender's Skin modifier + subdivision, so they get
smooth shoulders, a neck, elbows, hands, hips and knees. Heads get a nose, ears, eyes and
brows; hair is a displaced cap (short, long or bald); swimwear is painted onto body faces
by height (trunks, one-piece, bikini). Skin is wet: subsurface + a glossy clear coat.
Imported by wavepool_v2.py inside Blender.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

# Fitzpatrick-like skin range (sRGB), wet-dark hair, swimwear colours
SKIN = [
    (0.96, 0.80, 0.69),
    (0.91, 0.72, 0.58),
    (0.84, 0.62, 0.47),
    (0.72, 0.51, 0.36),
    (0.58, 0.39, 0.26),
    (0.44, 0.28, 0.18),
    (0.32, 0.20, 0.13),
    (0.23, 0.14, 0.09),
]
HAIR = [
    (0.02, 0.015, 0.012),
    (0.05, 0.035, 0.025),
    (0.12, 0.07, 0.04),
    (0.30, 0.19, 0.10),
    (0.55, 0.42, 0.25),
]
SUIT = [
    (0.75, 0.08, 0.08),
    (0.08, 0.15, 0.55),
    (0.03, 0.03, 0.03),
    (0.92, 0.45, 0.08),
    (0.15, 0.50, 0.20),
    (0.85, 0.20, 0.55),
    (0.92, 0.92, 0.92),
    (0.08, 0.55, 0.65),
    (0.95, 0.85, 0.15),
    (0.45, 0.15, 0.55),
]
# wave-park tubes: mostly teal/aqua translucent vinyl, some blue (matched to the reference frame)
TUBE = [
    (0.04, 0.62, 0.64),
    (0.06, 0.66, 0.62),
    (0.03, 0.55, 0.62),
    (0.10, 0.45, 0.78),
    (0.05, 0.60, 0.70),
]


def lin(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def mat(
    name,
    color,
    rough=0.5,
    transmission=0.0,
    sss=0.0,
    ior=1.45,
    coat=0.0,
    coat_rough=0.1,
    metal=0.0,
):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*lin(color), 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Transmission Weight"].default_value = transmission
    p.inputs["Subsurface Weight"].default_value = sss
    p.inputs["IOR"].default_value = ior
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = coat_rough
    p.inputs["Metallic"].default_value = metal
    return m


def skin_mat(name, color):
    m = mat(name, color, rough=0.42, sss=0.12, coat=0.6, coat_rough=0.08)
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Subsurface Radius"].default_value = (0.9, 0.35, 0.2)
    p.inputs["Subsurface Scale"].default_value = 0.02
    return m


# ---------------------------------------------------------------- skeleton + poses

BASE = {  # standing adult, metres; x = side, y = forward, z = up
    "pelvis": (0, 0, 0.95),
    "spine": (0, 0, 1.13),
    "chest": (0, 0, 1.31),
    "neck0": (0, 0.0, 1.44),
    "neck1": (0, 0.01, 1.52),
    "shL": (-0.18, 0, 1.43),
    "shR": (0.18, 0, 1.43),
    "hipL": (-0.09, 0, 0.90),
    "hipR": (0.09, 0, 0.90),
}
EDGES = [
    ("pelvis", "spine"),
    ("spine", "chest"),
    ("chest", "neck0"),
    ("neck0", "neck1"),
    ("chest", "shL"),
    ("shL", "elL"),
    ("elL", "wrL"),
    ("wrL", "haL"),
    ("chest", "shR"),
    ("shR", "elR"),
    ("elR", "wrR"),
    ("wrR", "haR"),
    ("pelvis", "hipL"),
    ("hipL", "knL"),
    ("knL", "anL"),
    ("anL", "toL"),
    ("pelvis", "hipR"),
    ("hipR", "knR"),
    ("knR", "anR"),
    ("anR", "toR"),
]
RADII = {
    "pelvis": (0.16, 0.11),
    "spine": (0.135, 0.095),
    "chest": (0.165, 0.105),
    "neck0": (0.058, 0.058),
    "neck1": (0.052, 0.052),
    "shL": (0.062, 0.062),
    "shR": (0.062, 0.062),
    "elL": (0.042, 0.042),
    "elR": (0.042, 0.042),
    "wrL": (0.03, 0.026),
    "wrR": (0.03, 0.026),
    "haL": (0.042, 0.016),
    "haR": (0.042, 0.016),
    "hipL": (0.085, 0.085),
    "hipR": (0.085, 0.085),
    "knL": (0.055, 0.055),
    "knR": (0.055, 0.055),
    "anL": (0.038, 0.038),
    "anR": (0.038, 0.038),
    "toL": (0.035, 0.02),
    "toR": (0.035, 0.02),
}


def limb(side, sh, el, wr, ha):
    """Arm joints from offsets (x mirrored by side)."""
    s = -1 if side == "L" else 1
    out = {}
    for k, v in (("el", el), ("wr", wr), ("ha", ha)):
        out[k + side] = (s * v[0], v[1], v[2])
    return out


ARM_POSES = {
    # (elbow, wrist, hand) for the right arm; mirrored for the left
    "tread": (
        (0.34, 0.12, 1.20),
        (0.46, 0.25, 1.06),
        (0.50, 0.30, 1.02),
    ),  # sculling under the surface
    "up": ((0.28, 0.02, 1.68), (0.32, 0.04, 1.92), (0.33, 0.05, 2.00)),
    "down": ((0.23, 0.0, 1.16), (0.26, 0.05, 0.92), (0.27, 0.07, 0.84)),
    "tube": ((0.30, 0.26, 1.30), (0.24, 0.47, 1.33), (0.20, 0.55, 1.33)),
    "rim": (
        (0.36, -0.04, 1.30),
        (0.56, 0.02, 1.20),
        (0.63, 0.04, 1.17),
    ),  # arms resting out on a tube rim
    "press": (
        (0.32, 0.12, 1.24),
        (0.40, 0.22, 1.12),
        (0.43, 0.26, 1.06),
    ),  # distress: pressing down
    "forward": (
        (0.16, 0.30, 1.62),
        (0.14, 0.40, 1.88),
        (0.13, 0.42, 1.97),
    ),  # crawl stroke reach
}
LEG_POSES = {
    "stand": ((0.10, 0.0, 0.50), (0.10, -0.02, 0.09), (0.10, 0.12, 0.03)),
    "tread": ((0.13, 0.16, 0.55), (0.13, -0.06, 0.18), (0.13, 0.06, 0.10)),
    "kick": ((0.09, 0.03, 0.50), (0.09, -0.05, 0.10), (0.09, -0.10, 0.02)),
    "sit": ((0.11, 0.42, 0.88), (0.11, 0.48, 0.45), (0.11, 0.60, 0.42)),
}


def skeleton(rng, arms_r, arms_l, legs, female):
    j = dict(BASE)
    jit = lambda v, a=0.04: tuple(c + rng.uniform(-a, a) for c in v)  # noqa: E731
    for side, name in (("R", arms_r), ("L", arms_l)):
        el, wr, ha = ARM_POSES[name]
        j.update(limb(side, None, jit(el), jit(wr), jit(ha)))
    kn, an, to = LEG_POSES[legs]
    for side in ("R", "L"):
        s = -1 if side == "L" else 1
        for k, v in (("kn", kn), ("an", an), ("to", to)):
            v = jit(v, 0.03)
            j[k + side] = (s * abs(v[0]), v[1], v[2])
    # real swimmers read as bulkier than a skeleton: thicker trunk and limbs, varied build
    build = rng.uniform(1.15, 1.45)
    radii = {k: (a * build, b * build) for k, (a, b) in RADII.items()}
    if female:
        radii["chest"] = (0.15 * build, 0.115 * build)
        radii["pelvis"] = (0.165 * build, 0.11 * build)
        radii["shL"] = radii["shR"] = (0.055 * build, 0.055 * build)
    return j, radii


def body_mesh(name, joints, radii):
    names = list(joints)
    me = bpy.data.meshes.new(name)
    me.from_pydata(
        [joints[n] for n in names],
        [(names.index(a), names.index(b)) for a, b in EDGES],
        [],
    )
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    sk = ob.modifiers.new("skin", "SKIN")
    sk.use_smooth_shade = True
    sk.branch_smoothing = 0.6
    for i, n in enumerate(names):
        sv = me.skin_vertices[0].data[i]
        sv.radius = radii[n]
        sv.use_root = n == "pelvis"
    sub = ob.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    dg = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    out = bpy.data.objects.new(name, new)
    bpy.context.collection.objects.link(out)
    for p in new.polygons:
        p.use_smooth = True
    return out


def paint_suit(ob, style):
    """Material index 1 (suit) on faces by height/side in the standing frame."""
    me = ob.data
    for p in me.polygons:
        x, y, z = p.center
        on = False
        if style == "trunks":
            on = 0.70 <= z <= 1.00 and abs(x) < 0.28
        elif style == "onepiece":
            on = 0.84 <= z <= 1.40 and abs(x) < 0.20
        elif style == "bikini":
            on = (0.84 <= z <= 0.97 and abs(x) < 0.2) or (
                1.24 <= z <= 1.36 and abs(x) < 0.17 and y > -0.02
            )
        elif style == "rashguard":
            on = 0.70 <= z <= 1.44 and abs(x) < 0.30
        p.material_index = 1 if on else 0


def add_sphere(bm_target, center, scale, rot=(0, 0, 0), mat_index=0, seg=24, rings=14):
    m = bmesh.ops.create_uvsphere(
        bm_target, u_segments=seg, v_segments=rings, radius=1.0
    )
    verts = m["verts"]
    from mathutils import Euler, Matrix

    T = (
        Matrix.Translation(center)
        @ Euler(rot).to_matrix().to_4x4()
        @ Matrix.Diagonal((*scale, 1))
    )
    bmesh.ops.transform(bm_target, matrix=T, verts=verts)
    faces = {f for v in verts for f in v.link_faces}
    for f in faces:
        f.material_index = mat_index
        f.smooth = True


def head_mesh(name, r, hair_style, rng):
    """Head centred at the origin, facing +y: skull, jaw, nose, ears, eyes, brows, hair."""
    bm = bmesh.new()
    add_sphere(
        bm, (0, 0, 0), (r * 0.86, r * 0.98, r * 1.08), mat_index=0, seg=32, rings=20
    )  # skull
    add_sphere(
        bm, (0, r * 0.25, -r * 0.48), (r * 0.62, r * 0.62, r * 0.55), mat_index=0
    )  # jaw
    add_sphere(
        bm,
        (0, r * 0.95, -r * 0.08),
        (r * 0.12, r * 0.16, r * 0.24),
        mat_index=0,
        seg=12,
        rings=8,
    )  # nose
    for s in (-1, 1):
        add_sphere(
            bm,
            (s * r * 0.86, -r * 0.05, -r * 0.05),
            (r * 0.08, r * 0.2, r * 0.28),
            mat_index=0,
            seg=12,
            rings=8,
        )
        add_sphere(
            bm,
            (s * r * 0.34, r * 0.84, r * 0.12),
            (r * 0.12, r * 0.06, r * 0.08),
            mat_index=3,
            seg=10,
            rings=6,
        )  # eye
        add_sphere(
            bm,
            (s * r * 0.36, r * 0.86, r * 0.30),
            (r * 0.2, r * 0.05, r * 0.045),
            mat_index=2,
            seg=10,
            rings=6,
        )  # brow
    add_sphere(
        bm,
        (0, r * 0.88, -r * 0.42),
        (r * 0.22, r * 0.05, r * 0.04),
        mat_index=4,
        seg=10,
        rings=6,
    )  # mouth
    if hair_style != "bald":
        add_sphere(
            bm,
            (0, -r * 0.08, r * 0.16),
            (r * 0.93, r * 1.04, r * 1.0),
            mat_index=2,
            seg=32,
            rings=20,
        )
        if hair_style == "long":
            add_sphere(
                bm, (0, -r * 0.55, -r * 0.75), (r * 0.8, r * 0.45, r * 1.2), mat_index=2
            )
        if hair_style == "bun":
            add_sphere(
                bm, (0, -r * 0.8, r * 0.6), (r * 0.42, r * 0.42, r * 0.42), mat_index=2
            )
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def build_person(rng, pid, pose, role="swimmer"):
    """Returns (object, head radius, head offset from object origin in the object's frame).

    The object origin is the head centre, so placing the object places the head exactly.
    """
    child = role == "swimmer" and rng.random() < 0.25
    female = rng.random() < 0.5
    s = rng.uniform(0.66, 0.85) if child else rng.uniform(0.95, 1.12)
    arms = {
        "vertical": ("tread", "tread"),
        "arms_up": ("up", "up"),
        "low": ("press", "press"),
        "tube": ("rim", "rim"),
        "swim": ("forward", "down"),
        "stand": ("down", "down"),
        "shallow": ("down", "tread"),
        "seated": ("down", "down"),
    }[pose]
    if pose == "vertical" and rng.random() < 0.3:
        arms = ("tread", "down")
    legs = {
        "vertical": "tread",
        "arms_up": "tread",
        "low": "tread",
        "tube": "sit",
        "swim": "kick",
        "stand": "stand",
        "shallow": "stand",
        "seated": "sit",
    }[pose]
    joints, radii = skeleton(rng, arms[0], arms[1], legs, female)
    body = body_mesh(f"body_{pid}", joints, radii)
    style = ("onepiece" if rng.random() < 0.5 else "bikini") if female else "trunks"
    if child and rng.random() < 0.3:
        style = "rashguard"
    paint_suit(body, style)
    r = 0.108 * (s**0.5) / s  # in skeleton units: kids have relatively bigger heads
    hair_style = rng.choice(
        ["short", "short", "long", "bun"]
        if female
        else ["short", "short", "short", "bald"]
    )
    head = head_mesh(f"head_{pid}", r, hair_style, rng)
    head_c = Vector((0, 0.03, 1.52 + r * 1.05))
    tilt = (
        rng.uniform(-0.25, 0.15) if pose != "low" else rng.uniform(-0.7, -0.35)
    )  # low: head back
    head.rotation_euler = (tilt, 0, rng.uniform(-0.4, 0.4))
    head.location = head_c
    skin = SKIN[rng.integers(len(SKIN))]
    mats = [
        skin_mat(f"skin{pid}", skin),
        mat(f"suit{pid}", SUIT[rng.integers(len(SUIT))], rough=0.45, coat=0.4),
        mat(
            f"hair{pid}",
            HAIR[rng.integers(len(HAIR))],
            rough=0.28,
            coat=0.7,
            coat_rough=0.05,
        ),
        mat(f"eye{pid}", (0.05, 0.04, 0.035), rough=0.15),
        mat(f"lip{pid}", tuple(c * 0.75 for c in skin), rough=0.4),
    ]
    if role == "guard":
        mats[1] = mat(f"suit{pid}", (0.85, 0.05, 0.05), rough=0.45)
    for o in (body, head):
        for m in mats:
            o.data.materials.append(m)
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    ob = bpy.context.object
    # put the origin at the head centre, scale to the person's size
    me = ob.data
    me.transform(__import__("mathutils").Matrix.Translation(-head_c))
    ob.location = (0, 0, 0)
    ob.scale = (s, s, s)
    ob.name = f"person_{pid}"
    ob["category_id"] = pid
    return ob, r * s


# ---------------------------------------------------------------- tubes + clutter


TUBE_MATS = {}  # vinyl + handle materials shared by all tubes in a scene; cleared per scene


def build_tube(rng, loc, mats_cache=None, yaw=None):
    """Thick translucent vinyl ring (air inside) with two dark-blue grab handles, ~1.1-1.3 m across."""
    if mats_cache is None:
        mats_cache = TUBE_MATS
    key = int(rng.integers(len(TUBE)))
    if key not in mats_cache or mats_cache[key].name not in bpy.data.materials:
        m = mat(
            f"vinyl{key}",
            tuple(0.35 + 0.65 * c for c in TUBE[key]),  # light aqua vinyl
            rough=0.05,
            transmission=0.9,
            ior=1.4,
            coat=1.0,
            coat_rough=0.02,
        )
        p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Alpha"].default_value = 0.55
        p.inputs["Subsurface Weight"].default_value = 0.2
        mats_cache[key] = m
    if (
        "handle" not in mats_cache
        or mats_cache["handle"].name not in bpy.data.materials
    ):
        mats_cache["handle"] = mat("handle", (0.08, 0.18, 0.65), rough=0.4)
    R, rr = rng.uniform(0.40, 0.48), rng.uniform(0.15, 0.19)
    yaw = rng.uniform(0, 6.3) if yaw is None else yaw
    bpy.ops.mesh.primitive_torus_add(
        major_radius=R,
        minor_radius=rr,
        major_segments=48,
        minor_segments=18,
        location=loc,
        rotation=(rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), yaw),
    )
    t = bpy.context.object
    bpy.ops.object.shade_smooth()
    t.data.materials.append(mats_cache[key])
    t["category_id"] = 0
    for side in (-1, 1):
        hx, hy = side * R * math.cos(yaw), side * R * math.sin(yaw)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(loc[0] + hx, loc[1] + hy, loc[2] + rr * 0.95),
            rotation=(0, 0, yaw + math.pi / 2),
        )
        h = bpy.context.object
        h.scale = (0.16, 0.05, 0.03)
        h.data.materials.append(mats_cache["handle"])
        h["category_id"] = 0
    return t


def _box(loc, scale, m, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.scale = scale
    o.data.materials.append(m)
    return o


def _cyl(loc, r, h, m, rot=(0, 0, 0), verts=16):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=verts, radius=r, depth=h, location=loc, rotation=rot
    )
    o = bpy.context.object
    bpy.ops.object.shade_smooth()
    o.data.materials.append(m)
    return o


def build_clutter(rng, pool_w, pool_l, waves, next_pid, rope_y=None):
    """Deck and pool furniture that real false boxes come from. Returns labelled deck people."""
    white = mat("white_paint", (0.92, 0.92, 0.9), rough=0.5)
    steel = mat("steel", (0.8, 0.8, 0.82), rough=0.25, metal=1.0)
    people = []
    # rope line with floats across the pool
    yr = rng.uniform(6, 14) if rope_y is None else rope_y
    rope = mat("rope", (0.9, 0.85, 0.3), rough=0.7)
    _cyl(
        (0, yr, float(waves(0, yr)) + 0.02),
        0.008,
        pool_w,
        rope,
        rot=(0, math.pi / 2, 0),
        verts=8,
    )
    float_cols = [
        mat("f_white", (0.95, 0.95, 0.95), 0.4),
        mat("f_blue", (0.1, 0.3, 0.8), 0.4),
        mat("f_red", (0.85, 0.1, 0.1), 0.4),
    ]
    for i, x in enumerate(np.arange(-pool_w / 2 + 0.5, pool_w / 2, 0.7)):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.07,
            location=(x, yr, float(waves(x, yr)) + 0.03),
            segments=16,
            ring_count=10,
        )
        bpy.ops.object.shade_smooth()
        bpy.context.object.data.materials.append(float_cols[i % 3])
    # ladders + railings on both sides
    for side in (-1, 1):
        for y in rng.uniform(3, pool_l - 5, 2):
            x = side * (pool_w / 2 - 0.1)
            for dx in (-0.3, 0.3):
                _cyl((x, y + dx, 0.5), 0.025, 1.0, steel)
                _cyl(
                    (x - side * 0.2, y + dx, 1.05),
                    0.025,
                    0.45,
                    steel,
                    rot=(0, math.pi / 2, 0),
                )
    # lifeguard stand on a side deck with a guard on it
    side = rng.choice([-1, 1])
    gx, gy = side * (pool_w / 2 + 1.2), rng.uniform(8, 22)
    for dx in (-0.35, 0.35):
        for dy in (-0.35, 0.35):
            _cyl((gx + dx, gy + dy, 1.2), 0.04, 2.0, white)
    _box((gx, gy, 2.25), (0.8, 0.8, 0.08), white)
    _box((gx + side * 0.38, gy, 2.7), (0.06, 0.8, 0.9), white)
    _cyl((gx, gy, 3.6), 0.03, 1.4, white)
    bpy.ops.mesh.primitive_cone_add(
        vertices=16, radius1=1.0, depth=0.35, location=(gx, gy, 4.25)
    )
    bpy.context.object.data.materials.append(mat("umb_red", (0.85, 0.12, 0.1), 0.6))
    g, r = build_person(rng, next_pid, "seated", role="guard")
    g.location = (gx - side * 0.05, gy, 2.29 + (1.635 - 0.90) * g.scale[2])
    g.rotation_euler = (0, 0, math.pi / 2 * -side)
    people.append((g, r, "guard"))
    next_pid += 1
    # umbrellas, chairs and people on the decks
    umb_cols = [
        mat(f"umb{i}", c, 0.6)
        for i, c in enumerate(
            [(0.1, 0.4, 0.8), (0.95, 0.95, 0.95), (0.9, 0.6, 0.1), (0.2, 0.6, 0.3)]
        )
    ]
    for _ in range(int(rng.integers(6, 12))):
        side = rng.choice([-1, 1])
        x = side * rng.uniform(pool_w / 2 + 2, pool_w / 2 + 9)
        y = rng.uniform(0, pool_l + 4)
        _cyl((x, y, 1.4), 0.025, 2.2, white)
        bpy.ops.mesh.primitive_cone_add(
            vertices=12, radius1=rng.uniform(1.0, 1.5), depth=0.4, location=(x, y, 2.6)
        )
        bpy.context.object.data.materials.append(umb_cols[rng.integers(len(umb_cols))])
    for _ in range(int(rng.integers(8, 18))):
        side = rng.choice([-1, 1])
        if rng.random() < 0.25:  # far end deck
            x, y = rng.uniform(-pool_w / 2, pool_w / 2), pool_l + rng.uniform(1, 4)
        else:
            x, y = (
                side * rng.uniform(pool_w / 2 + 0.8, pool_w / 2 + 8),
                rng.uniform(0, pool_l),
            )
        p, r = build_person(rng, next_pid, "stand", role="deck")
        p.location = (x, y, 0.4 + 1.62 * p.scale[2] - 0.04)
        p.rotation_euler = (0, 0, rng.uniform(0, 6.3))
        people.append((p, r, "deck"))
        next_pid += 1
    # huts with blue roofs past the far deck
    roof = mat("roof", (0.15, 0.35, 0.7), 0.6)
    for _ in range(int(rng.integers(3, 6))):
        x, y = rng.uniform(-pool_w, pool_w), pool_l + rng.uniform(8, 14)
        w = rng.uniform(4, 8)
        for dx in (-w / 2, w / 2):
            for dy in (-1.5, 1.5):
                _cyl((x + dx, y + dy, 1.6), 0.08, 2.8, white, verts=8)
        bpy.ops.mesh.primitive_cone_add(
            vertices=4,
            radius1=w * 0.75,
            depth=1.2,
            location=(x, y, 3.6),
            rotation=(0, 0, math.pi / 4),
        )
        bpy.context.object.scale = (1, 0.45, 1)
        bpy.context.object.data.materials.append(roof)
    return people, next_pid


# ---------------------------------------------------------------- splash


def foam_material(name="foam"):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 9.0
    nz.inputs["Detail"].default_value = 10
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.48
    ramp.color_ramp.elements[1].position = 0.62
    white = nt.nodes.new("ShaderNodeBsdfPrincipled")
    white.inputs["Base Color"].default_value = (0.92, 0.95, 0.95, 1)
    white.inputs["Roughness"].default_value = 0.6
    white.inputs["Subsurface Weight"].default_value = 0.3
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    L = nt.links.new
    L(tc.outputs["Object"], nz.inputs["Vector"])
    L(nz.outputs["Fac"], ramp.inputs["Fac"])
    L(ramp.outputs["Color"], mix.inputs["Fac"])
    L(tr.outputs["BSDF"], mix.inputs[1])
    L(white.outputs["BSDF"], mix.inputs[2])
    L(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def spray_material(name="spray"):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = mat(name, (0.97, 0.99, 0.99), rough=0.35, transmission=0.2, ior=1.33)
    return m


def mist_material(name="mist"):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    vs = nt.nodes.new("ShaderNodeVolumeScatter")
    vs.inputs["Color"].default_value = (0.95, 0.98, 0.98, 1)
    # density fades to zero at the edge (spherical gradient) and is broken up by noise
    tc = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.inputs[1].default_value = 2.5
    L = nt.links.new
    L(tc.outputs["Object"], grad.inputs["Vector"])
    L(tc.outputs["Object"], nz.inputs["Vector"])
    L(grad.outputs["Fac"], m1.inputs[0])
    L(nz.outputs["Fac"], m1.inputs[1])
    L(m1.outputs[0], m2.inputs[0])
    L(m2.outputs[0], vs.inputs["Density"])
    L(vs.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    return m


def build_splash(rng, x, y, wz, strength):
    """Patchy foam ring on the surface + a cloud of spray droplets above it. strength 0..1."""
    bm = bmesh.new()
    n_patch = 1 + int(strength * 2)
    for _ in range(n_patch):
        cx, cy = x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3)
        rad = rng.uniform(0.25, 0.45) + 0.4 * strength
        add_sphere(
            bm,
            (cx, cy, wz + 0.015),
            (rad, rad * rng.uniform(0.6, 1.0), 0.025),
            mat_index=0,
            seg=24,
            rings=8,
        )
    fm = bpy.data.meshes.new("foam")
    bm.to_mesh(fm)
    bm.free()
    fo = bpy.data.objects.new("foam", fm)
    bpy.context.collection.objects.link(fo)
    fm.materials.append(foam_material())
    fo["category_id"] = 0
    # spray: many fine droplets kept low over the water, plus a small white mist cloud
    n = int(80 + 320 * strength)
    bm = bmesh.new()
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        rr = abs(rng.normal(0, 0.12 + 0.2 * strength))
        h = abs(rng.normal(0, 0.05 + 0.18 * strength))
        s = rng.uniform(0.003, 0.007 + 0.006 * strength)
        add_sphere(
            bm,
            (x + rr * math.cos(ang), y + rr * math.sin(ang), wz + h),
            (s, s, s * rng.uniform(1, 3)),
            mat_index=0,
            seg=6,
            rings=4,
        )
    dm = bpy.data.meshes.new("spray")
    bm.to_mesh(dm)
    bm.free()
    do = bpy.data.objects.new("spray", dm)
    bpy.context.collection.objects.link(do)
    dm.materials.append(spray_material())
    do["category_id"] = 0
    mist = mist_material()
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1, segments=16, ring_count=8, location=(x, y, wz + 0.08 + 0.1 * strength)
    )
    mo = bpy.context.object
    mo.scale = (0.25 + 0.3 * strength, 0.25 + 0.3 * strength, 0.08 + 0.18 * strength)
    mo.data.materials.append(mist)
    mo["category_id"] = 0
    return fo, do
