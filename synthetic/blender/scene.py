"""Wave-pool scene v3: one config file in, rendered frames + exact labels out.

v3 is matched to a real reference frame (camera pose, crowd density, tubes, water colour) from
a JSON config, so one render can be compared with that frame by the detectors
(synthetic/tools/score.py). Builds on the 2026-10-17 rendering spike.

    synthetic/blender-bin/Blender.app/Contents/MacOS/Blender -b -P synthetic/blender/scene.py -- \\
        --config synthetic/configs/ref_6f5c.json --out synthetic/renders/b01 [--samples 32] [--res 1920]

Writes per frame: rgb_<n>.png (linear-ish AgX render), depth_<n>.npy (metres, for haze),
seg_<n>.npy (person id per pixel, 0 = none), labels_<n>.json (per person: projected head
centre u/v, head radius px, fraction of head under water, pose, visible pixels, role).
"""

import argparse
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import humans  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument("--res", type=int, default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument(
    "--clip",
    type=int,
    default=1,
    help="consecutive 60 fps frames to render (moving water)",
)
ap.add_argument(
    "--keep",
    type=int,
    default=None,
    help="index of the labelled frame in the clip (default: middle)",
)
ap.add_argument(
    "--frames",
    type=int,
    default=1,
    help="scenes to build (each = 1 frame from the config camera)",
)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

with open(args.config) as fh:
    CFG = json.load(fh)
SEED = CFG.get("seed", 0) if args.seed is None else args.seed
SAMPLES = args.samples or CFG.get("samples", 32)
RES = args.res or CFG.get("res", 1920)
POOL_W, POOL_L, DEPTH = (
    CFG["pool"]["width"],
    CFG["pool"]["length"],
    CFG["pool"].get("depth", 1.4),
)
# wave pools are fans: narrow at the wave-machine end (camera, y = 0), wide at the beach (y = POOL_L)
POOL_W0 = CFG["pool"].get("width_near", POOL_W)


def half_w(y):
    return (POOL_W0 + (POOL_W - POOL_W0) * y / POOL_L) / 2


class Waves:
    """Water height z(x, y): a long swell along the pool plus a short cross chop."""

    def __init__(self, rng, c):
        self.a1 = rng.uniform(*c["swell_amp"])
        self.k1 = 2 * math.pi / rng.uniform(*c["swell_len"])
        self.p1 = rng.uniform(0, 6.3)
        self.a2 = rng.uniform(*c["chop_amp"])
        self.k2 = 2 * math.pi / rng.uniform(2.0, 4.0)
        self.p2 = rng.uniform(0, 6.3)
        self.dir2 = rng.uniform(-0.8, 0.8)

    def __call__(self, x, y, t=0.0):
        """t = seconds from the labelled frame; each wave moves at its deep-water speed."""
        w1, w2 = math.sqrt(9.81 * self.k1), math.sqrt(9.81 * self.k2)
        return self.a1 * np.sin(self.k1 * y + self.p1 - w1 * t) + self.a2 * np.sin(
            self.k2 * (self.dir2 * x + y) + self.p2 - w2 * t
        )


def link(nt, a, b):
    nt.links.new(a, b)


ANIM = {}  # per-scene handles for clip frames: water grid, ripple nodes, floating objects


def build_water(rng, waves, c):
    step = 0.25
    xs = np.arange(-POOL_W / 2, POOL_W / 2 + 1e-6, step)
    ys = np.arange(0, POOL_L + 1e-6, step)
    gx, gy = np.meshgrid(xs, ys)
    gz = waves(gx, gy)
    ANIM["grid"] = (gx.ravel(), gy.ravel())
    bm = bmesh.new()
    nx, ny = len(xs), len(ys)
    top = [
        [bm.verts.new((gx[j, i], gy[j, i], gz[j, i])) for i in range(nx)]
        for j in range(ny)
    ]
    for j in range(ny - 1):
        for i in range(nx - 1):
            bm.faces.new((top[j][i], top[j][i + 1], top[j + 1][i + 1], top[j + 1][i]))
    ring = (
        [top[0][i] for i in range(nx)]
        + [top[j][nx - 1] for j in range(1, ny)]
        + [top[ny - 1][i] for i in range(nx - 2, -1, -1)]
        + [top[j][0] for j in range(ny - 2, 0, -1)]
    )
    low = [bm.verts.new((v.co.x, v.co.y, -DEPTH)) for v in ring]
    for k in range(len(ring)):
        bm.faces.new(
            (ring[k], ring[(k + 1) % len(ring)], low[(k + 1) % len(ring)], low[k])
        )
    bm.faces.new(low[::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("water")
    bm.to_mesh(me)
    bm.free()
    for f in me.polygons:
        f.use_smooth = True
    ob = bpy.data.objects.new("water", me)
    bpy.context.collection.objects.link(ob)

    m = bpy.data.materials.new("water")
    m.use_nodes = True
    nt = m.node_tree
    nodes = nt.nodes
    p = nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (1, 1, 1, 1)
    p.inputs["Roughness"].default_value = rng.uniform(*c["roughness"])
    p.inputs["Transmission Weight"].default_value = 1.0
    p.inputs["IOR"].default_value = 1.333
    # two scales of ripple: metre-scale chop + fine wind ripples (glints)
    tc = nodes.new("ShaderNodeTexCoord")
    n1 = nodes.new("ShaderNodeTexNoise")
    n1.noise_dimensions = "4D"  # W = time, so ripples evolve between clip frames
    n1.inputs["Scale"].default_value = rng.uniform(0.6, 1.4)
    n1.inputs["Detail"].default_value = 6
    n2 = nodes.new("ShaderNodeTexNoise")
    n2.noise_dimensions = "4D"
    ANIM["ripples"] = (n1, n2)
    n2.inputs["Scale"].default_value = rng.uniform(
        *CFG["water"].get("ripple_scale", [5, 9])
    )
    n2.inputs["Detail"].default_value = 4
    b1 = nodes.new("ShaderNodeBump")
    b1.inputs["Strength"].default_value = c["bump"][0]
    b1.inputs["Distance"].default_value = 0.05
    b2 = nodes.new("ShaderNodeBump")
    b2.inputs["Strength"].default_value = c["bump"][1]
    b2.inputs["Distance"].default_value = 0.02
    link(nt, tc.outputs["Object"], n1.inputs["Vector"])
    link(nt, tc.outputs["Object"], n2.inputs["Vector"])
    link(nt, n1.outputs["Fac"], b1.inputs["Height"])
    link(nt, n2.outputs["Fac"], b2.inputs["Height"])
    link(nt, b1.outputs["Normal"], b2.inputs["Normal"])
    link(nt, b2.outputs["Normal"], p.inputs["Normal"])
    # shadow rays pass through the surface (no caustics), so submerged bodies are lit
    lp = nodes.new("ShaderNodeLightPath")
    tr = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    out = nodes["Material Output"]
    link(nt, lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    link(nt, p.outputs["BSDF"], mix.inputs[1])
    link(nt, tr.outputs["BSDF"], mix.inputs[2])
    link(nt, mix.outputs["Shader"], out.inputs["Surface"])
    absn = nodes.new("ShaderNodeVolumeAbsorption")
    absn.inputs["Color"].default_value = (*c["absorb_color"], 1)
    absn.inputs["Density"].default_value = rng.uniform(*c["absorb_density"])
    sca = nodes.new("ShaderNodeVolumeScatter")
    sca.inputs["Color"].default_value = (*c["scatter_color"], 1)
    sca.inputs["Density"].default_value = rng.uniform(*c["scatter_density"])
    sca.inputs["Anisotropy"].default_value = 0.4
    addv = nodes.new("ShaderNodeAddShader")
    link(nt, absn.outputs[0], addv.inputs[0])
    link(nt, sca.outputs[0], addv.inputs[1])
    link(nt, addv.outputs[0], out.inputs["Volume"])
    ob.data.materials.append(m)
    return ob


def build_pool(rng, c):
    fm = humans.mat("floor", tuple(c["floor_color"]), rough=0.85)
    # floor: soft blotchy pattern stands in for caustics and stains
    nt = fm.node_tree
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.8
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (
        *humans.lin([v * 0.9 for v in c["floor_color"]]),
        1,
    )
    ramp.color_ramp.elements[1].color = (*humans.lin(c["floor_color"]), 1)
    link(nt, nz.outputs["Fac"], ramp.inputs["Fac"])
    # fake caustics: thin bright network from Voronoi cell edges, warped by noise, added to the floor colour
    tcf = nt.nodes.new("ShaderNodeTexCoord")
    warp = nt.nodes.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 0.7
    warp.inputs["Detail"].default_value = 4
    mixv = nt.nodes.new("ShaderNodeVectorMath")
    mixv.operation = "MULTIPLY_ADD"
    mixv.inputs[1].default_value = (
        0.8,
        0.8,
        0.8,
    )  # strong warp hides the Voronoi cell grid
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = c.get("caustic_scale", 2.2)
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.0
    cr.color_ramp.elements[0].color = (1, 1, 1, 1)
    cr.color_ramp.elements[1].position = 0.06
    cr.color_ramp.elements[1].color = (0, 0, 0, 1)
    addc = nt.nodes.new("ShaderNodeMix")
    addc.data_type = "RGBA"
    addc.blend_type = "ADD"
    addc.inputs["Factor"].default_value = c.get("caustic_strength", 0.35)
    link(nt, tcf.outputs["Object"], warp.inputs["Vector"])
    link(nt, warp.outputs["Color"], mixv.inputs[0])
    link(nt, tcf.outputs["Object"], mixv.inputs[2])
    link(nt, mixv.outputs["Vector"], vor.inputs["Vector"])
    link(nt, vor.outputs["Distance"], cr.inputs["Fac"])
    link(nt, ramp.outputs["Color"], addc.inputs[6])
    link(nt, cr.outputs["Color"], addc.inputs[7])
    link(nt, addc.outputs[2], nt.nodes["Principled BSDF"].inputs["Base Color"])
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, POOL_L / 2, -DEPTH))
    fl = bpy.context.object
    fl.scale = (POOL_W, POOL_L, 1)
    fl.data.materials.append(fm)
    wm = humans.mat("wall", (0.80, 0.88, 0.90), rough=0.7)
    ang = math.atan((POOL_W - POOL_W0) / 2 / POOL_L)
    side_len = POOL_L / math.cos(ang) + 0.4
    dm = humans.mat("deck", tuple(c["deck_color"]), rough=0.9)
    nt = dm.node_tree
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.15
    link(nt, nz.outputs["Fac"], bmp.inputs["Height"])
    link(nt, bmp.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    for side in (-1, 1):
        xm = side * (half_w(POOL_L / 2) + 0.1)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(xm, POOL_L / 2, -DEPTH / 2 + 0.22),
            rotation=(0, 0, -side * ang),
        )
        w = bpy.context.object
        w.scale = (0.3, side_len, DEPTH + 0.44)
        w.data.materials.append(wm)
        # side deck: a wide slab whose inner edge sits on the wall top
        xd = side * (
            half_w(POOL_L / 2) + 0.25 + 15 / math.cos(ang)
        )  # inner edge on the wall top
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(xd, POOL_L / 2, 0.3), rotation=(0, 0, -side * ang)
        )
        d = bpy.context.object
        d.scale = (30, side_len + 40, 0.2)
        d.data.materials.append(dm)
    for loc, sc in [
        ((0, POOL_L - 0.05, -DEPTH / 2 + 0.22), (POOL_W + 0.4, 0.3, DEPTH + 0.44)),
        ((0, 0.05, -DEPTH / 2 + 0.22), (POOL_W0 + 0.4, 0.3, DEPTH + 0.44)),
    ]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        w = bpy.context.object
        w.scale = sc
        w.data.materials.append(wm)
    for loc, sc in [
        ((0, -6, 0.3), (POOL_W + 40, 12, 0.2)),
        ((0, POOL_L + 10.1, 0.3), (POOL_W + 80, 20, 0.2)),
    ]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        d = bpy.context.object
        d.scale = sc
        d.data.materials.append(dm)
    # grass beyond the far deck, then trees: clusters of small lumpy spheres read as foliage
    gm = humans.mat("grass", (0.32, 0.45, 0.18), rough=0.95)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, POOL_L + 40, 0.38))
    g = bpy.context.object
    g.scale = (300, 60, 1)
    g.data.materials.append(gm)
    for side in (-1, 1):
        bpy.ops.mesh.primitive_plane_add(
            size=1, location=(side * (POOL_W / 2 + 60), POOL_L / 2, 0.38)
        )
        g = bpy.context.object
        g.scale = (80, 200, 1)
        g.data.materials.append(gm)
    leaf_cols = [
        (0.10, 0.20, 0.07),
        (0.16, 0.28, 0.09),
        (0.08, 0.16, 0.06),
        (0.22, 0.32, 0.12),
    ]
    leaf = [humans.mat(f"leaf{i}", col, rough=0.8) for i, col in enumerate(leaf_cols)]
    tex = bpy.data.textures.new("leafy", "CLOUDS")
    tex.noise_scale = 0.5
    for _ in range(int(c["trees"])):
        side = rng.random()
        if side < 0.6:
            x, y = rng.uniform(-80, 80), POOL_L + rng.uniform(18, 45)
        else:
            x, y = (
                rng.choice([-1, 1]) * rng.uniform(POOL_W / 2 + 22, POOL_W / 2 + 60),
                rng.uniform(-10, POOL_L + 20),
            )
        h = rng.uniform(5, 12)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=8, radius=0.2, depth=h, location=(x, y, h / 2)
        )
        bpy.context.object.data.materials.append(
            humans.mat("trunk", (0.25, 0.2, 0.15), 0.9)
        )
        for _k in range(int(rng.integers(3, 7))):
            r = rng.uniform(1.5, 3.5)
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=3,
                radius=r,
                location=(
                    x + rng.uniform(-2, 2),
                    y + rng.uniform(-2, 2),
                    h + rng.uniform(-2, 1.5),
                ),
            )
            o = bpy.context.object
            d = o.modifiers.new("lumpy", "DISPLACE")
            d.texture = tex
            d.strength = rng.uniform(0.8, 1.6)
            o.data.materials.append(leaf[rng.integers(len(leaf))])


def set_camera(c):
    sc = bpy.context.scene
    cam = sc.camera
    cam.data.lens_unit = "FOV"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle = math.radians(c["hfov"])
    cam.location = Vector(c["pos"])
    cam.rotation_euler = Euler(
        (
            math.radians(90 - c["pitch"]),
            math.radians(c.get("roll", 0)),
            -math.radians(c["yaw"]),
        ),
        "XYZ",
    )
    bpy.context.view_layer.update()


def in_view(sc, cam, co, margin=0.05):
    ndc = world_to_camera_view(sc, cam, Vector(co))
    return (
        ndc.z > 0 and -margin <= ndc.x <= 1 + margin and -margin <= ndc.y <= 1 + margin
    )


def sample_crowd(rng, c, sc, cam):
    """Spots inside the pool and the camera view; density rises with distance (busier far end)."""
    pts, kinds = [], []
    n_target, grad = c["n_people"], c["far_density_gain"]
    tries = 0
    while len(pts) < n_target and tries < n_target * 200:
        tries += 1
        if pts and rng.random() < c.get("cluster", 0.0):  # families and friend groups
            q = pts[int(rng.integers(len(pts)))]
            x, y = q[0] + rng.normal(0, 1.0), q[1] + rng.normal(0, 1.0)
            if not (1.2 < y < POOL_L - 0.8 and abs(x) < half_w(y) - 0.7):
                continue
        else:
            y = rng.uniform(1.2, POOL_L - 0.8)
            x = rng.uniform(-half_w(y) + 0.7, half_w(y) - 0.7)
        if rng.random() > (1 + grad * y / POOL_L) / (1 + grad):
            continue
        if not in_view(sc, cam, (x, y, 0.3)):
            continue
        # tube riders are more common near the camera in the reference; far end = mostly heads
        p_tube = (
            c["tube_frac_near"]
            + (c["tube_frac_far"] - c["tube_frac_near"]) * y / POOL_L
        )
        kind = "lounge" if rng.random() < p_tube else "swimmer"
        need = 1.25 if kind == "lounge" else c["min_spacing"]
        if all(
            math.hypot(x - q[0], y - q[1])
            > max(need, 1.25 if k == "lounge" else c["min_spacing"])
            for q, k in zip(pts, kinds, strict=False)
        ):
            pts.append((x, y))
            kinds.append(kind)
    return pts, kinds


LIB_POSE = {
    "vertical": "tread",
    "low": "tread",
    "arms_up": "armsup",
    "swim": "prone",
    "shallow": "stand",
    "tube": "recline",
}


def load_lib(path):
    """Append the posed MPFB humans (synthetic/blender/make_human_lib.py); {pose: [template objects]}."""
    with bpy.data.libraries.load(os.path.abspath(path)) as (src, dst):
        dst.objects = [n for n in src.objects if n.startswith("hum_")]
    by = {}
    coat = CFG["crowd"].get("skin_coat")
    for o in dst.objects:
        by.setdefault(o["pose"], []).append(o)
        for m in o.data.materials if coat is not None else []:
            for n in m.node_tree.nodes if m and m.use_nodes else []:
                if n.type == "BSDF_PRINCIPLED" and not m.name.startswith("suit_"):
                    n.inputs["Coat Weight"].default_value = coat  # wet-skin gloss
    return by


def lib_person(rng, lib, pose, pid):
    """A linked copy (shared mesh) of a random library human in this pose; origin = head centre."""
    group = lib[LIB_POSE[pose]]
    t = group[int(rng.integers(len(group)))]
    ob = t.copy()
    bpy.context.collection.objects.link(ob)
    ob.hide_render = False
    ob["category_id"] = pid
    return ob, float(t["head_r"])


def place_lounger(rng, pid, x, y, wz, lib=None):
    """Person sitting in a tube: pelvis in the hole, back on the rim, legs over the front."""
    yaw = rng.uniform(0, 2 * math.pi)
    humans.build_tube(rng, (x, y, wz + 0.05), yaw=yaw)
    if lib:
        ob, r = lib_person(rng, lib, "tube", pid)
        rz = Euler((0, 0, yaw), "XYZ")
        ob.rotation_euler = rz
        ob.location = Vector((x, y, wz - 0.1)) - rz.to_matrix() @ Vector(
            ob["pelvis_off"]
        )
        return ob, r
    ob, r_world = humans.build_person(rng, pid, "tube")
    s = ob.scale[0]
    r = r_world / s
    pelvis_local = Vector((0, -0.03, 0.95 - (1.52 + r * 1.05)))
    pitch = rng.uniform(
        0.45, 0.75
    )  # lean back (+x rotation tips the head toward -y, behind the body)
    rot = Euler((pitch, 0, yaw), "XYZ")
    ob.rotation_euler = rot
    target = Vector((x, y, wz - 0.12))
    ob.location = target - rot.to_matrix() @ (pelvis_local * s)
    return ob, r_world


def build_scene(rng):
    for ob in list(bpy.data.objects):
        if ob.type == "MESH":
            bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials):
        for item in list(coll):
            coll.remove(item)
    humans.TUBE_MATS.clear()
    sc = bpy.context.scene
    world = sc.world
    wn = world.node_tree.nodes
    for nd in list(wn):
        if nd.type == "TEX_SKY":
            wn.remove(nd)
    set_camera(CFG["camera"])
    waves = Waves(rng, CFG["water"])
    ANIM["waves"] = waves
    build_water(rng, waves, CFG["water"])
    build_pool(rng, CFG["scene"])
    static = set(bpy.data.objects)
    sky = wn.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    lc = CFG["light"]
    sky.sun_elevation = math.radians(rng.uniform(*lc["sun_elevation"]))
    sky.sun_rotation = math.radians(rng.uniform(*lc["sun_azimuth"]))
    sky.sun_size = math.radians(0.545)
    sky.air_density = rng.uniform(0.8, 1.2)
    sky.dust_density = rng.uniform(*lc["dust"])
    world.node_tree.links.new(sky.outputs["Color"], wn["Background"].inputs["Color"])
    wn["Background"].inputs["Strength"].default_value = rng.uniform(*lc["strength"])

    cc = CFG["crowd"]
    lib = load_lib(cc["human_lib"]) if cc.get("human_lib") else None
    spots, kinds = sample_crowd(rng, cc, sc, sc.camera)
    people = []
    for k, ((x, y), kind) in enumerate(zip(spots, kinds, strict=False)):
        pid = k + 1
        wz = float(waves(x, y))
        if kind == "lounge":
            ob, r = place_lounger(rng, pid, x, y, wz, lib)
            bpy.context.view_layer.update()
            hz = ob.matrix_world.translation.z
            people.append(
                {
                    "id": pid,
                    "head": list(ob.matrix_world.translation),
                    "r": r,
                    "pose": "tube",
                    "under": float(
                        np.clip((wz - (hz - r * 1.08)) / (2 * r * 1.08), 0, 1)
                    ),
                    "role": "swimmer",
                }
            )
            continue
        u = rng.random()
        # the far end is the beach: mostly people standing waist to chest deep
        if y / POOL_L > cc.get("beach_from", 1.0) and rng.random() < cc.get(
            "beach_stand_prob", 0.0
        ):
            u = cc["pose_mix"][2] + 1e-6
        mix = cc["pose_mix"]  # cumulative: vertical, arms_up, swim, shallow, low
        if u < mix[0]:
            pose, dz, pitch = (
                "vertical",
                rng.uniform(0.05, 0.25),
                rng.uniform(-0.15, 0.15),
            )  # head and shoulders out
        elif u < mix[1]:
            pose, dz, pitch = "arms_up", rng.uniform(0.12, 0.32), rng.uniform(-0.1, 0.1)
        elif u < mix[2]:
            pose, dz, pitch = (
                "swim",
                rng.uniform(-0.1, -0.02),
                -rng.uniform(1.0, 1.3),
            )  # body slopes down into the water
        elif u < mix[3]:
            pose, dz, pitch = (
                "shallow",
                rng.uniform(*CFG["crowd"].get("beach_dz", [0.25, 0.6])),
                rng.uniform(-0.05, 0.05),
            )  # chest- to waist-deep at the beach
        else:
            pose, dz, pitch = "low", rng.uniform(-0.09, 0.06), rng.uniform(-0.2, 0.2)
        yaw = rng.uniform(0, 2 * math.pi)
        if lib:
            ob, r = lib_person(rng, lib, pose, pid)
            ob.rotation_euler = (
                0,
                0,
                yaw,
            )  # library poses are already oriented (prone is baked horizontal)
            if pose == "swim":
                dz = rng.uniform(-0.04, 0.04)
        else:
            ob, r = humans.build_person(rng, pid, pose)
            ob.rotation_euler = (pitch, 0, yaw)
        head_z = wz + dz
        ob.location = (x, y, head_z)
        if rng.random() < cc["splash_prob"]:
            humans.build_splash(
                rng,
                x + 0.2 * math.sin(yaw),
                y - 0.2 * math.cos(yaw),
                wz,
                rng.uniform(0.1, 0.5),
            )
        under = float(np.clip((wz - (head_z - r * 1.08)) / (2 * r * 1.08), 0, 1))
        people.append(
            {
                "id": pid,
                "head": [x, y, head_z],
                "r": r,
                "pose": pose,
                "under": under,
                "role": "swimmer",
            }
        )
    for _ in range(int(cc.get("empty_tubes", 0))):
        y = rng.uniform(2, POOL_L - 1)
        x = rng.uniform(-half_w(y) + 1, half_w(y) - 1)
        if all(math.hypot(x - q[0], y - q[1]) > 1.2 for q in spots):
            humans.build_tube(rng, (x, y, float(waves(x, y)) + 0.05))
    extra, _ = humans.build_clutter(
        rng,
        2 * half_w(CFG["scene"].get("rope_y") or 8),
        POOL_L,
        waves,
        len(spots) + 1,
        rope_y=CFG["scene"].get("rope_y"),
    )
    bpy.context.view_layer.update()
    for ob, r, role in extra:
        c = ob.matrix_world.translation
        people.append(
            {
                "id": int(ob["category_id"]),
                "head": [c.x, c.y, c.z],
                "r": r,
                "pose": role,
                "under": 0.0,
                "role": role,
            }
        )
    # everything placed in the pool after the pool itself floats on the waves between clip frames
    ANIM["floaters"] = [
        (o, o.location.x, o.location.y, o.location.z)
        for o in bpy.data.objects
        if o not in static
        and o.type == "MESH"
        and 0 < o.location.y < POOL_L
        and abs(o.location.x) < half_w(o.location.y)
    ]
    return people


def set_time(t):
    """Move the scene to t seconds from the labelled frame: swell, ripples, bobbing floaters."""
    waves = ANIM["waves"]
    me = bpy.data.objects["water"].data
    gx, gy = ANIM["grid"]
    co = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co[: len(gx), 2] = waves(gx, gy, t)
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    for n in ANIM["ripples"]:
        n.inputs["W"].default_value = CFG["water"].get("ripple_speed", 4.0) * t
    for o, x, y, z in ANIM["floaters"]:
        o.location.z = z + float(waves(x, y, t) - waves(x, y, 0.0))
    bpy.context.view_layer.update()


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA", "METAL"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            if any(d.type == dev_type for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = (
            d.type != "CPU"
        )  # GPUs only: CPU in a hybrid render slows multi-GPU machines
    print(
        "DEVICES",
        prefs.compute_device_type,
        [(d.name, d.use) for d in prefs.devices],
        flush=True,
    )
    sc.cycles.device = "GPU"
    sc.render.resolution_x, sc.render.resolution_y = RES, RES * 9 // 16
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.dither_intensity = 0.0
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam_data.clip_end = 400
    sc.view_layers[0].use_pass_z = True


def rgb_mode():
    sc = bpy.context.scene
    sc.view_layers[0].material_override = None
    sc.cycles.samples = SAMPLES
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.04
    sc.cycles.use_denoising = CFG.get("denoise", True)
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 8, 2, 3
    (
        sc.cycles.transmission_bounces,
        sc.cycles.transparent_max_bounces,
        sc.cycles.volume_bounces,
    ) = 8, 8, 1
    sc.cycles.sample_clamp_indirect = 8.0
    sc.cycles.blur_glossy = 0.0  # keep sun glints sharp
    sc.cycles.filter_width = 1.5
    sc.view_settings.view_transform = CFG["light"].get("view_transform", "AgX")
    sc.view_settings.look = CFG["light"].get("look", "None")
    sc.view_settings.exposure = CFG["light"].get("exposure", 0.0)
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = sc[
        "sky_strength"
    ]


def seg_material():
    """Flat emission whose colour encodes the object's category_id: r = id % 256, g = id // 256."""
    m = bpy.data.materials.get("seg") or bpy.data.materials.new("seg")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = "category_id"
    mod = nt.nodes.new("ShaderNodeMath")
    mod.operation = "MODULO"
    mod.inputs[1].default_value = 256
    r = nt.nodes.new("ShaderNodeMath")
    r.operation = "DIVIDE"
    r.inputs[1].default_value = 255
    dv = nt.nodes.new("ShaderNodeMath")
    dv.operation = "DIVIDE"
    dv.inputs[1].default_value = 256
    fl = nt.nodes.new("ShaderNodeMath")
    fl.operation = "FLOOR"
    g = nt.nodes.new("ShaderNodeMath")
    g.operation = "DIVIDE"
    g.inputs[1].default_value = 255
    comb = nt.nodes.new("ShaderNodeCombineColor")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    L = nt.links.new
    L(at.outputs["Fac"], mod.inputs[0])
    L(mod.outputs[0], r.inputs[0])
    L(at.outputs["Fac"], dv.inputs[0])
    L(dv.outputs[0], fl.inputs[0])
    L(fl.outputs[0], g.inputs[0])
    L(r.outputs[0], comb.inputs[0])
    L(g.outputs[0], comb.inputs[1])
    L(comb.outputs[0], em.inputs["Color"])
    L(em.outputs[0], out.inputs["Surface"])
    return m


def seg_mode():
    sc = bpy.context.scene
    sc.view_layers[0].material_override = seg_material()
    sc.cycles.samples = 1
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0
    sc.cycles.filter_width = 0.01
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0


def read_seg(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return (np.rint(px[..., 0] * 255) + 256 * np.rint(px[..., 1] * 255)).astype(
        np.uint16
    )


def main():
    setup_render()
    os.makedirs(args.out, exist_ok=True)
    sc = bpy.context.scene
    rng = np.random.default_rng(SEED)
    n = len([f for f in os.listdir(args.out) if f.startswith("labels_")])
    W, H = sc.render.resolution_x, sc.render.resolution_y
    for s in range(args.frames):
        t0 = time.time()
        people = build_scene(rng)
        print(f"TIMING build {time.time() - t0:.1f}s", flush=True)
        sc["sky_strength"] = (
            sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value
        )
        nm = f"{n + s:04d}"
        rgb_mode()
        keep = args.clip // 2 if args.keep is None else args.keep
        for f in range(args.clip):  # neighbours first, the labelled frame (t = 0) last
            if f == keep:
                continue
            set_time((f - keep) / 60.0)
            sc.render.filepath = os.path.abspath(
                os.path.join(args.out, f"clip_{nm}_{f:02d}.png")
            )
            tr = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"TIMING render {time.time() - tr:.1f}s", flush=True)
        set_time(0.0)
        sc.render.filepath = os.path.abspath(os.path.join(args.out, f"rgb_{nm}.png"))
        tr = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"TIMING render {time.time() - tr:.1f}s", flush=True)
        if args.clip > 1:
            with open(os.path.join(args.out, f"clip_{nm}.json"), "w") as fh:
                json.dump({"n": args.clip, "keep": keep}, fh)
        seg_mode()
        seg_path = os.path.abspath(os.path.join(args.out, f"segimg_{nm}.png"))
        sc.render.filepath = seg_path
        tr = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"TIMING render {time.time() - tr:.1f}s", flush=True)
        seg = read_seg(seg_path)
        os.remove(seg_path)
        np.save(os.path.join(args.out, f"seg_{nm}.npy"), seg)
        counts = np.bincount(seg.ravel(), minlength=max(p["id"] for p in people) + 2)
        cam = sc.camera
        fpx = (W / 2) / math.tan(cam.data.angle / 2)
        lab = {"config": CFG, "seed": SEED, "scene": s, "people": []}
        for p in people:
            co = Vector(p["head"])
            ndc = world_to_camera_view(sc, cam, co)
            d = (co - cam.location).length
            lab["people"].append(
                {
                    **p,
                    "u": float(ndc.x * W),
                    "v": float((1 - ndc.y) * H),
                    "in_front": bool(ndc.z > 0),
                    "dist_m": float(d),
                    "head_r_px": float(fpx * p["r"] / d),
                    "visible_px": int(counts[p["id"]]),
                }
            )
        with open(os.path.join(args.out, f"labels_{nm}.json"), "w") as f:
            json.dump(lab, f)
        print(f"wrote frame {nm} ({len(people)} people)", flush=True)


main()
