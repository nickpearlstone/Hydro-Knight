"""Build a library of posed, dressed MPFB humans (CC0 MakeHuman assets) for the pool scenes.

Each human: MakeHuman body with continuous age (kids to older adults), gender, weight, muscle and
race mix; a skin chosen to match (young skins on kids); hair, eyebrows and eyes; swimwear painted
on the body faces by rest-pose region (trunks, one-piece, bikini, rash guard, T-shirt), so no
body is ever rendered without swimwear; a wet-skin gloss; one of five poses (tread, armsup,
recline, prone, stand). The rig is baked into a single static mesh whose origin is the head
centre; custom props hold pose, head radius, pelvis offset and age.

Needs the MPFB extension + CC0 system asset pack installed under synthetic/mpfb (see README):
    synthetic/blender/mpfb_env.sh synthetic/blender-bin/Blender.app/Contents/MacOS/Blender -b \\
        -P synthetic/blender/make_human_lib.py -- --out synthetic/mpfb/humans.blend --per-pose 14
"""

import argparse
import importlib
import math
import os
import random
import sys

import addon_utils
import bpy
from mathutils import Matrix, Quaternion, Vector

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-pose", type=int, default=14)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sheet", default="", help="optional PNG contact sheet of the library")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

addon_utils.enable("bl_ext.user_default.mpfb", default_set=True)
# MPFB only exists once the extension is enabled, so import it after enable()
_svc = "bl_ext.user_default.mpfb.services"
HumanService = importlib.import_module(f"{_svc}.humanservice").HumanService
LocationService = importlib.import_module(f"{_svc}.locationservice").LocationService

bpy.ops.wm.read_homefile(use_empty=True)
R = math.radians
X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
DATA = LocationService.get_user_data()


def assets(kind, ext):
    hits = []
    for base in {DATA, LocationService.get_mpfb_data()}:
        for root, _, files in os.walk(os.path.join(base, kind)):
            hits += [os.path.join(root, f) for f in files if f.endswith(ext)]
    return sorted(set(hits))


SKINS = {os.path.basename(os.path.dirname(s)): s for s in assets("skins", ".mhmat")}
HAIR = {os.path.basename(os.path.dirname(h)): h for h in assets("hair", ".mhclo")}
BROWS = assets("eyebrows", ".mhclo")
EYES = [e for e in assets("eyes", ".mhclo") if "low" in e.lower()] or assets(
    "eyes", ".mhclo"
)
SUIT_COLS = [
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
    (0.03, 0.03, 0.03),
    (0.10, 0.10, 0.18),
    (0.02, 0.02, 0.03),
    (0.05, 0.07, 0.20),
    (0.15, 0.15, 0.15),
]


def lin(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def rot(rig, bone, axis_world, deg):
    """Rotate a pose bone about a world axis by deg at its joint. Character faces -Y, left = +X."""
    bpy.context.view_layer.update()
    pb = rig.pose.bones[bone]
    pb.rotation_mode = "QUATERNION"
    local_axis = pb.matrix.to_quaternion().inverted() @ axis_world
    pb.rotation_quaternion = pb.rotation_quaternion @ Quaternion(local_axis, R(deg))


def j(rng, a=12):
    return rng.uniform(-a, a)


def pose_tread(rig, rng):  # upright, arms sculling just under the surface, scissor kick
    for s, sgn in (("l", 1), ("r", -1)):
        rot(
            rig, f"upperarm_{s}", Y * sgn, 30 + j(rng, 10)
        )  # rest pose is a T-pose: +Y*sgn lowers the arm
        rot(rig, f"upperarm_{s}", -X, 35 + j(rng))
        rot(
            rig, f"lowerarm_{s}", -X, 10 + j(rng, 10)
        )  # forearms stay low, sculling under water
        rot(rig, f"thigh_{s}", -X, 25 + 15 * sgn + j(rng, 8))
        rot(rig, f"calf_{s}", X, 45 + j(rng, 10))


def pose_armsup(rig, rng):
    for s, sgn in (("l", 1), ("r", -1)):
        rot(rig, f"upperarm_{s}", -Y * sgn, 140 + j(rng, 15))
        rot(rig, f"lowerarm_{s}", -Y * sgn, 15 + j(rng))


def pose_recline(
    rig, rng
):  # in an inner tube: leaned back, thighs up, knees over the rim, arms on the rim
    for s, sgn in (("l", 1), ("r", -1)):
        rot(rig, f"thigh_{s}", -X, 90 + j(rng, 10))
        rot(rig, f"calf_{s}", X, 80 + j(rng, 10))
        rot(rig, f"upperarm_{s}", -Y * sgn, 15 + j(rng))
        rot(rig, f"upperarm_{s}", X, 30 + j(rng))
    rot(rig, "neck_01", -X, 30 + j(rng, 8))
    rig.rotation_euler = (R(-50 + j(rng, 8)), 0, 0)


def pose_prone(rig, rng):  # freestyle: horizontal, one arm reaching forward
    rig.rotation_euler = (R(-80 + j(rng, 6)), 0, 0)
    a, b = ("l", "r") if rng.random() < 0.5 else ("r", "l")
    sa = 1 if a == "l" else -1
    rot(rig, f"upperarm_{a}", -Y * sa, 160)
    rot(rig, f"upperarm_{b}", X, 20 + j(rng))
    rot(rig, "head", -X, 25)
    rot(rig, "thigh_l", -X, 10 + j(rng, 6))
    rot(rig, "thigh_r", X, 10 + j(rng, 6))


def pose_stand(
    rig, rng
):  # waist- to chest-deep at the beach end, in one of several everyday stances
    stance = rng.choice(
        ["down", "down", "on_water", "on_water", "one_up", "hips", "hold"]
    )
    for s, sgn in (("l", 1), ("r", -1)):
        if stance == "down":
            rot(rig, f"upperarm_{s}", Y * sgn, 65 + j(rng, 12))
            rot(rig, f"lowerarm_{s}", -X, 15 + j(rng, 15))
        elif stance == "on_water":  # forearms resting on / skimming the surface
            rot(rig, f"upperarm_{s}", Y * sgn, 50 + j(rng, 10))
            rot(rig, f"upperarm_{s}", -X, 30 + j(rng, 10))
            rot(rig, f"lowerarm_{s}", -X, 50 + j(rng, 15))
        elif stance == "hips":
            rot(rig, f"upperarm_{s}", Y * sgn, 60 + j(rng, 8))
            rot(rig, f"lowerarm_{s}", Y * sgn, -70 + j(rng, 10))
        elif stance == "hold":  # arms in front, as if holding a child or a tube
            rot(rig, f"upperarm_{s}", Y * sgn, 45 + j(rng, 10))
            rot(rig, f"upperarm_{s}", -X, 55 + j(rng, 10))
            rot(rig, f"lowerarm_{s}", -Y * sgn, 60 + j(rng, 15))
    if stance == "one_up":
        s, sgn = ("l", 1) if rng.random() < 0.5 else ("r", -1)
        o, osg = ("r", -1) if s == "l" else ("l", 1)
        rot(rig, f"upperarm_{s}", -Y * sgn, 110 + j(rng, 25))
        rot(rig, f"lowerarm_{s}", -Y * sgn, 20 + j(rng, 20))
        rot(rig, f"upperarm_{o}", Y * osg, 60 + j(rng, 10))


def natural(rig, rng):
    """Small whole-body variation so no two people stand like mannequins: twist, lean, look around."""
    rot(rig, "spine_02", Z, j(rng, 15))
    rot(rig, "spine_01", X, j(rng, 6))
    rot(rig, "spine_01", Y, j(rng, 5))
    rot(rig, "neck_01", Z, j(rng, 25))
    rot(rig, "head", X, j(rng, 12))
    rot(rig, "head", Y, j(rng, 8))


POSES = {
    "tread": pose_tread,
    "armsup": pose_armsup,
    "recline": pose_recline,
    "prone": pose_prone,
    "stand": pose_stand,
}


def paint_swimwear(body, style, rng):
    """Second material on body faces chosen by rest-pose region (fractions of body height)."""
    me = body.data
    zs = [v.co.z for v in me.vertices]
    z0, h = min(zs), max(zs) - min(zs)
    col = rng.choice(SUIT_COLS)
    m = bpy.data.materials.new(f"suit_{body.name}")
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*lin(col), 1)
    p.inputs["Roughness"].default_value = 0.45
    p.inputs["Coat Weight"].default_value = 0.4
    me.materials.append(m)
    idx = len(me.materials) - 1
    for poly in me.polygons:
        c = poly.center
        zn, xn, front = (c.z - z0) / h, abs(c.x) / h, c.y < 0
        on = False
        if style == "trunks":
            on = 0.42 <= zn <= 0.57 and xn < 0.13
        elif style == "onepiece":
            on = 0.47 <= zn <= 0.76 and xn < 0.105
        elif style == "bikini":
            on = (0.48 <= zn <= 0.56 and xn < 0.11) or (
                0.68 <= zn <= 0.735 and xn < 0.10 and front
            )
        elif style == "rashguard":
            on = 0.47 <= zn <= 0.82 and xn < 0.17
        elif style == "tee":
            on = (0.52 <= zn <= 0.82 and xn < 0.17) or (
                0.42 <= zn <= 0.55 and xn < 0.13
            )
        if on:
            poly.material_index = idx


def wet(obj):
    for m in obj.data.materials:
        if m and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    n.inputs["Coat Weight"].default_value = max(
                        n.inputs["Coat Weight"].default_value, 0.5
                    )
                    n.inputs["Coat Roughness"].default_value = 0.08


def darken_hair(body, rng):
    """Most real swimmers have dark (wet) hair: multiply the hair texture by a dark tint."""
    dark = rng.random() < 0.75
    tint = (0.12, 0.09, 0.07) if dark else (0.6, 0.5, 0.4)
    for ob in [
        o
        for o in bpy.data.objects
        if o.parent in (body, body.parent) and "hair" in o.name.lower()
    ]:
        for m in ob.data.materials:
            if not (m and m.use_nodes):
                continue
            nt = m.node_tree
            for n in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
                src = (
                    n.inputs["Base Color"].links[0].from_socket
                    if n.inputs["Base Color"].is_linked
                    else None
                )
                mx = nt.nodes.new("ShaderNodeMix")
                mx.data_type = "RGBA"
                mx.blend_type = "MULTIPLY"
                mx.inputs["Factor"].default_value = 1.0
                mx.inputs[7].default_value = (*lin(tint), 1)
                if src:
                    nt.links.new(src, mx.inputs[6])
                else:
                    mx.inputs[6].default_value = n.inputs["Base Color"].default_value
                nt.links.new(mx.outputs[2], n.inputs["Base Color"])
                n.inputs["Coat Weight"].default_value = 0.6


def choose_skin(rng, macro, kid):
    race = max(macro["race"], key=macro["race"].get)
    g = "male" if macro["gender"] > 0.5 else "female"
    age = (
        "young"
        if kid or macro["age"] < 0.6
        else ("middleage" if macro["age"] < 0.8 else "old")
    )
    name = f"{age}_{race}_{g}"
    if name not in SKINS:
        name = f"young_{race}_{g}"
    return SKINS.get(name)


def make_human(rng, pose, idx):
    kid = rng.random() < 0.3
    g = rng.random()
    r3 = [rng.random() ** 2 for _ in range(3)]
    macro = {
        "gender": g,
        "age": rng.uniform(0.12, 0.3) if kid else rng.uniform(0.42, 0.85),
        "muscle": rng.uniform(0.3, 0.7),
        "weight": rng.uniform(0.3, 0.9) if not kid else rng.uniform(0.3, 0.6),
        "height": rng.uniform(0.3, 0.7),
        "proportions": rng.uniform(0.4, 0.7),
        "cupsize": 0.5,
        "firmness": 0.5,
        "race": {
            "african": r3[0] / sum(r3),
            "asian": r3[1] / sum(r3),
            "caucasian": r3[2] / sum(r3),
        },
    }
    before = set(bpy.data.objects)
    body = HumanService.create_human(macro_detail_dict=macro)
    rig = HumanService.add_builtin_rig(body, "game_engine")
    skin = choose_skin(rng, macro, kid)
    HumanService.set_character_skin(
        skin, body, skin_type="ENHANCED_SSS" if skin else "LAYERED"
    )
    hair_names = (
        ["short01", "short02", "short03", "short04", "afro01"]
        if g > 0.5
        else list(HAIR)
    )
    if g > 0.5 and rng.random() < 0.15:
        hair_names = []
    if hair_names:
        HumanService.add_mhclo_asset(
            HAIR[rng.choice(hair_names)],
            body,
            asset_type="Hair",
            subdiv_levels=0,
            material_type="MAKESKIN",
        )
    darken_hair(body, rng)
    for pool, kind in ((BROWS, "Eyebrows"), (EYES, "Eyes")):
        if pool:
            HumanService.add_mhclo_asset(
                rng.choice(pool),
                body,
                asset_type=kind,
                subdiv_levels=0,
                material_type="MAKESKIN",
            )
    if g > 0.5:
        style = rng.choice(
            ["trunks"] * 5 + ["tee"] * 4 + ["rashguard"] * (3 if kid else 1)
        )
    else:
        style = rng.choice(
            ["onepiece"] * 4
            + ["bikini"] * (0 if kid else 4)
            + ["rashguard"] * (3 if kid else 1)
            + ["tee"]
        )
    paint_swimwear(body, style, rng)
    bpy.context.view_layer.objects.active = rig
    POSES[pose](rig, rng)
    if pose in ("tread", "armsup", "stand"):
        natural(rig, rng)
    bpy.context.view_layer.update()
    hb = rig.pose.bones["head"]
    hh, ht = rig.matrix_world @ hb.head, rig.matrix_world @ hb.tail
    head_c = hh + 0.45 * (ht - hh)
    head_r = 0.7 * (ht - hh).length
    pelvis = rig.matrix_world @ rig.pose.bones["pelvis"].head
    # bake: evaluated copies of every mesh driven by the rig, joined into one object
    dg = bpy.context.evaluated_depsgraph_get()
    parts = []
    for ob in [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and (o.parent == rig or o.parent == body or o == body)
    ]:
        me = bpy.data.meshes.new_from_object(
            ob.evaluated_get(dg), preserve_all_data_layers=False
        )
        me.transform(ob.matrix_world)
        new = bpy.data.objects.new(f"part_{ob.name}", me)
        bpy.context.scene.collection.objects.link(new)
        parts.append(new)
    for o in [o for o in bpy.data.objects if o not in parts and o not in before]:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    h = bpy.context.object
    wet(h)
    h.data.transform(Matrix.Translation(-head_c))
    h.name = f"hum_{pose}_{idx:02d}"
    h["pose"] = pose
    h["head_r"] = head_r
    h["pelvis_off"] = list(pelvis - head_c)
    h["age"] = macro["age"]
    h["kid"] = kid
    h["style"] = style
    for p in h.data.polygons:
        p.use_smooth = True
    h.hide_render = True
    return h


def main():
    rng = random.Random(args.seed)
    lib = []
    for pose in POSES:
        for i in range(args.per_pose):
            hum = make_human(rng, pose, i)
            lib.append(hum)
            print(
                f"HUMAN {hum.name} kid={hum['kid']} style={hum['style']} r={hum['head_r']:.3f}",
                flush=True,
            )
    for m in bpy.data.meshes:
        if m.users == 0:
            bpy.data.meshes.remove(m)
    if args.sheet:
        contact_sheet(lib)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    print("saved", args.out, len(lib))


def contact_sheet(lib):
    """Front view of every human (they face -Y), 14 per row, rows stacked downward."""
    scn = bpy.context.scene
    for k, h in enumerate(lib):
        h.hide_render = False
        h.location = ((k % 14) * 1.3, 0, -(k // 14) * 2.3)
    rows = (len(lib) + 13) // 14
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 19
    cam.location = (8.45, -20, -(rows - 1) * 1.15 - 0.5)
    cam.rotation_euler = (R(90), 0, 0)
    scn.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4
    sun.rotation_euler = (R(60), R(10), R(-20))
    scn.collection.objects.link(sun)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.3, 0.6, 0.65, 1)
    scn.world = world
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.render.resolution_x, scn.render.resolution_y = (
        1800,
        int(1800 * rows * 2.3 / 19) + 40,
    )
    scn.render.filepath = os.path.abspath(args.sheet)
    bpy.ops.render.render(write_still=True)
    for h in lib:
        h.hide_render = True
        h.location = (0, 0, 0)
    bpy.data.objects.remove(cam)
    bpy.data.objects.remove(sun)


main()
