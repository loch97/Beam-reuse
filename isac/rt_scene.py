"""Scene assembly and the path extraction used in the RT comparison."""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
NO_INTERACTION = 4294967295
C_LIGHT = 299792458.0


@dataclass
class Pose:
    phi_az_deg: float = 45.0
    r_h: float = 30.0
    h_rsu: float = 5.0
    rx_offset: float = 0.02
    target_xy: tuple = (0.0, 0.0)
    target_heading_rad: float = 0.0


def rsu_position(pose: Pose):
    a = np.deg2rad(pose.phi_az_deg)
    target = np.asarray(pose.target_xy, float)
    return np.array(
        [target[0] + pose.r_h * np.cos(a), target[1] + pose.r_h * np.sin(a), pose.h_rsu]
    )


def place_devices(scene, pose: Pose, boresight_to_target=True):
    """Protocol A of the plan: boresight at the vehicle centre, so height comparisons are
    not contaminated by antenna-pattern loss."""
    from sionna.rt import Transmitter, Receiver, PlanarArray

    p = rsu_position(pose)
    target = np.asarray(pose.target_xy, float)
    look = np.array([target[0], target[1], 0.5]) - p
    az = float(np.arctan2(look[1], look[0]))
    el = float(np.arctan2(look[2], np.hypot(look[0], look[1])))
    orient = [az, el, 0.0] if boresight_to_target else [np.pi, 0.0, 0.0]
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter("tx", position=[float(x) for x in p], orientation=orient))
    scene.add(
        Receiver(
            "rx",
            position=[float(p[0] + pose.rx_offset), float(p[1]), float(p[2])],
            orientation=orient,
        )
    )
    return (p, orient)


@dataclass
class RTCfg:
    max_depth: int = 3
    samples_per_src: int = 2000000
    los: bool = False
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = False
    diffraction: bool = False
    edge_diffraction: bool = False
    seed: int = 1


def solve(scene, rt: RTCfg):
    from sionna.rt import PathSolver

    solver = PathSolver()
    return solver(
        scene,
        max_depth=rt.max_depth,
        los=bool(rt.los),
        samples_per_src=rt.samples_per_src,
        specular_reflection=rt.specular_reflection,
        diffuse_reflection=rt.diffuse_reflection,
        refraction=rt.refraction,
        diffraction=rt.diffraction,
        edge_diffraction=rt.edge_diffraction,
        synthetic_array=True,
        seed=rt.seed,
    )


@dataclass
class SceneCfg:
    L: float
    W: float
    H: float
    clearance: float


def load_vehicle(category, scattering):
    from sionna.rt import load_scene, RadioMaterial, LambertianPattern

    path = ROOT / "assets" / f"complex_low_poly_{category}" / f"complex_low_poly_{category}.xml"
    scene = load_scene(str(path))
    tyre = scene.get("vehicle-tyres")
    proxy = RadioMaterial(
        "tyre-28ghz-proxy",
        relative_permittivity=3.0,
        conductivity=0.02,
        scattering_coefficient=scattering,
    )
    scene.add(proxy)
    tyre.radio_material = proxy
    scene.remove("tyre-proxy")
    scene.frequency = 28000000000.0
    for material in scene.radio_materials.values():
        material.scattering_coefficient = scattering
        material.scattering_pattern = LambertianPattern()
    box = scene.mi_scene.bbox()
    lo = np.array([float(x) for x in box.min])
    hi = np.array([float(x) for x in box.max])
    return (scene, hi - lo, float(lo[2]))


def add_scene_meshes(scene, xml_path):
    from sionna.rt import ITURadioMaterial, SceneObject

    path = Path(xml_path)
    objects = []
    for shape in ET.parse(path).getroot().findall("shape"):
        material = shape.find("ref").get("id").removeprefix("mat-")
        if material not in scene.radio_materials:
            scene.add(ITURadioMaterial(material, material.removeprefix("itu_"), thickness=0.2))
        objects.append(
            SceneObject(
                fname=str(path.parent / shape.find("string[@name='filename']").get("value")),
                name="env-" + shape.get("id"),
                radio_material=scene.radio_materials[material],
            )
        )
    scene.edit(add=objects)


def pose_vehicle_only(scene, initial_positions, target_xy, heading_rad, centre_xy):
    """Pose the vehicle without touching the environment.

    ``p14_rt_trajectory_cache._set_vehicle_pose`` iterates over every object in the
    scene, which is correct when the scene *is* the vehicle but would drag the ground and
    the blocks along with it once an environment is present.  ``initial_positions`` is
    captured before the environment is added, so its keys name exactly the vehicle
    components; this applies the identical rigid transform to those and leaves the rest
    of the scene fixed.
    """
    target_xy = np.asarray(target_xy, float)
    centre = np.asarray(centre_xy, float)
    ch, sh = (np.cos(float(heading_rad)), np.sin(float(heading_rad)))
    rot = np.array([[ch, -sh], [sh, ch]], float)
    for name, start in initial_positions.items():
        obj = scene.objects.get(name)
        if obj is None:
            continue
        obj.orientation = [float(heading_rad), 0.0, 0.0]
        p0 = np.asarray(start, float)
        xy = target_xy + rot @ (p0[:2] - centre)
        obj.position = [float(xy[0]), float(xy[1]), float(p0[2])]


def _set_vehicle_velocity(scene, vehicle_names, velocity_xy):
    """Assign the translational velocity used by Sionna's path-wise Doppler solver.

    ``SceneObject.velocity`` is the API input from which Sionna computes each path's
    Doppler shift.  Only the vehicle objects captured before environment meshes are
    added receive this velocity; the ground/buildings remain stationary.
    """
    vx, vy = (float(velocity_xy[0]), float(velocity_xy[1]))
    for name in vehicle_names:
        if name in scene.objects:
            scene.objects[name].velocity = [vx, vy, 0.0]


def _set_devices(scene, pose: Pose, boresight_az_rad: float, boresight_el_rad: float):
    """Move the TX/RX pair while keeping one fixed array boresight for the track."""
    p = rsu_position(pose)
    orient = [float(boresight_az_rad), float(boresight_el_rad), 0.0]
    tx = scene.transmitters["tx"]
    rx = scene.receivers["rx"]
    tx.position = [float(x) for x in p]
    rx.position = [float(p[0] + pose.rx_offset), float(p[1]), float(p[2])]
    tx.orientation = orient
    rx.orientation = orient


def add_entity(scene, case, draw):
    from sionna.rt import SceneObject, ITURadioMaterial, RadioMaterial, LambertianPattern

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    if case["entity"] == "pedestrian":
        path = assets / "pedestrian_proxy.ply"
        rm = RadioMaterial(
            "served-human-proxy",
            relative_permittivity=20.0,
            conductivity=1.0,
            scattering_coefficient=0.2,
        )
        rm.scattering_pattern = LambertianPattern()
        scene.add(rm)
        objects = [SceneObject(fname=str(path), name="served-human", radio_material=rm)]
    else:
        folder = ROOT / "assets" / f"complex_low_poly_{case['entity']}"
        root = ET.parse(folder / f"complex_low_poly_{case['entity']}.xml").getroot()
        objects = []
        for shape in root.findall("shape"):
            name = shape.get("id")
            kind = "tyres" if "tyre" in name else "glass" if "glaz" in name else "metal"
            matname = "served-" + kind
            if matname not in scene.radio_materials:
                if kind == "tyres":
                    rm = RadioMaterial(
                        matname,
                        relative_permittivity=3.0,
                        conductivity=0.02,
                        scattering_coefficient=draw["scattering"]["vehicle_tyres"],
                    )
                else:
                    rm = ITURadioMaterial(
                        matname,
                        kind,
                        thickness=0.2,
                        scattering_coefficient=draw["scattering"]["vehicle_" + kind],
                    )
                rm.scattering_pattern = LambertianPattern()
                scene.add(rm)
            path = folder / shape.find("string[@name='filename']").get("value")
            objects.append(
                SceneObject(
                    fname=str(path),
                    name="served-" + name,
                    radio_material=scene.radio_materials[matname],
                )
            )
    scene.edit(add=objects)
    lo = np.min([[float(o.mi_mesh.bbox().min[k]) for k in range(3)] for o in objects], axis=0)
    hi = np.max([[float(o.mi_mesh.bbox().max[k]) for k in range(3)] for o in objects], axis=0)
    centre = (lo + hi) / 2
    h = case["ue_heading_rad"]
    rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
    for obj in objects:
        p = np.asarray(obj.position, float).reshape(3)
        xy = np.array(case["ue_xy"]) + rot @ (p[:2] - centre[:2])
        ground = 0.15 if case["entity"] == "pedestrian" else 0.0
        obj.orientation = [h, 0.0, 0.0]
        obj.position = [float(xy[0]), float(xy[1]), float(p[2] - lo[2] + ground)]
    antenna = np.r_[case["ue_xy"], hi[2] - lo[2] + 0.15]
    if case["entity"] == "pedestrian":
        antenna = np.r_[
            np.array(case["ue_xy"]) + 0.35 * np.array([math.cos(h), math.sin(h)]), 1.35
        ]
    return ([o.name for o in objects], antenna, (hi - lo).tolist())


def setup(case, draw, source):
    from sionna.rt import ITURadioMaterial, LambertianPattern

    scene, dims, clearance = load_vehicle(case["target_class"], 0.2)
    for name, obj in scene.objects.items():
        group = (
            "vehicle_tyres"
            if "tyre" in name
            else "vehicle_glass" if "glaz" in name else "vehicle_metal"
        )
        obj.radio_material.scattering_coefficient = draw["scattering"][group]
        obj.radio_material.scattering_pattern = LambertianPattern()
    bb = scene.mi_scene.bbox()
    centre = np.array([(float(bb.min[i]) + float(bb.max[i])) / 2 for i in range(2)])
    initial = {
        n: np.array(o.position, float).reshape(3).copy() for n, o in scene.objects.items()
    }
    add_scene_meshes(scene, str(ROOT / "assets/crossroad/crossroad.xml"))
    mats = {}
    for inv in source["inventory"]:
        group = inv["group"]
        if group not in mats:
            m = ITURadioMaterial(
                "div-" + group,
                inv["itu_material"],
                thickness=0.2,
                scattering_coefficient=draw["scattering"][group],
            )
            m.scattering_pattern = LambertianPattern()
            scene.add(m)
            mats[group] = m
        scene.objects["env-" + inv["object"]].radio_material = mats[group]
    r = case["row"]
    pose = Pose(
        phi_az_deg=r["rsu_bearing_deg"],
        r_h=r["range_horizontal_m"],
        h_rsu=7.5,
        target_xy=tuple(r["target_xy_m"]),
        target_heading_rad=r["heading_rad"],
    )
    place_devices(scene, pose)
    cfg = SceneCfg(
        L=float(dims[0]), W=float(dims[1]), H=float(dims[2]), clearance=float(clearance)
    )
    return (scene, initial, centre, cfg)


def solve_one(scene, row, initial, centre, cfg, protocol, spp):
    pose_vehicle_only(scene, initial, row["target_xy_m"], row["heading_rad"], centre)
    _set_vehicle_velocity(
        scene,
        initial.keys(),
        row["speed_mps"] * np.array([np.cos(row["heading_rad"]), np.sin(row["heading_rad"])]),
    )
    pose = Pose(
        phi_az_deg=row["rsu_bearing_deg"],
        r_h=row["range_horizontal_m"],
        h_rsu=7.5,
        target_xy=tuple(row["target_xy_m"]),
        target_heading_rad=row["heading_rad"],
    )
    bore = np.deg2rad(row["boresight_az_deg"])
    _set_devices(scene, pose, bore, 0.0)
    paths = solve(
        scene, RTCfg(max_depth=3, samples_per_src=spp, seed=protocol["raytracing"]["rt_seed"])
    )
    tau = np.asarray(paths.tau).reshape(-1)
    amplitudes = np.asarray(paths.a)
    ex = dict(
        ok=np.asarray(paths.valid).reshape(-1)
        & np.isfinite(tau)
        & (tau > 0)
        & ((amplitudes[0] ** 2 + amplitudes[1] ** 2).reshape(-1) > 0),
        tau=tau,
        r_mono=C_LIGHT * tau / 2,
        doppler_hz=np.asarray(paths.doppler).reshape(-1),
    )
    ob = np.asarray(paths.objects).reshape(np.asarray(paths.objects).shape[0], -1)
    vertices = np.asarray(paths.vertices).reshape(ob.shape[0], -1, 3)
    first = np.full((ob.shape[1], 3), np.nan)
    last = first.copy()
    seen = np.zeros(ob.shape[1], bool)
    for depth in range(len(ob)):
        real = ob[depth] != NO_INTERACTION
        first[real & ~seen] = vertices[depth, real & ~seen]
        last[real] = vertices[depth, real]
        seen |= real
    tx = np.array(scene.transmitters["tx"].position, float).reshape(3)
    rx = np.array(scene.receivers["rx"].position, float).reshape(3)
    dtx = first - tx
    drx = last - rx
    nt = np.linalg.norm(dtx, axis=1)
    nr = np.linalg.norm(drx, axis=1)
    direction_t = dtx / np.maximum(nt[:, None], 1e-20)
    direction_r = drx / np.maximum(nr[:, None], 1e-20)
    axis = np.array([-np.sin(bore), np.cos(bore), 0.0])
    ut = direction_t @ axis
    ur = direction_r @ axis
    phit = (np.arctan2(dtx[:, 1], dtx[:, 0]) - bore + np.pi) % (2 * np.pi) - np.pi
    phir = (np.arctan2(drx[:, 1], drx[:, 0]) - bore + np.pi) % (2 * np.pi) - np.pi
    thetat = np.rad2deg(np.arccos(np.clip(direction_t[:, 2], -1, 1)))
    thetar = np.rad2deg(np.arccos(np.clip(direction_r[:, 2], -1, 1)))
    vehicle_ids = [int(scene.objects[n].object_id) for n in initial]
    hit = np.isin(ob, vehicle_ids).any(axis=0)
    a = np.asarray(paths.a)
    amp = (a[0] + 1j * a[1]).reshape(-1)
    valid = (
        ex["ok"]
        & seen
        & (ex["r_mono"] > 0)
        & (ex["r_mono"] <= 250)
        & np.isfinite(ut)
        & np.isfinite(ur)
    )
    data = dict(
        a=amp[valid],
        tau=ex["tau"][valid],
        r=ex["r_mono"][valid],
        ut=ut[valid],
        ur=ur[valid],
        phi_t=phit[valid],
        phi_r=phir[valid],
        theta_t=thetat[valid],
        theta_r=thetar[valid],
        doppler=ex["doppler_hz"][valid],
        target=hit[valid],
        objects=ob[:, valid],
        direction_t=direction_t[valid],
        direction_r=direction_r[valid],
    )
    return data
