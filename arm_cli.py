r"""
arm_cli.py - drive the Franka in "Arm with Cam.usd" from the command line.

Launches Isaac Sim, opens the stage, then reads commands from the SAME terminal
(or from -c / piped stdin) and runs them one after another while the simulation
keeps stepping in the background.

Launch (Isaac Sim 5.x/6.x, isaacsim.* namespace):
    C:\isaac-sim\python.bat arm_cli.py
    C:\isaac-sim\python.bat arm_cli.py --usd "C:/path/to/other scene.usd"
    C:\isaac-sim\python.bat arm_cli.py --headless -c "open; move 0.4 0.0 0.3; snap" --quit-when-done
    type commands.txt | C:\isaac-sim\python.bat arm_cli.py       (one command per line)

Commands (type `help` at the arm> prompt):
    move X Y Z                    move the gripper to a world position (metres), pointing
                                  straight down (the same top-down grasp pose the pick-place
                                  scripts use).
    move X Y Z ROLL PITCH YAW     ... with an orientation in DEGREES (extrinsic XYZ, exactly
                                  what euler_angles_to_quat expects). Straight down = 0 180 0;
                                  add a yaw to rotate the fingers about vertical.
    move X Y Z q QW QX QY QZ      ... with a quaternion (w, x, y, z).
    move X Y Z keep               ... keeping the current orientation.
    open | close                  open / close the gripper.
    snap [PATH] [depth] [now]     save a picture from the hand camera as PNG. Default path is
                                  captures/capture_<timestamp>.png next to this script. Add
                                  `depth` to also save the depth image as <PATH>_depth.npy.
                                  Add `now` to take it immediately, while the running command
                                  keeps going, instead of waiting in the queue.
    points [PATH] [ply]           save the 3D world position of EVERY pixel of the hand camera.
                                  <PATH>.npy is a float32 array of shape (H, W, 3) holding world
                                  X Y Z per pixel (NaN where there is no depth) and <PATH>.png is
                                  the matching colour image, so image[row, col] <-> xyz[row, col].
                                  Add `ply` to also write a coloured point cloud (<PATH>.ply) that
                                  opens in MeshLab / CloudCompare. Default PATH is
                                  captures/points_<timestamp>. `now` works as for snap.
    watch [SECONDS] [PATH]        keep PATH (default captures/live.png) refreshed with the hand
                                  camera image every SECONDS (default 1) of simulated time, in
                                  the background, whatever else is running. `watch off` ends it.
    pick X Y Z [to PX PY PZ] [yaw DEG]
                                  run the built-in PickPlaceController: grasp the object whose
                                  centre is at X Y Z, release it at PX PY PZ (default: above the
                                  basket), with the fingers rotated by yaw degrees.
    pose                          print the end-effector pose and finger positions (queued, so
                                  it runs at its place in the sequence).
    joints                        print the arm joint angles, speeds, limits, drive gains, link
                                  masses/inertias, and joint 1's PhysX reading against the angle
                                  measured from the link poses (they must agree).
    jog N DEG                     move arm joint N (1..7) to DEG degrees; the other joints hold.
    home                          drive the arm back to the home joint configuration.
    reset                         put the scene back to its start state (objects, arm, gripper,
                                  controllers) without relaunching, like Stop/Play in the GUI.
    wait SECONDS                  pause the command queue for a while.
    status                        immediately print what is running, how many commands are
                                  queued, and the live pose (works while a move is running).
    stop                          abort the running action and drop queued ones.
    record [PATH] | record off    immediately start (or end) recording the session to an MCAP
                                  file for Foxglove: joint states, targets, end-effector pose,
                                  link transforms, hand-camera and viewport images, every
                                  `snap`/`points` capture (point clouds included), every command
                                  and every "[arm]" line. Default PATH is
                                  runs/foxglove/arm_<timestamp>.mcap. See foxglove_recorder.py.
    note TEXT...                  immediately write TEXT into the recording (/notes topic), stamped
                                  at the moment it is typed. Use it to annotate what you see.
    mark TEXT...                  the same, but queued: the note is stamped when its turn comes in
                                  the command sequence (e.g. right after a move finishes).
    think TEXT...                 immediately record overarching ideas / problem solving on
                                  /reasoning/strategy (no prefix needed; the topic says what it is).
    motor TEXT...                 immediately record motor-control reasoning (grasp heights,
                                  waypoints, orientations, why a motion failed) on /reasoning/motor.
                                  None of note / mark / think / motor appear on /commands, which
                                  holds only the arm_cli commands themselves.
    quit                         close Isaac Sim.

Commands run one after another in the order given. `status`, `stop`, `help`, `quit`,
`watch`, `record`, `note`, `think`, `motor` and `snap`/`points` with `now` take effect immediately;
everything else waits its turn.

Positions are in the world frame of the stage. The frame that `move` targets and `pose`
reports is RMPflow's "right_gripper" frame, which sits between the fingertips.

Launch flags: --headless, --usd PATH, -c "cmd; cmd", --quit-when-done, --events-dt a,..,j
(pick-place phase timings), --no-fix-inertia (keep the stage's link inertias; see
Arm.fix_link_inertias for why they are repaired by default), --trace FILE (per-step
joint CSV for diagnosing motion problems), --record [PATH] (start recording to MCAP at
launch; --record-hz, --record-stride and --no-record-viewport tune it).
Tests: test_arm_cli.py, test_foxglove_recorder.py.
"""

import argparse
import os
import queue
import shlex
import threading
from datetime import datetime

# ----------------------------------------------------------------------------
# Launch options. Parsed BEFORE SimulationApp is created because Kit has to know
# about headless mode before it starts.
# ----------------------------------------------------------------------------
DEFAULT_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "worlds", "Arm with Cam.usd"
).replace("\\", "/")


def parse_launch_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--usd", default=DEFAULT_USD_PATH, metavar="PATH",
        help="USD stage to open instead of the default. Accepts an absolute or relative path "
             "with either slash style, a ~ path, or an omniverse:// URL. "
             f"Default: {DEFAULT_USD_PATH}",
    )
    parser.add_argument("--headless", action="store_true", help="run without the Isaac Sim window")
    parser.add_argument(
        "-c", "--commands", default=None,
        help='commands to run at startup, separated by ";"   e.g. -c "open; move 0.4 0 0.3; snap"',
    )
    parser.add_argument(
        "--quit-when-done", action="store_true",
        help="exit after the -c commands finish instead of dropping to the prompt",
    )
    parser.add_argument(
        "--events-dt", default=None,
        help="10 comma-separated per-phase time steps for the pick-place controller "
             "(smaller = slower/gentler), e.g. 0.008,0.005,0.1,0.1,0.01,0.003,0.0025,1,0.008,0.08. "
             "Default: the EVENTS_DT constant in this file.",
    )
    parser.add_argument(
        "--no-fix-inertia", action="store_true",
        help="keep the link inertias authored in the stage. By default links whose rotational "
             "inertia is implausibly small (this stage authors ~1e-6 kg m^2 for 2-3 kg links) get "
             "inertia = mass * (0.06 m)^2, which is what lets joint 1 hold a position.",
    )
    parser.add_argument(
        "--trace", default=None,
        help="write one CSV row per physics step (time, task, joint positions, joint velocities, "
             "physical joint-1 angle) to this file, for diagnosing joint-reading problems",
    )
    parser.add_argument(
        "--record", nargs="?", const="", default=None, metavar="PATH",
        help="record the session to an MCAP file for Foxglove from launch on (same as the "
             "`record` command). Without PATH: runs/foxglove/arm_<timestamp>.mcap",
    )
    parser.add_argument(
        "--record-hz", type=float, default=5.0, metavar="HZ",
        help="camera image rate (simulated seconds) in the recording. Default 5. Joint states, "
             "poses and transforms are always recorded every physics step",
    )
    parser.add_argument(
        "--record-stride", type=int, default=2, metavar="N",
        help="keep every N-th pixel row and column of a `points` cloud in the recording "
             "(1 = all 2 M points of a 1080p frame, about 33 MB each). Default 2",
    )
    parser.add_argument(
        "--no-record-viewport", action="store_true",
        help="do not record the Isaac Sim perspective viewport (/viewport/image)",
    )
    return parser.parse_args()


def resolve_stage_path(path: str) -> str:
    """Normalise --usd and check it exists BEFORE Isaac Sim starts.

    open_stage() would also fail on a bad path, but only after the ~40 s
    SimulationApp start-up, and its error does not say which path it tried.
    URLs (omniverse://, http://) are passed through untouched.
    """
    if "://" in path:
        return path
    resolved = os.path.abspath(os.path.expanduser(path)).replace("\\", "/")
    if not os.path.isfile(resolved):
        raise SystemExit(f"--usd: no such file: {resolved}")
    if not resolved.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
        raise SystemExit(f"--usd: not a USD stage (expected .usd/.usda/.usdc/.usdz): {resolved}")
    return resolved


LAUNCH = parse_launch_args()
LAUNCH.usd = resolve_stage_path(LAUNCH.usd)

from isaacsim import SimulationApp

# SimulationApp MUST be created before any other isaacsim import.
simulation_app = SimulationApp({"headless": LAUNCH.headless})

import numpy as np
from isaacsim.core.utils.extensions import enable_extension

# The Franka example lives in a deprecated extension that is off by default, so
# its module is not importable until we enable it. Must precede the imports below.
enable_extension("isaacsim.robot.manipulators.examples")

from isaacsim.core.api import World
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import (
    euler_angles_to_quat,
    quat_to_euler_angles,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
)
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import (
    PickPlaceController,
    RMPFlowController,
)
from isaacsim.sensors.camera import Camera
from PIL import Image

from foxglove_recorder import CAMERA_FRAME, LOG_INFO, LOG_WARNING, McapRecorder

# ----------------------------------------------------------------------------
# CONFIG - matches the other scripts in this folder.
# ----------------------------------------------------------------------------
FRANKA_PRIM_PATH = "/Franka"
CAMERA_PRIM_PATH = "/Franka/panda_hand/Cube/Hand_Camera"
CAMERA_RESOLUTION = (1920, 1080)
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

# Home configuration of the 7 arm joints (fingers are handled separately).
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
HOME_ARM_JOINTS = np.array([0.0, 0.0, 0.0, -np.pi / 4, 0.0, np.pi / 4, np.pi / 4])

# [0, pi, 0] points the gripper straight down.
TOP_DOWN_QUAT = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

# Default release point for `pick`: above the basket opening (see pick_props_into_basket.py).
BASKET_RELEASE_POSITION = np.array([-0.29964, 0.64953, 0.28])

# Per-phase durations of the PickPlaceController state machine (10 phases).
EVENTS_DT = [0.008, 0.005, 0.1, 0.1, 0.05, 0.05, 0.0025, 1.0, 0.008, 0.08]
if LAUNCH.events_dt:
    EVENTS_DT = [float(t) for t in LAUNCH.events_dt.split(",")]
    if len(EVENTS_DT) != 10:
        raise SystemExit("--events-dt needs exactly 10 comma-separated numbers")
PICK_PLACE_PHASES = [
    "move above the object", "lower onto the object", "settle", "close gripper", "lift",
    "carry to the place position", "lower to place height", "open gripper", "lift", "return",
]

# `move` is considered arrived once BOTH errors stay under tolerance for a few frames.
MOVE_POS_TOL = 0.005        # m
MOVE_ROT_TOL_DEG = 2.0
MOVE_SETTLE_STEPS = 10
MOVE_TIMEOUT_S = 20.0
# A move also ends early once it stops making progress: neither error has improved
# by the margins below for MOVE_STALL_S. Logs showed ~30% of moves hovering (often
# ~6 mm off, just outside tolerance) or pinned far away for the full timeout.
MOVE_STALL_S = 1.5
MOVE_PROGRESS_POS = 0.0005  # m
MOVE_PROGRESS_ROT_DEG = 0.25
MOVE_NEAR_POS_TOL = 0.010   # m: a stall this close is reported as "settled short of target"
MOVE_NEAR_ROT_TOL_DEG = 3.0

# Link inertia repair (see Arm.fix_link_inertias). A link whose inertia trace is
# below mass * MIN_GYRATION^2 is treated as broken and given mass * GYRATION^2 on
# each axis; 6 cm is a typical radius of gyration for these 10-30 cm links.
INERTIA_MIN_GYRATION = 0.01   # m
INERTIA_GYRATION = 0.06       # m

GRIPPER_STEPS = 60          # ~1 s at 60 Hz for the fingers to travel
HOME_JOINT_TOL = 0.01       # rad
HOME_REST_SPEED = 0.05      # rad/s: every joint slower than this counts as "at rest"
HOME_JOINT_SPEED = 0.6      # rad/s for the slowest-to-arrive joint during `home`
HOME_MIN_DURATION_S = 1.0
SNAP_SETTLE_STEPS = 5       # rendered frames to wait so the image matches the settled pose
SNAP_MAX_WAIT_STEPS = 240
WATCH_DEFAULT_PERIOD_S = 1.0
WATCH_DEFAULT_PATH = os.path.join(CAPTURE_DIR, "live.png")

# Foxglove recording (see foxglove_recorder.py and the `record` command).
RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "foxglove")
VIEWPORT_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"   # the GUI's Perspective viewport camera
VIEWPORT_RESOLUTION = (1280, 720)
VIEWPORT_FRAME = "viewport_camera"

EOF_SENTINEL = "\x00eof"

# Commentary goes to its own topics, never to /commands (which holds only arm commands).
REASONING_COMMANDS = {"think": "strategy", "motor": "motor"}
COMMENTARY_COMMANDS = {"note", "mark", *REASONING_COMMANDS}
# ----------------------------------------------------------------------------

_ARM = None  # the live Arm, so say() can copy its output into the recording


def say(msg: str) -> None:
    print(f"[arm] {msg}", flush=True)
    if _ARM is not None and _ARM.recorder is not None:
        level = LOG_WARNING if msg.startswith(("WARNING", "error", "OUTSIDE")) else LOG_INFO
        _ARM.recorder.log(_ARM.sim_time(), msg, level)


def fmt(v) -> str:
    return np.array2string(np.asarray(v, dtype=float), precision=3, suppress_small=True)


def rotation_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Angle of the rotation that takes rot_a to rot_b, in degrees."""
    rel = rot_a.T @ rot_b
    cos = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def to_numpy(data):
    """Annotator output -> numpy (it is a warp array when the annotator lives on the GPU)."""
    if data is None:
        return None
    if hasattr(data, "numpy"):
        data = data.numpy()
    return np.asarray(data)


def camera_rgb(cam: Camera):
    """Latest hand-camera frame as a contiguous uint8 (H, W, 3) array, or None if there is none yet."""
    rgba = to_numpy(cam.get_rgba())
    if rgba is None or rgba.size == 0:
        return None
    rgb = rgba[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


# A USD camera looks down -Z with +Y up; the OpenCV / ROS optical frame looks
# down +Z with +Y down. The same flip converts in either direction.
USD_FROM_CV = np.diag([1.0, -1.0, -1.0])


def unproject_depth(cam: Camera, depth: np.ndarray) -> np.ndarray:
    """World-frame XYZ of every pixel of a distance_to_image_plane image, shape (H, W, 3).

    Built from the camera's world POSE plus intrinsics rather than from
    get_view_matrix_ros(): Hand_Camera inherits a 0.01 scale from its parent
    Cube, the view matrix keeps that scale and shrinks everything toward the
    camera, while get_world_pose() decomposes the transform and drops it
    (see get_position() in pick_place_with_Claude.py). Pixels with no depth
    come back as NaN.
    """
    depth = np.squeeze(np.asarray(depth, dtype=np.float64))
    h, w = depth.shape[:2]
    K = np.asarray(cam.get_intrinsics_matrix(), dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)
    # Pinhole back-projection: the same p_cam = inv(K) @ [u, v, 1] * depth Isaac uses.
    p_cv = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=-1)

    position, quat = cam.get_world_pose(camera_axes="usd")
    world_from_cv = quat_to_rot_matrix(np.asarray(quat, dtype=np.float64)) @ USD_FROM_CV
    return (p_cv @ world_from_cv.T + np.asarray(position, dtype=np.float64)).astype(np.float32)


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> int:
    """Write the valid points as a binary little-endian PLY coloured from the image. Returns the count."""
    mask = np.isfinite(xyz).all(axis=-1)
    pts = xyz[mask]
    cols = rgb[mask]
    rec = np.empty(
        pts.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    return int(pts.shape[0])


class Arm:
    """The stage, the Franka, both controllers and the hand camera."""

    def __init__(self, usd_path: str) -> None:
        global _ARM
        self.recorder = None
        self.viewport_camera = None
        _ARM = self
        say(f"opening stage {usd_path}")
        if not open_stage(usd_path):
            raise RuntimeError(f"open_stage failed - could not open {usd_path}")

        # Referenced assets resolve asynchronously; pump until loading settles.
        for _ in range(500):
            simulation_app.update()
            if not is_stage_loading():
                break
        else:
            say("WARNING: stage still loading after 500 updates.")

        if not is_prim_path_valid(FRANKA_PRIM_PATH):
            raise RuntimeError(f"No prim at {FRANKA_PRIM_PATH!r} in {usd_path}")

        self.world = World(stage_units_in_meters=1.0)
        self.franka = self.world.scene.add(Franka(prim_path=FRANKA_PRIM_PATH, name="franka"))
        self.world.reset()  # initialises physics handles; controllers need that first.
        self.fix_link_inertias()

        self.arm_dof_indices = [self.franka.get_dof_index(n) for n in ARM_JOINT_NAMES]
        self.rmp = RMPFlowController(name="rmpflow", robot_articulation=self.franka)
        self.pick_place = PickPlaceController(
            name="pick_place",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
            events_dt=EVENTS_DT,
        )

        self.camera = None
        if is_prim_path_valid(CAMERA_PRIM_PATH):
            self.camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=CAMERA_RESOLUTION)
            self.camera.initialize()
            self.camera.add_distance_to_image_plane_to_frame()
        else:
            say(f"WARNING: no camera at {CAMERA_PRIM_PATH!r}; `snap` is disabled.")

        # NOTE: do NOT wrap articulation links in SingleRigidPrim to read their poses.
        # Creating a separate rigid-body view over panda_link1 stops that link from
        # rendering (its mesh is left inside the base). link_pose() below reads the
        # poses from the articulation's own PhysX view instead.

        self.trace_file = None
        if LAUNCH.trace:
            os.makedirs(os.path.dirname(os.path.abspath(LAUNCH.trace)), exist_ok=True)
            self.trace_file = open(LAUNCH.trace, "w")
            cols = ["t", "task"] + [f"q{i}" for i in range(1, 8)] + [f"v{i}" for i in range(1, 8)] + ["phys1", "base_yaw"]
            self.trace_file.write(",".join(cols) + "\n")

        self.go_to_default_pose()

    # ---- Foxglove recording -------------------------------------------------------
    def sim_time(self) -> float:
        try:
            return float(self.world.current_time)
        except Exception:
            return 0.0

    def start_recording(self, path: str = "") -> None:
        """Start (or restart, to a new file) the MCAP recording; see foxglove_recorder.py."""
        if self.recorder is not None:
            self.stop_recording()
        if not path:
            path = os.path.join(RECORD_DIR, datetime.now().strftime("arm_%Y%m%d_%H%M%S.mcap"))
        elif not path.lower().endswith(".mcap"):
            path += ".mcap"
        if not LAUNCH.no_record_viewport and self.viewport_camera is None:
            self.viewport_camera = self._attach_viewport_camera()
        self.recorder = McapRecorder(
            path,
            image_period_s=1.0 / max(LAUNCH.record_hz, 1e-3),
            point_stride=LAUNCH.record_stride,
        )
        self.recorder.log(self.sim_time(), f"recording started: stage {LAUNCH.usd}")
        say(f"recording to {self.recorder.path} (images at {LAUNCH.record_hz:g} Hz, "
            f"viewport {'on' if self.viewport_camera is not None else 'off'}; `record off` ends it)")

    def stop_recording(self) -> None:
        if self.recorder is None:
            return
        rec = self.recorder
        self.recorder = None   # say() must not write into a file being closed
        rec.close()
        size_mb = os.path.getsize(rec.path) / 1e6
        say(f"recording saved: {rec.path} ({size_mb:.1f} MB; {rec.summary()})")

    def _attach_viewport_camera(self):
        """A render product on the GUI's Perspective camera, so the recording shows the scene
        the way the user sees it. Returns None (with a warning) when that is not possible."""
        if not is_prim_path_valid(VIEWPORT_CAMERA_PRIM_PATH):
            say(f"WARNING: no viewport camera at {VIEWPORT_CAMERA_PRIM_PATH!r}; /viewport/image is off.")
            return None
        try:
            cam = Camera(prim_path=VIEWPORT_CAMERA_PRIM_PATH, resolution=VIEWPORT_RESOLUTION)
            cam.initialize()
            return cam
        except Exception as exc:
            say(f"WARNING: could not attach to the viewport camera ({exc}); /viewport/image is off.")
            return None

    def link_poses(self):
        """{link name: (position, (w, x, y, z))} for every articulation link, from one PhysX read."""
        view = self.franka._articulation_view
        out = {}
        try:
            tf = np.asarray(view._physics_view.get_link_transforms(), dtype=float).reshape(view.count, -1, 7)[0]
            for i, name in enumerate(view.body_names):
                row = tf[i]
                out[name] = (row[:3].copy(), np.array([row[6], row[3], row[4], row[5]]))
        except Exception:
            for name in view.body_names:
                out[name] = self.link_pose(name)
        return out

    def record_step(self, task_label: str) -> None:
        """One recording sample per physics step: joints, targets, EE pose, transforms, images."""
        rec = self.recorder
        if rec is None:
            return
        t = self.sim_time()
        try:
            q = self.franka.get_joint_positions()
            v = self.franka.get_joint_velocities()
            target_q = target_v = None
            try:
                applied = self.franka.get_articulation_controller().get_applied_action()
                target_q, target_v = applied.joint_positions, applied.joint_velocities
            except Exception:
                pass
            rec.joint_states(t, self.franka.dof_names, q, v, target_q, target_v, task_label)

            pos, rot = self.ee_pose()
            rec.ee_pose(t, pos, rot_matrix_to_quat(rot))

            frames = [(name, p, quat) for name, (p, quat) in self.link_poses().items()]
            if self.camera is not None:
                cam_pos, cam_quat = self.camera.get_world_pose(camera_axes="ros")
                frames.append((CAMERA_FRAME, np.asarray(cam_pos, dtype=float), np.asarray(cam_quat, dtype=float)))
            if self.viewport_camera is not None:
                vp_pos, vp_quat = self.viewport_camera.get_world_pose(camera_axes="ros")
                frames.append((VIEWPORT_FRAME, np.asarray(vp_pos, dtype=float), np.asarray(vp_quat, dtype=float)))
            rec.transforms(t, frames)
            rec.sim(t, int(self.world.current_time_step_index), self.physics_dt())

            if self.camera is not None and rec.image_due("/hand_camera/image", t):
                rgb = camera_rgb(self.camera)
                if rgb is not None and rgb.any():
                    rec.camera_calibration(t, self.camera.get_intrinsics_matrix(), rgb.shape[1], rgb.shape[0])
                    rec.image("/hand_camera/image", t, rgb)
            if self.viewport_camera is not None and rec.image_due("/viewport/image", t):
                rgb = camera_rgb(self.viewport_camera)
                if rgb is not None and rgb.any():
                    rec.image("/viewport/image", t, rgb, frame_id=VIEWPORT_FRAME)
        except Exception as exc:
            # Never let the recording take the simulation down: close the file and stop.
            say(f"WARNING: recording stopped after an error: {exc!r}")
            self.stop_recording()

    def record_capture(self, rgb: np.ndarray, path: str = "", xyz=None) -> None:
        """An image arm_cli handed out (`snap`, `points`, `watch`): goes to /hand_camera/retrieved
        at full quality, plus the world-frame cloud on /hand_camera/points if given."""
        if self.recorder is None:
            return
        t = self.sim_time()
        self.recorder.retrieved_image(t, rgb, path)
        if xyz is not None:
            n = self.recorder.point_cloud(t, xyz, rgb)
            say(f"recorded {n} points on /hand_camera/points (stride {self.recorder.point_stride})")

    def link_pose(self, link_name: str):
        """World position and (w, x, y, z) quaternion of one articulation link.

        Read from the articulation's PhysX view (link transforms are x y z qx qy qz qw),
        falling back to the USD transform. No extra prim views are created.
        """
        view = self.franka._articulation_view
        try:
            tf = np.asarray(view._physics_view.get_link_transforms(), dtype=float).reshape(view.count, -1, 7)[0]
            row = tf[view.get_link_index(link_name)]
            return row[:3].copy(), np.array([row[6], row[3], row[4], row[5]])
        except Exception:
            from pxr import Usd, UsdGeom  # available once SimulationApp is up
            prim = self.world.stage.GetPrimAtPath(f"{FRANKA_PRIM_PATH}/{link_name}")
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = np.array(xf.ExtractTranslation(), dtype=float)
            q = xf.ExtractRotationQuat()
            return pos, np.array([q.GetReal(), *q.GetImaginary()], dtype=float)

    def joint1_physical(self) -> float:
        """Joint 1 angle (rad, in -pi..pi) computed from the link0 -> link1 rotation."""
        _, q0 = self.link_pose("panda_link0")
        _, q1 = self.link_pose("panda_link1")
        rel = quat_to_rot_matrix(q0).T @ quat_to_rot_matrix(q1)
        return float(np.arctan2(rel[1, 0], rel[0, 0]))

    def base_pose(self):
        """World position and yaw (rad) of the base link. It must never move."""
        pos, q0 = self.link_pose("panda_link0")
        rot = quat_to_rot_matrix(q0)
        return pos, float(np.arctan2(rot[1, 0], rot[0, 0]))

    def trace(self, label: str) -> None:
        if self.trace_file is None:
            return
        q = np.asarray(self.franka.get_joint_positions(joint_indices=self.arm_dof_indices), dtype=float)
        v = np.asarray(self.franka.get_joint_velocities(joint_indices=self.arm_dof_indices), dtype=float)
        row = [f"{self.world.current_time:.4f}", label.replace(",", ";")]
        row += [f"{x:.5f}" for x in q] + [f"{x:.5f}" for x in v]
        row += [f"{self.joint1_physical():.5f}", f"{self.base_pose()[1]:.5f}"]
        self.trace_file.write(",".join(row) + "\n")

    def print_joints(self) -> None:
        q = np.asarray(self.franka.get_joint_positions(joint_indices=self.arm_dof_indices), dtype=float)
        v = np.asarray(self.franka.get_joint_velocities(joint_indices=self.arm_dof_indices), dtype=float)
        phys = self.joint1_physical()
        # Reading minus physical angle, wrapped to -pi..pi: ~0 means the reading is
        # consistent with the links (possibly offset by whole turns), anything else
        # means PhysX's reading has drifted away from the actual joint.
        wrapped = (q[0] - phys + np.pi) % (2 * np.pi) - np.pi
        say(f"arm joint positions (rad) : {fmt(q)}")
        say(f"arm joint velocities      : {fmt(v)}")
        try:
            ctrl = self.franka.get_articulation_controller()
            kps, kds = ctrl.get_gains()
            idx = self.arm_dof_indices
            say(f"drive stiffness (kp)      : {fmt(np.asarray(kps, dtype=float)[idx])}")
            say(f"drive damping (kd)        : {fmt(np.asarray(kds, dtype=float)[idx])}")
            say(f"drive max effort          : {fmt(np.asarray(ctrl.get_max_efforts(), dtype=float)[idx])}")
            applied = ctrl.get_applied_action()
            if applied.joint_positions is not None:
                say(f"applied position targets  : {fmt(np.asarray(applied.joint_positions, dtype=float)[idx])}")
            if applied.joint_velocities is not None:
                say(f"applied velocity targets  : {fmt(np.asarray(applied.joint_velocities, dtype=float)[idx])}")
        except Exception as exc:
            say(f"drive gains unavailable ({exc})")
        try:
            view = self.franka._articulation_view
            idx = self.arm_dof_indices
            say(f"joint armature            : {fmt(np.asarray(view.get_armatures(), dtype=float)[0][idx])}")
            say(f"joint friction coeff      : {fmt(np.asarray(view.get_friction_coefficients(), dtype=float)[0][idx])}")
            say(f"max joint velocity        : {fmt(np.asarray(view.get_joint_max_velocities(), dtype=float)[0][idx])}")
            meta = getattr(view._physics_view, "shared_metatype", None)
            if meta is not None:
                say(f"fixed base                : {getattr(meta, 'fixed_base', 'n/a')}")
        except Exception as exc:
            say(f"physx joint properties unavailable ({exc})")
        try:
            view = self.franka._articulation_view
            masses = np.asarray(view.get_body_masses(), dtype=float)[0]
            inertias = np.asarray(view.get_body_inertias(), dtype=float)[0].reshape(len(masses), 3, 3)
            for name, m, inertia in zip(view.body_names, masses, inertias):
                diag = " ".join(f"{x:.2e}" for x in np.diag(inertia))
                say(f"body {name:18s}: mass {m:9.4f} kg, inertia diag [{diag}] kg m^2")
        except Exception as exc:
            say(f"body masses unavailable ({exc})")
        try:
            # (num_envs, num_dofs, 2) on the underlying articulation view; env 0 is ours.
            limits = np.asarray(self.franka._articulation_view.get_dof_limits(), dtype=float)[0][self.arm_dof_indices]
            outside = [ARM_JOINT_NAMES[i] for i in range(7) if not (limits[i, 0] <= q[i] <= limits[i, 1])]
            say(f"joint limits lower        : {fmt(limits[:, 0])}")
            say(f"joint limits upper        : {fmt(limits[:, 1])}")
            if outside:
                say(f"OUTSIDE LIMITS: {', '.join(outside)}")
        except Exception as exc:  # API differences between Isaac versions
            say(f"joint limits unavailable ({exc})")
        say(f"joint 1: reading {q[0]:.4f}, physical (from link poses) {phys:.4f}, "
            f"reading - physical = {q[0] - phys:+.4f} rad = {(q[0] - phys) / (2 * np.pi):+.3f} turns; "
            f"wrapped difference {wrapped:+.4f}")
        base_pos, base_yaw = self.base_pose()
        say(f"base link: position {fmt(base_pos)}, yaw {np.degrees(base_yaw):+.3f} deg (must stay at its start pose)")

    def go_to_default_pose(self) -> None:
        """Teleport to home with the gripper open - the same start as the other scripts."""
        self.franka.set_joint_positions(HOME_ARM_JOINTS, joint_indices=self.arm_dof_indices)
        self.franka.gripper.set_joint_positions(self.franka.gripper.joint_opened_positions)

    def on_timeline_restart(self) -> None:
        """Called on the first physics step after Stop/Play in the GUI."""
        self.world.reset()
        self.fix_link_inertias()
        self.rmp.reset()
        self.pick_place.reset()
        self.go_to_default_pose()

    def fix_link_inertias(self) -> None:
        """Give links with a degenerate inertia tensor a plausible one.

        This stage authors physics:diagonalInertia ~ 1e-6 kg m^2 on every Franka
        link (2-3 kg bodies should be ~1e-2). With the mass matrix that badly
        conditioned, PhysX's articulation solver cannot make joint 1 - the only
        joint with no gravity load - follow its drive: commanded to hold, the joint
        drifts at ~0.5 rad/s, slides through its limits and eventually spins whole
        turns (the "joint wrap" seen in the 2026-09-08 session). Realistic
        inertias make the drive behave like the other six joints.
        """
        if LAUNCH.no_fix_inertia:
            return
        view = self.franka._articulation_view
        masses = np.asarray(view.get_body_masses(), dtype=float)[0]
        inertias = np.asarray(view.get_body_inertias(), dtype=float)[0].reshape(len(masses), 3, 3)
        fixed = []
        for i, (name, m) in enumerate(zip(view.body_names, masses)):
            if np.trace(inertias[i]) < m * INERTIA_MIN_GYRATION ** 2:
                inertias[i] = np.eye(3) * m * INERTIA_GYRATION ** 2
                fixed.append(name)
        if fixed:
            view.set_body_inertias(inertias.reshape(1, len(masses), 9), indices=np.array([0]))
            say(f"repaired degenerate inertia on {len(fixed)} link(s): {', '.join(fixed)} "
                f"(now mass x {INERTIA_GYRATION:g} m^2; --no-fix-inertia keeps the stage values)")

    def hold(self) -> None:
        """Freeze the arm where it is.

        RMPflow (and the pick-place controller on top of it) commands BOTH position
        and velocity targets on the joint drives. An action that omits velocities
        leaves the last velocity target in place, so a position-only command
        afterwards (e.g. `home`) settles with a steady offset. Zero the velocity
        targets and pin the position targets to the current pose whenever a task
        ends or is aborted.
        """
        current = self.franka.get_joint_positions(joint_indices=self.arm_dof_indices)
        self.franka.apply_action(
            ArticulationAction(
                joint_positions=current,
                joint_velocities=np.zeros(len(self.arm_dof_indices)),
                joint_indices=self.arm_dof_indices,
            )
        )

    def ee_pose(self):
        """Position and 3x3 rotation of the frame RMPflow drives (between the fingertips)."""
        joints = self.rmp.get_articulation_motion_policy().get_active_joints_subset().get_joint_positions()
        pos, rot = self.rmp.get_motion_policy().get_end_effector_pose(joints)
        return np.asarray(pos, dtype=float), np.asarray(rot, dtype=float)

    def physics_dt(self) -> float:
        return float(self.world.get_physics_dt())

    def print_pose(self) -> None:
        pos, rot = self.ee_pose()
        quat = rot_matrix_to_quat(rot)
        euler = quat_to_euler_angles(quat, degrees=True)
        say(f"end effector position (m)        : {fmt(pos)}")
        say(f"end effector roll/pitch/yaw (deg): {fmt(euler)}")
        say(f"end effector quaternion (w x y z): {fmt(quat)}")
        say(f"finger joints (m)                : {fmt(self.franka.gripper.get_joint_positions())}")


# ----------------------------------------------------------------------------
# Tasks: one per command. step() is called once per simulation frame and
# returns True when the task is finished.
# ----------------------------------------------------------------------------
class Task:
    label = "task"

    def start(self, arm: Arm) -> None:
        pass

    def step(self, arm: Arm) -> bool:
        raise NotImplementedError

    def cancel(self, arm: Arm) -> None:
        pass


class MoveTask(Task):
    def __init__(self, position, orientation) -> None:
        self.position = np.asarray(position, dtype=float)
        self.orientation = None if orientation is None else np.asarray(orientation, dtype=float)
        orient_txt = "keep orientation" if orientation is None else f"quat {fmt(self.orientation)}"
        self.label = f"move to {fmt(self.position)} ({orient_txt})"

    def start(self, arm: Arm) -> None:
        if self.orientation is None:
            _, rot = arm.ee_pose()
            self.orientation = rot_matrix_to_quat(rot)
        self.target_rot = quat_to_rot_matrix(self.orientation)
        self.settled = 0
        self.steps = 0
        self.max_steps = int(MOVE_TIMEOUT_S / arm.physics_dt())
        self.stall_steps = int(MOVE_STALL_S / arm.physics_dt())
        self.best_pos_err = np.inf
        self.best_rot_err = np.inf
        self.last_progress = 0
        self.dt = arm.physics_dt()
        arm.rmp.reset()

    def step(self, arm: Arm) -> bool:
        action = arm.rmp.forward(
            target_end_effector_position=self.position,
            target_end_effector_orientation=self.orientation,
        )
        arm.franka.apply_action(action)
        self.steps += 1

        pos, rot = arm.ee_pose()
        pos_err = float(np.linalg.norm(pos - self.position))
        rot_err = rotation_error_deg(self.target_rot, rot)
        if pos_err < MOVE_POS_TOL and rot_err < MOVE_ROT_TOL_DEG:
            self.settled += 1
        else:
            self.settled = 0

        if pos_err < self.best_pos_err - MOVE_PROGRESS_POS or rot_err < self.best_rot_err - MOVE_PROGRESS_ROT_DEG:
            self.last_progress = self.steps
        self.best_pos_err = min(self.best_pos_err, pos_err)
        self.best_rot_err = min(self.best_rot_err, rot_err)

        elapsed = self.steps * self.dt
        if self.settled >= MOVE_SETTLE_STEPS:
            say(f"arrived: position error {pos_err * 1000:.1f} mm, rotation error {rot_err:.2f} deg "
                f"({elapsed:.1f} s)")
            return True
        if self.steps - self.last_progress >= self.stall_steps and self.settled == 0:
            if pos_err < MOVE_NEAR_POS_TOL and rot_err < MOVE_NEAR_ROT_TOL_DEG:
                say(f"settled short of target: position error {pos_err * 1000:.1f} mm, "
                    f"rotation error {rot_err:.2f} deg (no progress for {MOVE_STALL_S:g} s, {elapsed:.1f} s)")
            else:
                say(f"move stalled after {elapsed:.1f} s with position error {pos_err * 1000:.1f} mm "
                    f"and rotation error {rot_err:.1f} deg (target probably out of reach or blocked).")
            return True
        if self.steps >= self.max_steps:
            say(
                f"move timed out after {MOVE_TIMEOUT_S:.0f} s with position error "
                f"{pos_err * 1000:.1f} mm and rotation error {rot_err:.1f} deg "
                "(target probably out of reach or blocked)."
            )
            return True
        return False


class GripperTask(Task):
    def __init__(self, action: str) -> None:
        self.action = action  # "open" or "close"
        self.label = f"{action} gripper"

    def start(self, arm: Arm) -> None:
        self.steps = 0

    def step(self, arm: Arm) -> bool:
        # Re-apply every frame like the PickPlaceController does, so a closing
        # gripper keeps squeezing whatever it meets.
        arm.franka.apply_action(arm.franka.gripper.forward(action=self.action))
        self.steps += 1
        if self.steps >= GRIPPER_STEPS:
            say(f"gripper {self.action}: fingers at {fmt(arm.franka.gripper.get_joint_positions())} m")
            return True
        return False


class CameraTask(Task):
    """Waits for a fresh frame from the hand camera, then hands it to save().

    With `immediate` set the task does not go through the command queue: the main
    loop steps it alongside whatever is running, and it takes the next frame as is.
    """

    needs_depth = False
    immediate = False

    def start(self, arm: Arm) -> None:
        self.steps = 0

    def step(self, arm: Arm) -> bool:
        if arm.camera is None:
            say("no camera in the stage; nothing captured.")
            return True
        self.steps += 1
        if not self.immediate and self.steps < SNAP_SETTLE_STEPS:
            return False  # let the renderer catch up with the settled arm pose

        rgb = camera_rgb(arm.camera)
        depth = to_numpy(arm.camera.get_depth()) if self.needs_depth else None
        have_depth = (not self.needs_depth) or (depth is not None and depth.size > 0)
        if rgb is None or not have_depth:
            if self.steps >= SNAP_MAX_WAIT_STEPS:
                say("camera produced no frame; nothing captured.")
                return True
            return False

        if self.immediate:
            # The arm may be mid-move: say where the gripper was for this frame.
            say(f"captured with the end effector at {fmt(arm.ee_pose()[0])}")
        self.save(arm, rgb, depth)
        return True

    def save(self, arm: Arm, rgb: np.ndarray, depth) -> None:
        raise NotImplementedError


class SnapTask(CameraTask):
    def __init__(self, path=None, with_depth=False, immediate=False) -> None:
        self.path = path
        self.needs_depth = with_depth
        self.immediate = immediate
        self.label = f"snap -> {path or 'captures/'}"

    def start(self, arm: Arm) -> None:
        super().start(arm)
        if self.path is None:
            stamp = datetime.now().strftime("capture_%Y%m%d_%H%M%S")
            self.path = os.path.join(CAPTURE_DIR, stamp + ".png")

    def save(self, arm: Arm, rgb: np.ndarray, depth) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        Image.fromarray(rgb).save(self.path)
        say(f"saved {self.path} ({rgb.shape[1]}x{rgb.shape[0]})")
        if self.needs_depth:
            depth_path = os.path.splitext(self.path)[0] + "_depth.npy"
            np.save(depth_path, depth)
            say(f"saved {depth_path} (float32 metres, shape {depth.shape})")
        arm.record_capture(rgb, self.path)


class PointsTask(CameraTask):
    """World XYZ for every pixel: <base>.npy (H, W, 3) + <base>.png, optionally <base>.ply."""

    needs_depth = True

    def __init__(self, base=None, with_ply=False, immediate=False) -> None:
        self.base = base
        self.with_ply = with_ply
        self.immediate = immediate
        self.label = f"points -> {base or 'captures/'}"

    def start(self, arm: Arm) -> None:
        super().start(arm)
        if self.base is None:
            self.base = os.path.join(CAPTURE_DIR, datetime.now().strftime("points_%Y%m%d_%H%M%S"))
        elif os.path.splitext(self.base)[1].lower() in (".npy", ".png", ".ply"):
            self.base = os.path.splitext(self.base)[0]

    def save(self, arm: Arm, rgb: np.ndarray, depth) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.base)), exist_ok=True)
        xyz = unproject_depth(arm.camera, depth)

        Image.fromarray(rgb).save(self.base + ".png")
        np.save(self.base + ".npy", xyz)

        valid = np.isfinite(xyz).all(axis=-1)
        n_valid = int(valid.sum())
        say(f"saved {self.base}.npy: world XYZ per pixel, shape {xyz.shape}, "
            f"{n_valid} of {valid.size} pixels have depth")
        say(f"saved {self.base}.png (matching colour image; image[row, col] <-> xyz[row, col])")
        if n_valid:
            lo = np.nanmin(xyz.reshape(-1, 3), axis=0)
            hi = np.nanmax(xyz.reshape(-1, 3), axis=0)
            say(f"XYZ range: min {fmt(lo)}  max {fmt(hi)}")
        if self.with_ply:
            n = write_ply(self.base + ".ply", xyz, rgb)
            say(f"saved {self.base}.ply ({n} coloured points)")
        arm.record_capture(rgb, self.base + ".png", xyz)


class WatchTask(Task):
    """Rewrites one PNG with the latest hand-camera frame every `period` simulated seconds.

    Never queued and never finishes: the main loop steps it next to the running
    command until `watch off`. Each frame is written to a temporary file and
    renamed over the target, so a reader never sees a half-written image.
    """

    def __init__(self, period: float, path: str) -> None:
        self.period = period
        self.path = path
        self.label = f"watch -> {path} every {period:g} s"

    def start(self, arm: Arm) -> None:
        self.every = max(1, int(round(self.period / arm.physics_dt())))
        self.countdown = 0
        self.frames = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def step(self, arm: Arm) -> bool:
        self.countdown -= 1
        if self.countdown > 0:
            return False
        rgb = camera_rgb(arm.camera)
        if rgb is None or not rgb.any():
            # No frame yet, or the all-black one the renderer hands out right after a
            # reset: keep the last good image and try again next step.
            return False
        tmp =self.path + ".tmp"
        try:
            # Fast compression: this runs inside the simulation loop.
            Image.fromarray(rgb).save(tmp, format="PNG", compress_level=1)
            os.replace(tmp, self.path)
            self.frames += 1
            arm.record_capture(rgb)   # no path: one log line per second would drown /log
        except OSError:
            pass  # target is open in a viewer right now; the next frame replaces it
        self.countdown = self.every
        return False


class WatchTask(Task):
    """Rewrites one PNG with the latest hand-camera frame every `period` simulated seconds.

    Never queued and never finishes: the main loop steps it next to the running
    command until `watch off`. Each frame is written to a temporary file and
    renamed over the target, so a reader never sees a half-written image.
    """

    def __init__(self, period: float, path: str) -> None:
        self.period = period
        self.path = path
        self.label = f"watch -> {path} every {period:g} s"

    def start(self, arm: Arm) -> None:
        self.every = max(1, int(round(self.period / arm.physics_dt())))
        self.countdown = 0
        self.frames = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def step(self, arm: Arm) -> bool:
        self.countdown -= 1
        if self.countdown > 0:
            return False
        rgb = camera_rgb(arm.camera)
        if rgb is None or not rgb.any():
            # No frame yet, or the all-black one the renderer hands out right after a
            # reset: keep the last good image and try again next step.
            return False
        tmp =self.path + ".tmp"
        try:
            # Fast compression: this runs inside the simulation loop.
            Image.fromarray(rgb).save(tmp, format="PNG", compress_level=1)
            os.replace(tmp, self.path)
            self.frames += 1
        except OSError:
            pass  # target is open in a viewer right now; the next frame replaces it
        self.countdown = self.every
        return False


class PickPlaceTask(Task):
    def __init__(self, pick, place, yaw_deg: float) -> None:
        self.pick = np.asarray(pick, dtype=float)
        self.place = np.asarray(place, dtype=float)
        self.yaw_deg = yaw_deg
        # Top-down grasp rotated about vertical so the fingers close across the object.
        self.orientation = euler_angles_to_quat(np.array([0.0, np.pi, np.deg2rad(yaw_deg)]))
        self.label = f"pick {fmt(self.pick)} -> place {fmt(self.place)} (yaw {yaw_deg:g} deg)"

    def start(self, arm: Arm) -> None:
        arm.pick_place.reset()
        self.last_phase = -1

    def step(self, arm: Arm) -> bool:
        actions = arm.pick_place.forward(
            picking_position=self.pick,
            placing_position=self.place,
            current_joint_positions=arm.franka.get_joint_positions(),
            end_effector_orientation=self.orientation,
        )
        arm.franka.apply_action(actions)

        phase = arm.pick_place.get_current_event()
        if phase != self.last_phase and phase < len(PICK_PLACE_PHASES):
            say(f"pick-place phase {phase + 1}/{len(PICK_PLACE_PHASES)}: {PICK_PLACE_PHASES[phase]}")
            self.last_phase = phase

        if arm.pick_place.is_done():
            say("pick-place finished.")
            return True
        return False

    def cancel(self, arm: Arm) -> None:
        arm.pick_place.reset()


class JointMoveTask(Task):
    """Smooth joint-space move of the 7 arm joints to `target` (radians).

    The targets are interpolated from the current configuration instead of being
    set in one jump: the Franka's drives are stiff enough that a jump whips the
    arm across the workspace. Arrival requires every joint to be within
    HOME_JOINT_TOL of its target AND slower than HOME_REST_SPEED for a few frames,
    so a joint swinging through its target does not count as arrived.
    """

    def __init__(self, target, label: str) -> None:
        self.target = np.asarray(target, dtype=float)
        self.label = label

    def start(self, arm: Arm) -> None:
        self.q0 = np.asarray(arm.franka.get_joint_positions(joint_indices=arm.arm_dof_indices), dtype=float)
        self.goal = self.target.copy()
        travel = float(np.max(np.abs(self.q0 - self.goal)))
        duration = max(HOME_MIN_DURATION_S, travel / HOME_JOINT_SPEED)
        self.ramp_steps = max(1, int(duration / arm.physics_dt()))
        self.max_steps = self.ramp_steps + int(MOVE_TIMEOUT_S / arm.physics_dt())
        self.steps = 0
        self.settled = 0
        self.zero_vel = np.zeros(len(arm.arm_dof_indices))

    def step(self, arm: Arm) -> bool:
        self.steps += 1
        s = min(1.0, self.steps / self.ramp_steps)
        alpha = 0.5 * (1.0 - np.cos(np.pi * s))  # ease in / ease out
        target = (1.0 - alpha) * self.q0 + alpha * self.goal
        arm.franka.apply_action(
            ArticulationAction(
                joint_positions=target,
                joint_velocities=self.zero_vel,
                joint_indices=arm.arm_dof_indices,
            )
        )
        joints = arm.franka.get_joint_positions(joint_indices=arm.arm_dof_indices)
        speeds = np.abs(arm.franka.get_joint_velocities(joint_indices=arm.arm_dof_indices))
        errors = np.abs(joints - self.goal)
        # "Arrived" means at the target AND at rest for a few frames in a row: a
        # joint swinging through the target angle must not count.
        if s >= 1.0 and float(np.max(errors)) < HOME_JOINT_TOL and float(np.max(speeds)) < HOME_REST_SPEED:
            self.settled += 1
        else:
            self.settled = 0
        if self.settled >= MOVE_SETTLE_STEPS:
            say(f"{self.done_message()}")
            return True
        if self.steps >= self.max_steps:
            say(f"{self.label} timed out; joint errors (rad): {fmt(errors)}; joint speeds (rad/s): {fmt(speeds)}")
            return True
        return False

    def done_message(self) -> str:
        return f"{self.label}: joints at {fmt(self.goal)}"


class HomeTask(JointMoveTask):
    label = "home"

    def __init__(self) -> None:
        super().__init__(HOME_ARM_JOINTS, "home")

    def done_message(self) -> str:
        return "at home configuration."


class JogTask(JointMoveTask):
    """Move ONE arm joint to an angle (degrees); the other joints hold where they are."""

    def __init__(self, joint_number: int, angle_deg: float) -> None:
        self.joint_number = joint_number
        self.angle = np.deg2rad(angle_deg)
        super().__init__(HOME_ARM_JOINTS, f"jog joint {joint_number} to {angle_deg:g} deg")

    def start(self, arm: Arm) -> None:
        current = np.asarray(arm.franka.get_joint_positions(joint_indices=arm.arm_dof_indices), dtype=float)
        current[self.joint_number - 1] = self.angle
        self.target = current
        super().start(arm)


class WaitTask(Task):
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.label = f"wait {seconds:g} s"

    def start(self, arm: Arm) -> None:
        self.remaining = int(round(self.seconds / arm.physics_dt()))

    def step(self, arm: Arm) -> bool:
        self.remaining -= 1
        return self.remaining <= 0


class PoseTask(Task):
    """Queued so that scripts print the pose at the right point in the sequence."""
    label = "pose"

    def step(self, arm: Arm) -> bool:
        arm.print_pose()
        return True


class JointsTask(Task):
    """Print joint readings, limits, and joint 1's reading vs its physical angle."""
    label = "joints"

    def step(self, arm: Arm) -> bool:
        arm.print_joints()
        return True


class ResetTask(Task):
    """Reset the physics scene and controllers without relaunching Isaac Sim.

    Same as Stop/Play in the GUI: everything in the stage goes back to its
    default state (objects included), the arm is teleported home, the gripper
    opened and both controllers reset.
    """
    label = "reset"

    def step(self, arm: Arm) -> bool:
        arm.on_timeline_restart()
        say("scene reset: objects restored, arm teleported home, gripper open.")
        return True


class MarkTask(Task):
    """`mark TEXT`: a note stamped when it reaches the front of the queue."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.label = f"mark {text!r}"

    def step(self, arm: Arm) -> bool:
        if arm.recorder is None:
            say(f"mark (not recording): {self.text}")
        else:
            arm.recorder.note(arm.sim_time(), self.text, "mark")
            say(f"mark: {self.text}")
        return True


class QuitTask(Task):
    """Ordered quit: runs after everything queued before it (used for -c / piped input)."""
    label = "quit"

    def step(self, arm: Arm) -> bool:
        return True


# ----------------------------------------------------------------------------
# Command parsing
# ----------------------------------------------------------------------------
HELP = """Commands:
  move X Y Z                         top-down orientation
  move X Y Z ROLL PITCH YAW          orientation in degrees (straight down = 0 180 0)
  move X Y Z q QW QX QY QZ           orientation as a quaternion (w x y z)
  move X Y Z keep                    keep the current orientation
  open | close                       gripper
  snap [PATH] [depth] [now]          save a camera image (PNG), optionally depth too (.npy);
                                     now = immediately, while the running command continues
  points [PATH] [ply] [now]          save world XYZ of every pixel: PATH.npy (H,W,3) + PATH.png,
                                     add ply for a coloured point cloud (PATH.ply); now as above
  watch [SECONDS] [PATH] | watch off immediately: refresh PATH (default captures/live.png) with
                                     the camera image every SECONDS (default 1) in the background
  pick X Y Z [to PX PY PZ] [yaw DEG] run the pick-and-place controller
  pose                               print the end-effector pose (runs in queue order)
  joints                             print joint readings, limits, and joint 1 reading vs physical
  jog N DEG                          move arm joint N (1..7) to DEG degrees, other joints hold
  home                               return to the home joint configuration
  reset                              reset the scene (objects, arm, controllers) without relaunching
  wait SECONDS                       pause the queue
  status                             immediately: what is running, queue length, live pose
  stop                               abort the running action, drop queued ones
  record [PATH] | record off         immediately: start / end recording the session to an MCAP
                                     file for Foxglove (default runs/foxglove/arm_<timestamp>.mcap)
  note TEXT...                       immediately: write TEXT into the recording (/notes), stamped now
  mark TEXT...                       same, but queued: stamped when its turn comes in the sequence
  think TEXT...                      immediately: overall plan / problem solving -> /reasoning/strategy
  motor TEXT...                      immediately: motor-control reasoning -> /reasoning/motor
  help | quit"""


def _floats(tokens, n, what):
    if len(tokens) < n:
        raise ValueError(f"expected {n} numbers for {what}")
    try:
        return [float(t) for t in tokens[:n]]
    except ValueError:
        raise ValueError(f"expected {n} numbers for {what}, got {' '.join(tokens[:n])!r}")


def parse_move(args):
    position = _floats(args, 3, "position")
    rest = args[3:]
    if not rest:
        return MoveTask(position, TOP_DOWN_QUAT)
    if len(rest) == 1 and rest[0].lower() == "keep":
        return MoveTask(position, None)
    if len(rest) == 3:
        rpy = _floats(rest, 3, "roll/pitch/yaw")
        return MoveTask(position, euler_angles_to_quat(np.array(rpy), degrees=True))
    if len(rest) == 5 and rest[0].lower() == "q":
        quat = np.array(_floats(rest[1:], 4, "quaternion"))
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            raise ValueError("quaternion has zero length")
        return MoveTask(position, quat / norm)
    raise ValueError(
        "usage: move X Y Z | move X Y Z ROLL PITCH YAW | move X Y Z q QW QX QY QZ | move X Y Z keep"
    )


def parse_pick(args):
    pick = _floats(args, 3, "pick position")
    place = BASKET_RELEASE_POSITION
    yaw = 0.0
    rest = args[3:]
    while rest:
        key = rest[0].lower()
        if key == "to":
            place = _floats(rest[1:], 3, "place position")
            rest = rest[4:]
        elif key == "yaw":
            yaw = _floats(rest[1:], 1, "yaw")[0]
            rest = rest[2:]
        else:
            raise ValueError("usage: pick X Y Z [to PX PY PZ] [yaw DEG]")
    return PickPlaceTask(pick, place, yaw)


def parse_snap(args):
    path = None
    with_depth = False
    immediate = False
    for tok in args:
        if tok.lower() == "depth":
            with_depth = True
        elif tok.lower() == "now":
            immediate = True
        elif path is None:
            path = tok
        else:
            raise ValueError("usage: snap [PATH] [depth] [now]")
    return SnapTask(path, with_depth, immediate)


def parse_points(args):
    base = None
    with_ply = False
    immediate = False
    for tok in args:
        if tok.lower() == "ply":
            with_ply = True
        elif tok.lower() == "now":
            immediate = True
        elif base is None:
            base = tok
        else:
            raise ValueError("usage: points [PATH] [ply] [now]")
    return PointsTask(base, with_ply, immediate)


def parse_watch(args):
    """`watch [SECONDS] [PATH]` -> WatchTask, `watch off` -> None."""
    usage = "usage: watch [SECONDS] [PATH] | watch off"
    if len(args) == 1 and args[0].lower() in ("off", "stop"):
        return None
    period = None
    path = None
    for tok in args:
        try:
            value = float(tok)
        except ValueError:
            value = None
        if value is not None and period is None:
            period = value
        elif value is None and path is None:
            path = tok
        else:
            raise ValueError(usage)
    if period is not None and not period > 0:
        raise ValueError("watch: SECONDS must be greater than 0")
    return WatchTask(WATCH_DEFAULT_PERIOD_S if period is None else period, path or WATCH_DEFAULT_PATH)


def build_task(argv):
    """Turn a tokenised command into a Task, or raise ValueError with usage."""
    cmd, args = argv[0].lower(), argv[1:]
    if cmd == "move":
        return parse_move(args)
    if cmd in ("open", "close"):
        if args:
            raise ValueError(f"usage: {cmd}")
        return GripperTask(cmd)
    if cmd in ("snap", "photo", "picture", "capture"):
        return parse_snap(args)
    if cmd in ("points", "pointcloud", "cloud"):
        return parse_points(args)
    if cmd == "pick":
        return parse_pick(args)
    if cmd == "home":
        return HomeTask()
    if cmd == "pose":
        return PoseTask()
    if cmd == "joints":
        return JointsTask()
    if cmd == "jog":
        n, deg = _floats(args, 2, "joint number and angle")
        if n != int(n) or not 1 <= int(n) <= 7:
            raise ValueError("usage: jog N DEG   (N = 1..7)")
        return JogTask(int(n), deg)
    if cmd == "reset":
        return ResetTask()
    if cmd == "wait":
        return WaitTask(_floats(args, 1, "seconds")[0])
    if cmd == "mark":
        if not args:
            raise ValueError("usage: mark TEXT...")
        return MarkTask(" ".join(args))
    raise ValueError(f"unknown command {cmd!r} (type help)")


def stdin_reader(cmd_queue: "queue.Queue[str]") -> None:
    """Runs in a background thread: the sim loop must never block on input()."""
    while True:
        try:
            line = input("arm> ")
        except EOFError:
            cmd_queue.put(EOF_SENTINEL)
            return
        cmd_queue.put(line)


# ----------------------------------------------------------------------------
def main() -> None:
    arm = Arm(LAUNCH.usd)

    cmd_queue: "queue.Queue[str]" = queue.Queue()
    if LAUNCH.commands:
        for part in LAUNCH.commands.split(";"):
            if part.strip():
                cmd_queue.put(part.strip())
        if LAUNCH.quit_when_done:
            cmd_queue.put(EOF_SENTINEL)

    if not (LAUNCH.commands and LAUNCH.quit_when_done):
        threading.Thread(target=stdin_reader, args=(cmd_queue,), daemon=True).start()

    if LAUNCH.record is not None:
        arm.start_recording(LAUNCH.record)

    say(f"ready. {len(arm.franka.dof_names)} DOFs, camera {'on' if arm.camera else 'off'}. Type help for commands.")

    pending: list = []
    current = None
    captures: list = []   # `snap now` / `points now`: stepped alongside `current`
    watch = None          # the running WatchTask, if any
    quit_requested = False

    while simulation_app.is_running() and not quit_requested:
        arm.world.step(render=True)

        # Drain the console. Immediate commands run now; the rest queue up.
        while True:
            try:
                line = cmd_queue.get_nowait()
            except queue.Empty:
                break
            if line == EOF_SENTINEL:
                pending.append(QuitTask())
                continue
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                argv = shlex.split(line)
            except ValueError as exc:
                say(f"could not parse {line!r}: {exc}")
                continue
            cmd = argv[0].lower()
            if arm.recorder is not None and cmd not in COMMENTARY_COMMANDS:
                arm.recorder.command(arm.sim_time(), line)
            if cmd in ("quit", "exit"):
                quit_requested = True
            elif cmd == "stop":
                if current is not None:
                    current.cancel(arm)
                    arm.hold()
                    say(f"aborted: {current.label}")
                    if arm.recorder is not None:
                        arm.recorder.task(arm.sim_time(), "aborted", current.label)
                    current = None
                if pending:
                    say(f"dropped {len(pending)} queued command(s)")
                    pending.clear()
            elif cmd == "status":
                running = current.label if current is not None else "idle"
                watching = "off" if watch is None else f"{watch.path} every {watch.period:g} s ({watch.frames} frames)"
                recording = "off" if arm.recorder is None else arm.recorder.path
                say(f"running: {running}; queued: {len(pending)}; watch: {watching}; recording: {recording}")
                arm.print_pose()
            elif cmd == "note":
                text = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                if not text:
                    say("error: usage: note TEXT...")
                elif arm.recorder is None:
                    say(f"note (not recording): {text}")
                else:
                    arm.recorder.note(arm.sim_time(), text, "note")
                    say(f"note: {text}")
            elif cmd in REASONING_COMMANDS:
                kind = REASONING_COMMANDS[cmd]
                text = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
                if not text:
                    say(f"error: usage: {cmd} TEXT...")
                    continue
                if arm.recorder is not None:
                    arm.recorder.reasoning(arm.sim_time(), kind, text)
                # Echo to the terminal only: copying it to /log would mix it back in.
                print(f"[arm] {kind}{'' if arm.recorder else ' (not recording)'}: {text}", flush=True)
            elif cmd == "record":
                if len(argv) == 2 and argv[1].lower() in ("off", "stop"):
                    if arm.recorder is None:
                        say("recording is not running.")
                    arm.stop_recording()
                elif len(argv) <= 2:
                    arm.start_recording(argv[1] if len(argv) == 2 else "")
                else:
                    say("error: usage: record [PATH] | record off")
            elif cmd == "watch":
                try:
                    new_watch = parse_watch(argv[1:])
                except ValueError as exc:
                    say(f"error: {exc}")
                    continue
                if new_watch is not None and arm.camera is None:
                    say("no camera in the stage; watch is disabled.")
                    continue
                if watch is not None:
                    say(f"watch off: {watch.frames} frame(s) written to {watch.path}")
                elif new_watch is None:
                    say("watch is not running.")
                watch = new_watch
                if watch is not None:
                    watch.start(arm)
                    say(f"watch on: {watch.path} refreshed every {watch.period:g} s (`watch off` ends it)")
            elif cmd in ("help", "?"):
                print(HELP, flush=True)
            else:
                try:
                    task = build_task(argv)
                except ValueError as exc:
                    say(f"error: {exc}")
                    continue
                if getattr(task, "immediate", False):
                    # Not "> label": that prefix marks the start of a queued command.
                    say(f"now: {task.label}")
                    task.start(arm)
                    captures.append(task)
                else:
                    pending.append(task)

        if quit_requested:
            break
        if not arm.world.is_playing():
            continue

        # First physics step after Stop/Play in the GUI: start over.
        if arm.world.current_time_step_index == 0:
            arm.on_timeline_restart()
            if current is not None:
                say(f"timeline restarted; aborted {current.label}")
                if arm.recorder is not None:
                    arm.recorder.task(arm.sim_time(), "aborted", current.label)
                current = None
            pending.clear()

        # An ordered quit waits for `now` captures that are still after their frame.
        if current is None and pending and not (isinstance(pending[0], QuitTask) and captures):
            current = pending.pop(0)
            if isinstance(current, QuitTask):
                break
            say(f"> {current.label}")
            if arm.recorder is not None:
                arm.recorder.task(arm.sim_time(), "start", current.label)
            current.start(arm)

        if current is not None and current.step(arm):
            arm.hold()
            if arm.recorder is not None:
                arm.recorder.task(arm.sim_time(), "done", current.label)
            current = None

        captures = [task for task in captures if not task.step(arm)]
        if watch is not None:
            watch.step(arm)

        label = current.label if current is not None else "idle"
        arm.trace(label)
        arm.record_step(label)

    if arm.trace_file is not None:
        arm.trace_file.close()
    arm.stop_recording()
    if watch is not None:
        say(f"watch off: {watch.frames} frame(s) written to {watch.path}")
    say("closing.")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        simulation_app.close()
