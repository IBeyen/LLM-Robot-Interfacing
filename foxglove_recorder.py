r"""
foxglove_recorder.py - record an arm_cli.py session to an MCAP file for Foxglove.

The file opens directly in Foxglove (File > Open local file, or drag it onto the
window). Everything is JSON-encoded with Foxglove's well-known schemas, so the
3D, Image, Plot, Log and Raw Messages panels all work without a ROS install.

Topics written:
    /joint_states          JSON   every physics step: name[], position[], velocity[],
                                  target_position[], target_velocity[] for all 9 DOFs
                                  (7 arm joints + 2 fingers). Plot e.g. `/joint_states.position[0]`.
    /ee_pose               foxglove.PoseInFrame     end-effector (between the fingertips), world frame
    /tf                    foxglove.FrameTransforms world -> every Franka link, hand camera (optical frame)
    /task                  JSON   {event: "start"|"done"|"aborted", label} when a queued command changes
    /commands              foxglove.Log  every arm_cli command typed at the arm> prompt / fed on stdin
                                  (commentary - think / motor / note / mark - is left out)
    /reasoning/strategy    foxglove.Log  `think` text: overarching ideas, plans, problem solving
    /reasoning/motor       foxglove.Log  `motor` text: motor-control ideas (grasp heights, waypoints,
                                  orientations, why a motion went wrong)
    /log                   foxglove.Log  every "[arm] ..." line the CLI prints
    /hand_camera/image     foxglove.CompressedImage  JPEG from the hand camera, throttled (see image_period)
    /hand_camera/calibration  foxglove.CameraCalibration  intrinsics, once per start and on resolution change
    /hand_camera/retrieved foxglove.CompressedImage  the last image arm_cli handed out: every `snap`,
                                  `points` and `watch` frame, full quality, at the moment it was saved
    /notes                 foxglove.Log  free text from the `note` (immediate) / `mark` (queued) commands
    /hand_camera/points    foxglove.PointCloud  the world-frame cloud of every `points` capture
    /viewport/image        foxglove.CompressedImage  the Isaac Sim perspective viewport, same throttle
    /sim                   JSON   every step: sim_time (s), wall_time (wall s since the recording
                                  started), wall_clock (real date/time), rtf (sim/wall speed over
                                  the last ~1 s), rtf_avg (since the recording started), step, physics_dt

Time: every message carries one clock, `epoch + simulated time`, both as its MCAP
log time and inside its `timestamp` field. The epoch is the wall-clock moment the
recording started, so the Foxglove timeline shows a real date but advances with
the simulation (renders that stall the sim do not stretch the timeline). Wall
time is on /sim: plot `/sim.wall_time` and `/sim.sim_time` against the timeline,
or `/sim.rtf` for how fast the sim ran at each moment.

This module must stay importable without Isaac Sim: arm_cli.py hands it plain
numpy arrays, and test_foxglove_recorder.py exercises it on its own.
"""

import base64
import io
import json
import os
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from mcap.writer import CompressionType, Writer
from PIL import Image

WORLD_FRAME = "world"
CAMERA_FRAME = "hand_camera"

# foxglove.PointCloud NumericType / foxglove.Log LogLevel enum values.
PC_UINT8, PC_UINT32, PC_FLOAT32 = 1, 5, 7
LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR = 1, 2, 3, 4

# /sim.rtf is sim time / wall time over roughly this much recent wall time.
RTF_WINDOW_S = 1.0

# Point cloud layout: x y z float32 + packed rgba uint32 = 16 bytes per point.
POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "<u4")])
POINT_FIELDS = [
    {"name": "x", "offset": 0, "type": PC_FLOAT32},
    {"name": "y", "offset": 4, "type": PC_FLOAT32},
    {"name": "z", "offset": 8, "type": PC_FLOAT32},
    {"name": "rgba", "offset": 12, "type": PC_UINT32},
]


# ----------------------------------------------------------------------------
# JSON schemas (the subset of @foxglove/schemas we emit). Foxglove picks the
# panel support from the schema NAME; the schema body documents the fields.
# ----------------------------------------------------------------------------
def _obj(props: dict, **extra) -> dict:
    return {"type": "object", "properties": props, **extra}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


_NUM = {"type": "number"}
_INT = {"type": "integer"}
_STR = {"type": "string"}
_BYTES = {"type": "string", "contentEncoding": "base64"}
_TIME = _obj({"sec": _INT, "nsec": _INT}, title="time")
_VEC3 = _obj({"x": _NUM, "y": _NUM, "z": _NUM}, title="foxglove.Vector3")
_QUAT = _obj({"x": _NUM, "y": _NUM, "z": _NUM, "w": _NUM}, title="foxglove.Quaternion")
_POSE = _obj({"position": _VEC3, "orientation": _QUAT}, title="foxglove.Pose")

SCHEMAS: Dict[str, dict] = {
    "foxglove.CompressedImage": _obj(
        {"timestamp": _TIME, "frame_id": _STR, "data": _BYTES, "format": _STR},
        title="foxglove.CompressedImage",
    ),
    "foxglove.CameraCalibration": _obj(
        {
            "timestamp": _TIME, "frame_id": _STR, "width": _INT, "height": _INT,
            "distortion_model": _STR, "D": _arr(_NUM), "K": _arr(_NUM), "R": _arr(_NUM), "P": _arr(_NUM),
        },
        title="foxglove.CameraCalibration",
    ),
    "foxglove.PointCloud": _obj(
        {
            "timestamp": _TIME, "frame_id": _STR, "pose": _POSE, "point_stride": _INT,
            "fields": _arr(_obj({"name": _STR, "offset": _INT, "type": _INT}, title="foxglove.PackedElementField")),
            "data": _BYTES,
        },
        title="foxglove.PointCloud",
    ),
    "foxglove.FrameTransforms": _obj(
        {
            "transforms": _arr(_obj(
                {
                    "timestamp": _TIME, "parent_frame_id": _STR, "child_frame_id": _STR,
                    "translation": _VEC3, "rotation": _QUAT,
                },
                title="foxglove.FrameTransform",
            ))
        },
        title="foxglove.FrameTransforms",
    ),
    "foxglove.PoseInFrame": _obj({"timestamp": _TIME, "frame_id": _STR, "pose": _POSE}, title="foxglove.PoseInFrame"),
    "foxglove.Log": _obj(
        {"timestamp": _TIME, "level": _INT, "message": _STR, "name": _STR, "file": _STR, "line": _INT},
        title="foxglove.Log",
    ),
    "arm_cli.JointStates": _obj(
        {
            "timestamp": _TIME, "name": _arr(_STR), "position": _arr(_NUM), "velocity": _arr(_NUM),
            "target_position": _arr(_NUM), "target_velocity": _arr(_NUM), "task": _STR,
        },
        title="arm_cli.JointStates",
    ),
    "arm_cli.Task": _obj({"timestamp": _TIME, "event": _STR, "label": _STR}, title="arm_cli.Task"),
    "arm_cli.Sim": _obj(
        {
            "timestamp": _TIME, "sim_time": _NUM, "wall_time": _NUM, "wall_clock": _TIME,
            "rtf": _NUM, "rtf_avg": _NUM, "step": _INT, "physics_dt": _NUM,
        },
        title="arm_cli.Sim",
    ),
}

TOPICS: Dict[str, str] = {
    "/joint_states": "arm_cli.JointStates",
    "/ee_pose": "foxglove.PoseInFrame",
    "/tf": "foxglove.FrameTransforms",
    "/task": "arm_cli.Task",
    "/commands": "foxglove.Log",
    "/log": "foxglove.Log",
    "/hand_camera/image": "foxglove.CompressedImage",
    "/hand_camera/calibration": "foxglove.CameraCalibration",
    "/hand_camera/retrieved": "foxglove.CompressedImage",
    "/notes": "foxglove.Log",
    "/reasoning/strategy": "foxglove.Log",
    "/reasoning/motor": "foxglove.Log",
    "/hand_camera/points": "foxglove.PointCloud",
    "/viewport/image": "foxglove.CompressedImage",
    "/sim": "arm_cli.Sim",
}


REASONING_TOPICS: Dict[str, str] = {"strategy": "/reasoning/strategy", "motor": "/reasoning/motor"}


def _time(ns: int) -> dict:
    return {"sec": int(ns // 1_000_000_000), "nsec": int(ns % 1_000_000_000)}


def _vec3(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat_wxyz(q) -> dict:
    """(w, x, y, z) as Isaac Sim hands it -> Foxglove's {x, y, z, w}."""
    q = np.asarray(q, dtype=float)
    return {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])}


def _floats(v) -> List[float]:
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pack_point_cloud(xyz: np.ndarray, rgb: Optional[np.ndarray], stride: int = 1) -> bytes:
    """(H, W, 3) world XYZ (NaN = no depth) + matching uint8 RGB -> packed POINT_DTYPE bytes.

    `stride` keeps every stride-th row and column (1 = all 2 M points of a 1080p
    frame, about 33 MB; 2 = a quarter of that).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim == 3:
        xyz = xyz[::stride, ::stride]
        if rgb is not None:
            rgb = np.asarray(rgb)[::stride, ::stride]
    pts = xyz.reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=-1)
    pts = pts[mask]
    rec = np.empty(pts.shape[0], dtype=POINT_DTYPE)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if rgb is None:
        rec["rgba"] = 0xFFFFFFFF
    else:
        cols = np.asarray(rgb).reshape(-1, rgb.shape[-1])[mask].astype(np.uint32)
        rec["rgba"] = cols[:, 0] | (cols[:, 1] << 8) | (cols[:, 2] << 16) | np.uint32(0xFF000000)
    return rec.tobytes()


class McapRecorder:
    """Writes one MCAP file. Create one per recording; `close()` finalises it.

    All methods take a simulated time in seconds; the recorder converts it to the
    message clock described in the module docstring.
    """

    def __init__(
        self,
        path: str,
        image_period_s: float = 0.2,
        jpeg_quality: int = 80,
        point_stride: int = 1,
        epoch_ns: Optional[int] = None,
    ) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.image_period_s = image_period_s
        self.jpeg_quality = jpeg_quality
        self.point_stride = max(1, int(point_stride))
        self.epoch_ns = int(time.time_ns()) if epoch_ns is None else int(epoch_ns)
        self._file = open(self.path, "wb")
        try:
            compression = CompressionType.ZSTD
            import zstandard  # noqa: F401  (optional dependency of mcap)
        except ImportError:
            compression = CompressionType.NONE
        self._writer = Writer(self._file, compression=compression)
        self._writer.start(profile="", library="arm_cli foxglove_recorder")
        self._schema_ids: Dict[str, int] = {}
        self._channel_ids: Dict[str, int] = {}
        self.counts: Dict[str, int] = {}
        self._last_image_t: Dict[str, float] = {}
        self._calibration: Dict[str, Tuple[int, int]] = {}
        self._last_ns = 0
        self._wall0 = time.monotonic()
        self._rtf_window: Deque[Tuple[float, float]] = deque()   # (wall_s, sim_s) of recent /sim samples
        self._sim_elapsed = 0.0   # sim seconds recorded, summed across `reset`s
        self.closed = False

    # ---- plumbing -----------------------------------------------------------------
    def _channel(self, topic: str) -> int:
        cid = self._channel_ids.get(topic)
        if cid is None:
            schema_name = TOPICS[topic]
            sid = self._schema_ids.get(schema_name)
            if sid is None:
                sid = self._writer.register_schema(
                    name=schema_name, encoding="jsonschema",
                    data=json.dumps(SCHEMAS[schema_name]).encode("utf-8"),
                )
                self._schema_ids[schema_name] = sid
            cid = self._writer.register_channel(topic=topic, message_encoding="json", schema_id=sid)
            self._channel_ids[topic] = cid
        return cid

    def _ns(self, sim_time_s: float) -> int:
        ns = self.epoch_ns + int(round(float(sim_time_s) * 1e9))
        # MCAP readers assume log times never go backwards within a channel; the sim
        # clock restarts at 0 on `reset`, so never let the clock step back.
        if ns < self._last_ns:
            ns = self._last_ns
        self._last_ns = ns
        return ns

    def write(self, topic: str, sim_time_s: float, message: dict) -> None:
        if self.closed:
            return
        ns = self._ns(sim_time_s)
        message.setdefault("timestamp", _time(ns))
        self._writer.add_message(
            channel_id=self._channel(topic), log_time=ns, publish_time=ns,
            data=json.dumps(message, separators=(",", ":")).encode("utf-8"),
        )
        self.counts[topic] = self.counts.get(topic, 0) + 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._writer.finish()
        self._file.close()

    def summary(self) -> str:
        return ", ".join(f"{t} x{n}" for t, n in sorted(self.counts.items()))

    # ---- per-step state -----------------------------------------------------------
    def joint_states(
        self, t: float, names: Sequence[str], position, velocity,
        target_position=None, target_velocity=None, task: str = "",
    ) -> None:
        self.write("/joint_states", t, {
            "name": list(names),
            "position": _floats(position),
            "velocity": _floats(velocity),
            "target_position": [] if target_position is None else _floats(target_position),
            "target_velocity": [] if target_velocity is None else _floats(target_velocity),
            "task": task,
        })

    def ee_pose(self, t: float, position, quat_wxyz) -> None:
        self.write("/ee_pose", t, {
            "frame_id": WORLD_FRAME,
            "pose": {"position": _vec3(position), "orientation": _quat_wxyz(quat_wxyz)},
        })

    def transforms(self, t: float, frames: Iterable[Tuple[str, np.ndarray, np.ndarray]], parent: str = WORLD_FRAME) -> None:
        """frames: (child_frame_id, position, quaternion (w, x, y, z)) in the parent frame."""
        ns = self._ns(t)
        stamp = _time(ns)
        self.write("/tf", t, {
            "transforms": [
                {
                    "timestamp": stamp, "parent_frame_id": parent, "child_frame_id": name,
                    "translation": _vec3(pos), "rotation": _quat_wxyz(quat),
                }
                for name, pos, quat in frames
            ]
        })

    def sim(self, t: float, step: int, physics_dt: float, wall_s: Optional[float] = None) -> None:
        """Both clocks side by side. `wall_s` is wall-clock seconds since the recording
        started (default: measured now; tests pass it in)."""
        if wall_s is None:
            wall_s = time.monotonic() - self._wall0
        t, wall_s = float(t), float(wall_s)
        last = self._rtf_window[-1] if self._rtf_window else None
        if last is not None and t >= last[1]:
            self._sim_elapsed += t - last[1]
        elif last is not None:
            self._rtf_window.clear()   # `reset` restarted the sim clock
        self._rtf_window.append((wall_s, t))
        while len(self._rtf_window) > 2 and wall_s - self._rtf_window[1][0] >= RTF_WINDOW_S:
            self._rtf_window.popleft()
        w0, s0 = self._rtf_window[0]
        self.write("/sim", t, {
            "sim_time": t,
            "wall_time": wall_s,
            "wall_clock": _time(self.epoch_ns + int(round(wall_s * 1e9))),
            "rtf": (t - s0) / (wall_s - w0) if wall_s > w0 else 0.0,
            "rtf_avg": self._sim_elapsed / wall_s if wall_s > 0 else 0.0,
            "step": int(step),
            "physics_dt": float(physics_dt),
        })

    # ---- events -------------------------------------------------------------------
    def task(self, t: float, event: str, label: str) -> None:
        self.write("/task", t, {"event": event, "label": label})

    def command(self, t: float, line: str) -> None:
        self.write("/commands", t, {"level": LOG_INFO, "message": line, "name": "console", "file": "", "line": 0})

    def log(self, t: float, message: str, level: int = LOG_INFO) -> None:
        self.write("/log", t, {"level": level, "message": message, "name": "arm_cli", "file": "", "line": 0})

    def note(self, t: float, text: str, source: str = "note") -> None:
        """Free text typed by the user, stamped at `t` like any other message."""
        self.write("/notes", t, {"level": LOG_INFO, "message": text, "name": source, "file": "", "line": 0})

    def reasoning(self, t: float, kind: str, text: str) -> None:
        """The operator's thinking: kind "strategy" (overall plan, problem solving) or
        "motor" (how to move: grasp heights, waypoints, orientations, speeds)."""
        topic = REASONING_TOPICS[kind]
        self.write(topic, t, {"level": LOG_INFO, "message": text, "name": kind, "file": "", "line": 0})

    def retrieved_image(self, t: float, rgb: np.ndarray, path: str = "") -> None:
        """An image arm_cli wrote to disk (snap / points / watch). `path` goes to /log."""
        self.image("/hand_camera/retrieved", t, rgb, quality=95)
        if path:
            self.log(t, f"retrieved image -> {path}")

    # ---- camera -------------------------------------------------------------------
    def image_due(self, topic: str, t: float) -> bool:
        last = self._last_image_t.get(topic)
        return last is None or t - last >= self.image_period_s - 1e-9

    def image(self, topic: str, t: float, rgb: np.ndarray, frame_id: str = CAMERA_FRAME, quality: Optional[int] = None) -> None:
        self._last_image_t[topic] = t
        self.write(topic, t, {
            "frame_id": frame_id,
            "format": "jpeg",
            "data": base64.b64encode(jpeg_bytes(rgb, self.jpeg_quality if quality is None else quality)).decode("ascii"),
        })

    def camera_calibration(self, t: float, K: np.ndarray, width: int, height: int, frame_id: str = CAMERA_FRAME, topic: str = "/hand_camera/calibration") -> None:
        """Pinhole intrinsics. Only written when they differ from the last ones sent on this topic."""
        K = np.asarray(K, dtype=float).reshape(3, 3)
        key = (int(width), int(height), round(float(K[0, 0]), 3), round(float(K[0, 2]), 3))
        if self._calibration.get(topic) == key:
            return
        self._calibration[topic] = key
        P = np.zeros((3, 4))
        P[:, :3] = K
        self.write(topic, t, {
            "frame_id": frame_id, "width": int(width), "height": int(height),
            "distortion_model": "plumb_bob", "D": [0.0] * 5,
            "K": _floats(K), "R": _floats(np.eye(3)), "P": _floats(P),
        })

    def point_cloud(self, t: float, xyz: np.ndarray, rgb: Optional[np.ndarray], frame_id: str = WORLD_FRAME, topic: str = "/hand_camera/points") -> int:
        """World-frame cloud from a `points` capture. Returns the number of points written."""
        data = pack_point_cloud(xyz, rgb, self.point_stride)
        n = len(data) // POINT_DTYPE.itemsize
        self.write(topic, t, {
            "frame_id": frame_id,
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
            "point_stride": POINT_DTYPE.itemsize,
            "fields": POINT_FIELDS,
            "data": base64.b64encode(data).decode("ascii"),
        })
        return n


def read_summary(path: str) -> Dict[str, int]:
    """Topic -> message count of an MCAP file (for tests and `python foxglove_recorder.py FILE`)."""
    from mcap.reader import make_reader

    counts: Dict[str, int] = {}
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _, channel, _ in reader.iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
    return counts


if __name__ == "__main__":
    import sys

    for file in sys.argv[1:]:
        print(file)
        for topic, n in sorted(read_summary(file).items()):
            print(f"  {topic:28s} {n}")
