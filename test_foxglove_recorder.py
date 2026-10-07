r"""
Unit tests for foxglove_recorder.py. No Isaac Sim needed:

    python -m unittest -v test_foxglove_recorder
"""

import base64
import json
import os
import tempfile
import unittest

import numpy as np
from mcap.reader import make_reader

from foxglove_recorder import POINT_DTYPE, McapRecorder, pack_point_cloud, read_summary


def read_all(path):
    """[(topic, schema name, decoded JSON, log_time)] in file order."""
    out = []
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            out.append((channel.topic, schema.name, json.loads(message.data), message.log_time))
    return out


class TestPackPointCloud(unittest.TestCase):
    def test_drops_nan_and_packs_colour(self):
        xyz = np.full((2, 3, 3), np.nan, dtype=np.float32)
        xyz[0, 0] = [1.0, 2.0, 3.0]
        xyz[1, 2] = [-1.0, 0.5, 0.25]
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        rgb[1, 2] = [0, 128, 255]
        rec = np.frombuffer(pack_point_cloud(xyz, rgb), dtype=POINT_DTYPE)
        self.assertEqual(len(rec), 2)
        np.testing.assert_allclose([rec[0]["x"], rec[0]["y"], rec[0]["z"]], [1, 2, 3])
        self.assertEqual(int(rec[0]["rgba"]), 0xFF0000FF)              # A B G R little-endian packing
        self.assertEqual(int(rec[1]["rgba"]), 0xFF000000 | (255 << 16) | (128 << 8))

    def test_stride_subsamples(self):
        xyz = np.zeros((4, 6, 3), dtype=np.float32)
        self.assertEqual(len(pack_point_cloud(xyz, None, stride=1)) // 16, 24)
        self.assertEqual(len(pack_point_cloud(xyz, None, stride=2)) // 16, 6)


class TestMcapRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "sub", "run.mcap")   # parent dir is created

    def tearDown(self):
        self.dir.cleanup()

    def test_full_session_round_trips(self):
        rec = McapRecorder(self.path, image_period_s=0.5, point_stride=1, epoch_ns=1_000_000_000_000)
        names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[..., 1] = 200
        xyz = np.stack(np.meshgrid(np.arange(12.0), np.arange(8.0), indexing="xy") + [np.ones((8, 12))], axis=-1)
        xyz[0, 0] = np.nan

        rec.command(0.0, "move 0.4 0 0.3")
        rec.task(0.0, "start", "move to [0.4 0 0.3]")
        for step in range(6):
            t = step / 60.0
            rec.joint_states(t, names, np.arange(9) * 0.1, np.zeros(9), np.arange(9) * 0.1, np.zeros(9), "move")
            rec.ee_pose(t, [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
            rec.transforms(t, [("panda_link0", np.zeros(3), np.array([1.0, 0, 0, 0])),
                               ("hand_camera", np.array([0.4, 0, 0.5]), np.array([0.0, 1.0, 0, 0]))])
            rec.sim(t, step, 1 / 60)
            if rec.image_due("/hand_camera/image", t):
                rec.camera_calibration(t, np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1]]), 12, 8)
                rec.image("/hand_camera/image", t, rgb)
        rec.log(0.1, "arrived: position error 1.0 mm")
        n = rec.point_cloud(0.1, xyz, rgb)
        rec.task(0.1, "done", "move to [0.4 0 0.3]")
        rec.close()
        rec.close()  # idempotent

        self.assertEqual(n, 8 * 12 - 1)
        counts = read_summary(self.path)
        self.assertEqual(counts["/joint_states"], 6)
        self.assertEqual(counts["/tf"], 6)
        self.assertEqual(counts["/hand_camera/image"], 1, "0.1 s of sim time at 0.5 s period = one image")
        self.assertEqual(counts["/hand_camera/calibration"], 1)
        self.assertEqual(counts["/hand_camera/points"], 1)
        self.assertEqual(counts["/task"], 2)
        self.assertEqual(counts["/commands"], 1)
        self.assertEqual(counts["/log"], 1)
        self.assertEqual(counts, rec.counts)

        msgs = read_all(self.path)
        by_topic = {}
        for topic, schema, msg, log_time in msgs:
            by_topic.setdefault(topic, []).append((schema, msg, log_time))

        schema, img, log_time = by_topic["/hand_camera/image"][0]
        self.assertEqual(schema, "foxglove.CompressedImage")
        self.assertEqual(img["format"], "jpeg")
        self.assertEqual(img["frame_id"], "hand_camera")
        self.assertTrue(base64.b64decode(img["data"]).startswith(b"\xff\xd8"), "not a JPEG")
        self.assertEqual(log_time, 1_000_000_000_000)
        self.assertEqual(img["timestamp"], {"sec": 1000, "nsec": 0})

        schema, js, log_time = by_topic["/joint_states"][-1]
        self.assertEqual(js["name"], names)
        self.assertAlmostEqual(js["position"][8], 0.8)
        self.assertEqual(log_time, 1_000_000_000_000 + round(5 / 60 * 1e9))

        schema, tf, _ = by_topic["/tf"][0]
        self.assertEqual(schema, "foxglove.FrameTransforms")
        cam = tf["transforms"][1]
        self.assertEqual((cam["parent_frame_id"], cam["child_frame_id"]), ("world", "hand_camera"))
        self.assertEqual(cam["rotation"], {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}, "w,x,y,z -> x,y,z,w")

        schema, pc, _ = by_topic["/hand_camera/points"][0]
        self.assertEqual(schema, "foxglove.PointCloud")
        self.assertEqual(pc["point_stride"], 16)
        self.assertEqual([f["name"] for f in pc["fields"]], ["x", "y", "z", "rgba"])
        pts = np.frombuffer(base64.b64decode(pc["data"]), dtype=POINT_DTYPE)
        self.assertEqual(len(pts), n)
        self.assertEqual(int(pts[0]["rgba"]), 0xFF000000 | (200 << 8))

        schema, cal, _ = by_topic["/hand_camera/calibration"][0]
        self.assertEqual((cal["width"], cal["height"]), (12, 8))
        self.assertEqual(cal["K"][0], 10.0)
        self.assertEqual(len(cal["P"]), 12)

        schema, log, _ = by_topic["/log"][0]
        self.assertEqual(schema, "foxglove.Log")
        self.assertEqual(log["level"], 2)

    def test_clock_never_runs_backwards(self):
        rec = McapRecorder(self.path, epoch_ns=0)
        rec.sim(5.0, 300, 1 / 60)
        rec.sim(0.0, 0, 1 / 60)      # `reset` restarts the sim clock
        rec.sim(0.1, 6, 1 / 60)
        rec.close()
        times = [lt for _, _, _, lt in read_all(self.path)]
        self.assertEqual(times, sorted(times))
        self.assertEqual(times[1], times[0])

    def test_write_after_close_is_ignored(self):
        rec = McapRecorder(self.path, epoch_ns=0)
        rec.sim(0.0, 0, 1 / 60)
        rec.close()
        rec.sim(1.0, 60, 1 / 60)
        self.assertEqual(read_summary(self.path), {"/sim": 1})


if __name__ == "__main__":
    unittest.main()
