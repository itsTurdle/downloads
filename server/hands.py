"""Hand tracking, and detection of the two-handed "director's frame" gesture.

Runs on the same decoded frame the depth model already gets, so the phone stays dumb
and nothing extra crosses the network. Only a handful of landmarks are forwarded to
the viewer, which places the frame in real 3D by sampling the depth map at those
pixels -- so the square sits exactly where your fingers are, not pasted on the screen.

MediaPipe landmark indices used here:

      4  thumb tip          8  index tip           5  index MCP (knuckle)
     12  middle tip        16  ring tip           20  pinky tip
      0  wrist              9  middle MCP
"""

import threading
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent / "_models" / "hand_landmarker.task"

THUMB_TIP, INDEX_TIP, INDEX_MCP = 4, 8, 5
THUMB_MCP, THUMB_IP = 2, 3
MIDDLE_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
WRIST, MIDDLE_MCP = 0, 9

# (pip, tip) per finger. A finger is extended when its tip is further from the wrist
# than its own middle joint -- scale-free, and unlike a reach threshold it does not
# care how big the hand is or where the wrist happens to sit.
FINGERS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}


def _line_intersect(p1, d1, p2, d2):
    """Where two lines cross, or None if they are near-parallel."""
    det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
    if abs(det) < 1e-6:
        return None
    rhs = p2 - p1
    t = (rhs[0] * (-d2[1]) - (-d2[0]) * rhs[1]) / det
    return p1 + d1 * t


class HandTracker:
    """Not thread-safe; the bridge drives it from its single depth worker thread."""

    def __init__(self, max_hands: int = 2, min_conf: float = 0.5):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not MODEL.exists():
            raise FileNotFoundError(f"missing {MODEL}")

        self._mp_image = None
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.lock = threading.Lock()
        self._t = 0

        # How much further than its own middle joint a tip must reach to count as
        # extended, and the (lower) bar under which a finger counts as curled.
        self.extend_ratio = 1.15
        self.curl_ratio = 1.02

    def close(self):
        with self.lock:
            self.landmarker.close()

    def detect(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> dict:
        """Returns normalised landmarks plus a `frame` quad when the gesture is held."""
        import mediapipe as mp

        with self.lock:
            if timestamp_ms is None:
                self._t += 33
                timestamp_ms = self._t
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(image, timestamp_ms)

        hands = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            label = "?"
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                label = res.handedness[i][0].category_name
            hands.append({
                "label": label,
                "thumb": pts[THUMB_TIP, :2].tolist(),
                "thumbBase": pts[THUMB_MCP, :2].tolist(),
                "index": pts[INDEX_TIP, :2].tolist(),
                "knuckle": pts[INDEX_MCP, :2].tolist(),
                "framing": bool(self._is_framing(pts)),
            })

        out = {"hands": hands}
        quad = self._frame_quad(hands)
        if quad:
            out["frame"] = quad
        return out

    def _is_framing(self, pts: np.ndarray) -> bool:
        """Thumb and index extended, the other three curled: half of the gesture."""
        wrist = pts[WRIST, :2]

        def farther(tip, joint, margin):
            a = float(np.linalg.norm(pts[tip, :2] - wrist))
            b = float(np.linalg.norm(pts[joint, :2] - wrist))
            return a > b * margin

        index_pip = FINGERS["index"][0]
        index_out = farther(INDEX_TIP, index_pip, self.extend_ratio)
        # The thumb bends sideways rather than curling, so it is measured against its
        # own IP joint and given no extra margin.
        thumb_out = farther(THUMB_TIP, THUMB_IP, 1.0)
        folded = sum(not farther(tip, pip, self.curl_ratio)
                     for name, (pip, tip) in FINGERS.items() if name != "index")
        return thumb_out and index_out and folded >= 2

    @staticmethod
    def _l_vertex(h: dict):
        """The corner of the L: where the thumb's line crosses the index's line.

        Falls back to the midpoint of the two tips when the fingers are near-parallel
        and the intersection is unstable or off at infinity.
        """
        ip = np.array(h["knuckle"], np.float32)
        idir = np.array(h["index"], np.float32) - ip
        tp = np.array(h.get("thumbBase") or h["thumb"], np.float32)
        tdir = np.array(h["thumb"], np.float32) - tp
        if np.linalg.norm(idir) < 1e-5 or np.linalg.norm(tdir) < 1e-5:
            return None
        v = _line_intersect(ip, idir, tp, tdir)
        if v is None:
            return (np.array(h["index"], np.float32)
                    + np.array(h["thumb"], np.float32)) / 2
        # A wild intersection means the pose is not really an L.
        if not np.all(np.isfinite(v)) or np.abs(v).max() > 4.0:
            return (np.array(h["index"], np.float32)
                    + np.array(h["thumb"], np.float32)) / 2
        return v

    def _frame_quad(self, hands: list) -> dict | None:
        """Build an oriented rectangle from two hands both making the L shape.

        Orientation comes from the index fingers rather than from the corner-to-corner
        diagonal: the diagonal's angle changes as you open and close the rectangle,
        which would make the square appear to spin when you only meant to resize it.
        The finger direction only changes when you actually twist.
        """
        framing = [h for h in hands if h["framing"]]
        if len(framing) < 2:
            return None
        a, b = framing[0], framing[1]

        # The rectangle's corners are the two L-vertices -- where each hand's thumb
        # line meets its index line -- not a box around the four fingertips. A
        # fingertip box is dominated by the span of each hand rather than the gap
        # between them: measured, tripling the separation grew the area only 8%, so
        # moving your hands apart barely resized it.
        ca, cb = self._l_vertex(a), self._l_vertex(b)
        if ca is None or cb is None:
            return None
        pts = np.stack([ca, cb])
        centre = pts.mean(axis=0)

        da = np.array(a["index"], np.float32) - np.array(a["knuckle"], np.float32)
        db = np.array(b["index"], np.float32) - np.array(b["knuckle"], np.float32)
        # In this gesture the index fingers point at each other, so flip one before
        # averaging or they cancel out.
        if float(np.dot(da, db)) < 0:
            db = -db
        axis = da + db
        n = float(np.linalg.norm(axis))
        if n < 1e-5:
            return None
        axis /= n

        # Measure the corners in the rectangle's own frame to get its extent.
        perp = np.array([-axis[1], axis[0]], dtype=np.float32)
        rel = pts - centre
        u = rel @ axis
        v = rel @ perp
        half_u = float(np.abs(u).max())
        half_v = float(np.abs(v).max())
        # Two opposite corners give the rectangle directly, but a hand held slightly
        # off the ideal pose can collapse one dimension; keep a floor so the square
        # stays a square rather than degenerating to a line.
        half_u = max(half_u, 0.02)
        half_v = max(half_v, 0.02)
        if half_u * half_v < 0.0015:
            return None            # hands too close together to mean anything

        corners = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = centre + axis * (su * half_u) + perp * (sv * half_v)
            corners.append([float(p[0]), float(p[1])])

        return {
            "corners": corners,
            "centre": [float(centre[0]), float(centre[1])],
            "angle": float(np.arctan2(axis[1], axis[0])),
            "size": [half_u * 2, half_v * 2],
        }
