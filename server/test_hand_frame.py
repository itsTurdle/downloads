"""Geometry tests for the two-handed framing gesture.

Detection itself is MediaPipe's job; what needs checking is our interpretation of the
landmarks. These build exact synthetic hands so the expected answer is known, which a
photo could never give.

Layout convention: wrist at the origin, middle knuckle one unit away, so "hand size"
is 1 and every reach is directly comparable.
"""

import sys

import numpy as np

import hands as H

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def make_hand(thumb_out=True, index_out=True, others_out=False,
              rot=0.0, at=(0.0, 0.0)):
    """A 21-landmark right-angle "L" hand, posed so the L-vertex sits at `at`.

    The index runs straight up from its knuckle and the thumb straight out from its
    base, meeting at the origin before translation -- so `at` IS the rectangle corner
    that hand contributes, which is what makes the separation test meaningful.

    Every finger gets a real PIP joint, because that is what the extension test
    compares against: an extended finger puts its tip beyond that joint, a curled one
    folds the tip back toward the palm and therefore inside it.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    def place(idx, vec):
        pts[idx, :2] = R @ np.asarray(vec, np.float32) + np.asarray(at, np.float32)

    wrist = np.array([-0.5, -0.6], np.float32)
    place(H.WRIST, wrist)
    place(H.MIDDLE_MCP, (-0.4, 0.35))

    # Index: vertical line through x = 0, knuckle then joint then tip.
    place(H.INDEX_MCP, (0.0, 0.15))
    place(6, (0.0, 0.55))                                    # index PIP
    place(H.INDEX_TIP, (0.0, 1.05) if index_out else (-0.15, 0.35))

    # Thumb: horizontal line through y = 0, so the two lines meet at the origin.
    place(H.THUMB_MCP, (0.15, 0.0))
    place(H.THUMB_IP, (0.55, 0.0))
    place(H.THUMB_TIP, (1.0, 0.0) if thumb_out else (0.35, -0.1))

    # The remaining three, curled unless asked otherwise.
    for n, (pip, tip) in enumerate((H.FINGERS["middle"], H.FINGERS["ring"],
                                    H.FINGERS["pinky"])):
        base_x = -0.25 - 0.18 * n
        place(pip, (base_x, 0.30))
        place(tip, (base_x, 0.75) if others_out else (base_x + 0.05, 0.05))
    return pts


print("finger-pose gate:")
t = H.HandTracker.__new__(H.HandTracker)      # geometry only; no model needed
t.extend_ratio, t.curl_ratio = 1.15, 1.02

check("framing pose accepted",
      t._is_framing(make_hand()))
check("open palm rejected",
      not t._is_framing(make_hand(others_out=True)))
check("fist rejected",
      not t._is_framing(make_hand(thumb_out=False, index_out=False)))
check("index alone (pointing) rejected",
      not t._is_framing(make_hand(thumb_out=False)))
check("thumbs-up (index curled) rejected",
      not t._is_framing(make_hand(index_out=False)))

print("\nrectangle from two hands:")


def two_hands(rot=0.0, sep=0.4):
    """Two framing hands facing each other, optionally twisted together."""
    a = make_hand(rot=rot, at=(-sep, 0.0))
    b = make_hand(rot=rot + np.pi, at=(sep, 0.0))
    mk = lambda p: {
        "thumb": p[H.THUMB_TIP, :2].tolist(),
        "thumbBase": p[H.THUMB_MCP, :2].tolist(),
        "index": p[H.INDEX_TIP, :2].tolist(),
        "knuckle": p[H.INDEX_MCP, :2].tolist(),
        "framing": True,
    }
    return [mk(a), mk(b)]


q = t._frame_quad(two_hands())
check("quad produced from two framing hands", q is not None)
if q:
    c = np.array(q["corners"])
    check("four corners", c.shape == (4, 2), str(c.shape))
    # Opposite corners must share a midpoint, i.e. it is a real parallelogram.
    m1, m2 = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
    check("corners form a parallelogram", np.allclose(m1, m2, atol=1e-4),
          f"midpoints {m1.round(4)} vs {m2.round(4)}")
    # Adjacent edges perpendicular.
    e1, e2 = c[1] - c[0], c[2] - c[1]
    cosang = abs(float(e1 @ e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    check("corners are square (edges perpendicular)", cosang < 1e-4,
          f"|cos| = {cosang:.2e}")

check("one hand alone produces nothing", t._frame_quad(two_hands()[:1]) is None)
check("non-framing hands produce nothing",
      t._frame_quad([dict(h, framing=False) for h in two_hands()]) is None)

print("\ntwist follows the hands:")
base = t._frame_quad(two_hands(rot=0.0))
for deg in (15, 30, -25):
    tw = t._frame_quad(two_hands(rot=np.radians(deg)))
    got = np.degrees(tw["angle"] - base["angle"])
    got = (got + 180) % 360 - 180        # shortest signed difference
    check(f"twisting {deg:+d} deg rotates the square", abs(got - deg) < 1.0,
          f"got {got:+.2f} deg")

print("\nresizing does not spin it:")
wide = t._frame_quad(two_hands(sep=0.9))
narrow = t._frame_quad(two_hands(sep=0.25))
spin = np.degrees(abs(wide["angle"] - narrow["angle"]))
check("angle stable while opening/closing", spin < 1.0, f"{spin:.2f} deg drift")
# Area, not one edge: these synthetic hands separate along x while the index axis is
# near-vertical, so the growth lands on the perpendicular edge. Area is the claim that
# actually matters -- move your hands apart, the square gets bigger.
area = lambda q: q["size"][0] * q["size"][1]
check("separating the hands enlarges the rectangle",
      area(wide) > area(narrow) * 1.5,
      f"area {area(wide):.3f} vs {area(narrow):.3f}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
    sys.exit(1)
print("framing-gesture geometry is correct.")
