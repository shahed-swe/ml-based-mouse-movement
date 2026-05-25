import numpy as np

# Landmark indices per MediaPipe hand model
WRIST = 0
THUMB_CMC = 1;  THUMB_MCP = 2;  THUMB_IP = 3;  THUMB_TIP = 4
INDEX_MCP = 5;  INDEX_PIP = 6;  INDEX_DIP = 7;  INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13;  RING_PIP = 14;  RING_DIP = 15;  RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

_FINGER_TIPS = {
    "index":  (INDEX_TIP,  INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring":   (RING_TIP,   RING_PIP),
    "pinky":  (PINKY_TIP,  PINKY_PIP),
}


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def finger_extended(points: np.ndarray, finger: str) -> bool:
    tip_idx, pip_idx = _FINGER_TIPS[finger]
    # In normalized coords y increases downward, so a raised fingertip has smaller y.
    return bool(points[tip_idx, 1] < points[pip_idx, 1])


def thumb_extended(points: np.ndarray, handedness: str) -> bool:
    # MediaPipe reports handedness from the camera's POV (mirrored for selfie-view).
    # Rather than relying on left/right direction (which flips with mirroring), we
    # check whether the TIP has moved further from the MCP than the IP has — a
    # reliable proxy for "thumb abducted away from palm" regardless of orientation.
    return bool(
        abs(points[THUMB_TIP, 0] - points[THUMB_MCP, 0])
        > abs(points[THUMB_IP, 0] - points[THUMB_MCP, 0])
    )


def fingers_state(
    points: np.ndarray, handedness: str
) -> tuple[bool, bool, bool, bool, bool]:
    return (
        thumb_extended(points, handedness),
        finger_extended(points, "index"),
        finger_extended(points, "middle"),
        finger_extended(points, "ring"),
        finger_extended(points, "pinky"),
    )


def hand_scale(points: np.ndarray) -> float:
    return distance(points[WRIST], points[MIDDLE_MCP])


def hand_bbox(
    points: np.ndarray, frame_shape: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    xs = (points[:, 0] * w).astype(int)
    ys = (points[:, 1] * h).astype(int)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
