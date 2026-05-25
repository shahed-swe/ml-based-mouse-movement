"""
Hard-coded landmark arrays (21 x 3, normalized x/y/z) for each gesture.

Coordinate convention:
  x: 0.0 = left edge, 1.0 = right edge
  y: 0.0 = top edge,  1.0 = bottom edge  (smaller y = higher in image)
  z: depth (not used by classifier)

All arrays represent a RIGHT hand in mirrored / selfie-view unless stated otherwise.
"""

import numpy as np

from handgester.landmarks import HandLandmarks

# ---------------------------------------------------------------------------
# Base skeleton — a relaxed "fist" reference position.
# All fingertips start below their PIPs (closed).
# Layout (column = [x, y, z]):
#   rows 0–20 follow MediaPipe landmark order.
# ---------------------------------------------------------------------------
#                       x      y      z
_BASE: list[list[float]] = [
    [0.50, 0.90, 0.00],  # 0  WRIST
    [0.45, 0.80, 0.00],  # 1  THUMB_CMC
    [0.38, 0.74, 0.00],  # 2  THUMB_MCP
    [0.36, 0.70, 0.00],  # 3  THUMB_IP         — |IP-MCP|.x = 0.02
    [0.37, 0.68, 0.00],  # 4  THUMB_TIP        — |TIP-MCP|.x = 0.01 < 0.02 → not extended
    [0.45, 0.65, 0.00],  # 5  INDEX_MCP
    [0.44, 0.58, 0.00],  # 6  INDEX_PIP
    [0.44, 0.62, 0.00],  # 7  INDEX_DIP
    [0.44, 0.66, 0.00],  # 8  INDEX_TIP        — TIP.y > PIP.y (folded)
    [0.50, 0.64, 0.00],  # 9  MIDDLE_MCP
    [0.50, 0.57, 0.00],  # 10 MIDDLE_PIP
    [0.50, 0.62, 0.00],  # 11 MIDDLE_DIP
    [0.50, 0.66, 0.00],  # 12 MIDDLE_TIP       — folded
    [0.55, 0.65, 0.00],  # 13 RING_MCP
    [0.56, 0.58, 0.00],  # 14 RING_PIP
    [0.56, 0.63, 0.00],  # 15 RING_DIP
    [0.56, 0.67, 0.00],  # 16 RING_TIP         — folded
    [0.60, 0.68, 0.00],  # 17 PINKY_MCP
    [0.62, 0.62, 0.00],  # 18 PINKY_PIP
    [0.62, 0.66, 0.00],  # 19 PINKY_DIP
    [0.62, 0.70, 0.00],  # 20 PINKY_TIP        — folded
]


def _arr(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def make_hand(points: np.ndarray, handedness: str = "Right") -> HandLandmarks:
    return HandLandmarks(points=points, handedness=handedness, score=1.0)


# ---------------------------------------------------------------------------
# FIST — all five fingers folded (use base as-is)
# ---------------------------------------------------------------------------
FIST_RIGHT = make_hand(_arr(_BASE))


# ---------------------------------------------------------------------------
# OPEN_PALM — all fingertips raised above their PIPs, thumb abducted
# ---------------------------------------------------------------------------
_open_palm = [row[:] for row in _BASE]
# Extend index: TIP.y < PIP.y
_open_palm[8][1]  = 0.40   # INDEX_TIP   y (PIP is 0.58)
_open_palm[7][1]  = 0.49   # INDEX_DIP
# Extend middle
_open_palm[12][1] = 0.38   # MIDDLE_TIP  (PIP is 0.57)
_open_palm[11][1] = 0.47
# Extend ring
_open_palm[16][1] = 0.40   # RING_TIP    (PIP is 0.58)
_open_palm[15][1] = 0.49
# Extend pinky
_open_palm[20][1] = 0.44   # PINKY_TIP   (PIP is 0.62)
_open_palm[19][1] = 0.53
# Abduct thumb: move TIP far from MCP in x so |TIP.x - MCP.x| > |IP.x - MCP.x|
# MCP.x = 0.38, IP.x = 0.33 → |IP - MCP| = 0.05; set TIP.x so |TIP - MCP| = 0.15
_open_palm[4][0]  = 0.23   # THUMB_TIP   x
_open_palm[4][1]  = 0.68   # keep y similar to IP
OPEN_PALM_RIGHT = make_hand(_arr(_open_palm))


# ---------------------------------------------------------------------------
# THUMBS_UP — thumb extended upward, other four fingers folded
#   Requirements: fingers_state == (True,False,False,False,False)
#                 THUMB_TIP.y < INDEX_MCP.y
# ---------------------------------------------------------------------------
_thumbs_up = [row[:] for row in _BASE]
# Thumb TIP should be well above INDEX_MCP (y=0.65)
_thumbs_up[4][1]  = 0.45   # THUMB_TIP  y (well above INDEX_MCP at 0.65)
_thumbs_up[4][0]  = 0.23   # THUMB_TIP  x — abducted (MCP.x=0.38, |diff|=0.15 > 0.05)
THUMBS_UP_RIGHT = make_hand(_arr(_thumbs_up))


# ---------------------------------------------------------------------------
# PEACE — index + middle extended, ring + pinky folded, thumb ignored
# ---------------------------------------------------------------------------
_peace = [row[:] for row in _BASE]
_peace[8][1]  = 0.40   # INDEX_TIP  (PIP 0.58)
_peace[7][1]  = 0.49
_peace[12][1] = 0.38   # MIDDLE_TIP (PIP 0.57)
_peace[11][1] = 0.47
PEACE_RIGHT = make_hand(_arr(_peace))


# ---------------------------------------------------------------------------
# POINTING — only index extended
# ---------------------------------------------------------------------------
_pointing = [row[:] for row in _BASE]
_pointing[8][1]  = 0.40   # INDEX_TIP (PIP 0.58)
_pointing[7][1]  = 0.49
POINTING_RIGHT = make_hand(_arr(_pointing))


# ---------------------------------------------------------------------------
# OK — middle + ring + pinky extended, thumb TIP close to index TIP
#   hand_scale = distance(WRIST=0.90, MIDDLE_MCP=0.64) ≈ 0.264 (y-only, x same)
#   0.4 * scale ≈ 0.106 — place THUMB_TIP and INDEX_TIP within that distance
# ---------------------------------------------------------------------------
_ok = [row[:] for row in _BASE]
# Extend middle, ring, pinky
_ok[12][1] = 0.38   # MIDDLE_TIP (PIP 0.57)
_ok[11][1] = 0.47
_ok[16][1] = 0.40   # RING_TIP   (PIP 0.58)
_ok[15][1] = 0.49
_ok[20][1] = 0.44   # PINKY_TIP  (PIP 0.62)
_ok[19][1] = 0.53
# Pinch: move THUMB_TIP right next to INDEX_TIP
# INDEX_TIP stays folded at x=0.44, y=0.66
# Place THUMB_TIP at (0.44, 0.66) — distance = 0 < threshold
_ok[4][0]  = 0.44   # THUMB_TIP  x
_ok[4][1]  = 0.66   # THUMB_TIP  y
# hand_scale: WRIST=(0.50,0.90), MIDDLE_MCP=(0.50,0.64) → scale=0.26
# 0.4 * 0.26 = 0.104 — distance of 0.0 is well within limit
OK_RIGHT = make_hand(_arr(_ok))
