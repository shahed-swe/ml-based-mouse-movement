import cv2
import numpy as np

from .classifier import Gesture
from .geometry import hand_bbox
from .landmarks import HandLandmarks

# Standard MediaPipe hand skeleton connections (index pairs).
# Source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  (1, 2),   (2, 3),   (3, 4),   # thumb
    (0, 5),  (5, 6),   (6, 7),   (7, 8),   # index
    (0, 9),  (9, 10),  (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9),  (9, 13),  (13, 17),           # palm knuckle bar
]

_GESTURE_COLORS: dict[Gesture, tuple[int, int, int]] = {
    Gesture.OPEN_PALM: (0, 220, 0),
    Gesture.FIST:      (0, 0, 220),
    Gesture.THUMBS_UP: (0, 200, 255),
    Gesture.PEACE:     (200, 0, 200),
    Gesture.POINTING:  (255, 180, 0),
    Gesture.OK:        (0, 255, 180),
    Gesture.UNKNOWN:   (120, 120, 120),
}


def _to_px(point: np.ndarray, frame_shape: tuple) -> tuple[int, int]:
    h, w = frame_shape[:2]
    return int(point[0] * w), int(point[1] * h)


def draw_landmarks(frame: np.ndarray, hand: HandLandmarks) -> None:
    pts = hand.points
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, _to_px(pts[a], frame.shape), _to_px(pts[b], frame.shape),
                 (200, 200, 200), 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(frame, _to_px(pts[i], frame.shape), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, _to_px(pts[i], frame.shape), 4, (0, 0, 0), 1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, hand: HandLandmarks, gesture: Gesture) -> None:
    x1, y1, x2, y2 = hand_bbox(hand.points, frame.shape)
    color = _GESTURE_COLORS[gesture]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = gesture.value
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    banner_y1 = max(0, y1 - th - baseline - 8)
    banner_y2 = y1
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 8, banner_y2), color, -1)
    cv2.putText(frame, label, (x1 + 4, banner_y2 - baseline - 2),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, num_hands: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hands: {num_hands}", (10, 56), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
