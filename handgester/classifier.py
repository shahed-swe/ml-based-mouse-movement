from enum import Enum
from typing import Protocol

from .geometry import (
    INDEX_MCP,
    INDEX_TIP,
    THUMB_TIP,
    distance,
    finger_extended,
    fingers_state,
    hand_scale,
)
from .landmarks import HandLandmarks


class Gesture(str, Enum):
    OPEN_PALM = "Open Palm"
    FIST      = "Fist"
    THUMBS_UP = "Thumbs Up"
    PEACE     = "Peace"
    POINTING  = "Pointing"
    OK        = "OK"
    UNKNOWN   = "Unknown"


class GestureClassifier(Protocol):
    def classify(self, hand: HandLandmarks) -> Gesture: ...


class RuleBasedClassifier:
    def classify(self, hand: HandLandmarks) -> Gesture:
        pts = hand.points
        thumb, index, middle, ring, pinky = fingers_state(pts, hand.handedness)
        scale = hand_scale(pts)

        # OK: pinch between thumb and index while other three fingers are up
        if (
            middle and ring and pinky
            and distance(pts[THUMB_TIP], pts[INDEX_TIP]) < 0.4 * scale
        ):
            return Gesture.OK

        # THUMBS_UP: only thumb up, tip above the index knuckle
        if (thumb, index, middle, ring, pinky) == (True, False, False, False, False):
            if pts[THUMB_TIP, 1] < pts[INDEX_MCP, 1]:
                return Gesture.THUMBS_UP

        # OPEN_PALM: all five extended
        if (thumb, index, middle, ring, pinky) == (True, True, True, True, True):
            return Gesture.OPEN_PALM

        # FIST: none extended
        if (thumb, index, middle, ring, pinky) == (False, False, False, False, False):
            return Gesture.FIST

        # PEACE: index + middle up, ring + pinky down (thumb ignored)
        if index and middle and not ring and not pinky:
            return Gesture.PEACE

        # POINTING: index up, middle/ring/pinky down (thumb ignored)
        if index and not middle and not ring and not pinky:
            return Gesture.POINTING

        return Gesture.UNKNOWN
