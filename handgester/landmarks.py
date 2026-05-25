from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


@dataclass
class HandLandmarks:
    points: np.ndarray   # shape (21, 3) — normalized x, y, z
    handedness: str      # "Left" or "Right"
    score: float


def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
        urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandDetector:
    """Isolates all MediaPipe hand detection logic (Tasks API)."""

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_idx = 0

    def detect(self, bgr_frame: np.ndarray) -> list[HandLandmarks]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires a monotonically increasing timestamp in ms
        self._frame_idx += 1
        timestamp_ms = self._frame_idx * 33  # ~30 fps assumed; only ordering matters
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        out = []
        for hand_lm, hand_info in zip(result.hand_landmarks, result.handedness):
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32
            )
            label = hand_info[0].category_name
            score = hand_info[0].score
            out.append(HandLandmarks(points=points, handedness=label, score=score))

        return out

    def close(self):
        self._landmarker.close()
