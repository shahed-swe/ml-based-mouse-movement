import cv2
import numpy as np


class Camera:
    """Wraps cv2.VideoCapture with context manager support."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self._cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
