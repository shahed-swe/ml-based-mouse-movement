import time

import cv2

from .actions import MouseController
from .camera import Camera
from .classifier import GestureClassifier
from .landmarks import HandDetector
from .overlay import draw_hud, draw_landmarks, draw_label


class Pipeline:
    def __init__(
        self,
        camera: Camera,
        detector: HandDetector,
        classifier: GestureClassifier,
        mirror: bool = True,
        mouse: MouseController | None = None,
    ):
        self._camera = camera
        self._detector = detector
        self._classifier = classifier
        self._mirror = mirror
        self._mouse = mouse

    def run(self) -> None:
        fps = 0.0
        alpha = 0.1
        prev_time = time.perf_counter()

        cv2.namedWindow("Hand Gester", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gester", 640, 360)
        cv2.moveWindow("Hand Gester", 20, 20)  # tuck into top-left corner

        while True:
            frame = self._camera.read()
            if frame is None:
                continue

            if self._mirror:
                frame = cv2.flip(frame, 1)

            hands = self._detector.detect(frame)

            for i, hand in enumerate(hands):
                gesture = self._classifier.classify(hand)
                draw_landmarks(frame, hand)
                draw_label(frame, hand, gesture)
                # Only the first detected hand drives the mouse.
                if i == 0 and self._mouse is not None:
                    self._mouse.handle(hand, gesture)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = alpha * instant_fps + (1 - alpha) * fps
            prev_time = now

            draw_hud(frame, fps, len(hands))

            cv2.imshow("Hand Gester", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break

        cv2.destroyAllWindows()
