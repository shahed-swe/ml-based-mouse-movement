import pyautogui

from .classifier import Gesture
from .geometry import INDEX_MCP, INDEX_TIP, WRIST
from .landmarks import HandLandmarks


# Corner-abort is disabled — hand tracking naturally pushes cursor to corners.
# Use `q` or ESC in the preview window to quit instead.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # no artificial delay between calls


class MouseController:
    """Maps gestures to mouse actions.

    POINTING  -> move cursor to where the index fingertip points
    FIST      -> press & hold left mouse button (drag mode); release fist = mouse up.
                 A quick fist + release acts as a normal click.
    OK        -> right click (fires once per transition into OK)
    PEACE     -> scroll: move hand down = scroll down, move up = scroll up
    OPEN_PALM -> idle / release everything
    """

    def __init__(
        self,
        smoothing: float = 0.35,
        margin: float = 0.15,
        scroll_sensitivity: float = 80.0,
    ):
        self._screen_w, self._screen_h = pyautogui.size()
        print(f"[MouseController] Detected screen size: {self._screen_w} x {self._screen_h}")
        print("[MouseController] If cursor doesn't move across the whole screen:")
        print("  -> System Settings -> Privacy & Security -> Accessibility")
        print("  -> enable your terminal app (Terminal / iTerm / VS Code)")
        self._smoothing = smoothing  # 0 = no smoothing, 1 = never moves
        self._margin = margin        # ignore edges of frame so corners are reachable
        self._scroll_sensitivity = scroll_sensitivity
        self._last_x: float | None = None
        self._last_y: float | None = None
        self._scroll_last_y: float | None = None
        self._scroll_accumulator: float = 0.0
        # Drag state: when FIST starts we anchor the landmark and current cursor
        # so subsequent FIST frames apply a relative delta (no cursor jump).
        self._drag_anchor_landmark: tuple[float, float] | None = None
        self._drag_anchor_cursor: tuple[float, float] | None = None
        self._dragging: bool = False
        self._prev_gesture: Gesture | None = None

    def handle(self, hand: HandLandmarks, gesture: Gesture) -> None:
        # Right-click edge trigger
        if gesture == Gesture.OK and self._prev_gesture != Gesture.OK:
            pyautogui.click(button="right")

        # Drag lifecycle on FIST
        if gesture == Gesture.FIST and not self._dragging:
            self._start_drag(hand)
        elif gesture != Gesture.FIST and self._dragging:
            self._end_drag()

        # Continuous-mode handlers
        if gesture == Gesture.POINTING:
            self._move_to_index(hand)
            self._reset_scroll()
        elif gesture == Gesture.FIST:
            self._continue_drag(hand)
            self._reset_scroll()
        elif gesture == Gesture.PEACE:
            self._do_scroll(hand)
        else:
            self._reset_scroll()

        self._prev_gesture = gesture

    def _reset_scroll(self) -> None:
        self._scroll_last_y = None
        self._scroll_accumulator = 0.0

    def _do_scroll(self, hand: HandLandmarks) -> None:
        current_y = float(hand.points[WRIST, 1])
        if self._scroll_last_y is None:
            self._scroll_last_y = current_y
            return
        # In image coords y grows downward; negate so hand-down → scroll-down.
        dy = current_y - self._scroll_last_y
        self._scroll_accumulator += -dy * self._scroll_sensitivity
        self._scroll_last_y = current_y
        clicks = int(self._scroll_accumulator)
        if clicks != 0:
            pyautogui.scroll(clicks)
            self._scroll_accumulator -= clicks

    def _start_drag(self, hand: HandLandmarks) -> None:
        self._drag_anchor_landmark = (
            float(hand.points[INDEX_MCP, 0]),
            float(hand.points[INDEX_MCP, 1]),
        )
        cur_x, cur_y = pyautogui.position()
        self._drag_anchor_cursor = (float(cur_x), float(cur_y))
        pyautogui.mouseDown(button="left", _pause=False)
        self._dragging = True

    def _continue_drag(self, hand: HandLandmarks) -> None:
        if self._drag_anchor_landmark is None or self._drag_anchor_cursor is None:
            return
        inv = 1.0 / (1 - 2 * self._margin)
        dx_norm = float(hand.points[INDEX_MCP, 0]) - self._drag_anchor_landmark[0]
        dy_norm = float(hand.points[INDEX_MCP, 1]) - self._drag_anchor_landmark[1]
        target_x = self._drag_anchor_cursor[0] + dx_norm * inv * self._screen_w
        target_y = self._drag_anchor_cursor[1] + dy_norm * inv * self._screen_h
        target_x = min(max(target_x, 0), self._screen_w - 1)
        target_y = min(max(target_y, 0), self._screen_h - 1)
        pyautogui.moveTo(target_x, target_y, _pause=False)

    def _end_drag(self) -> None:
        pyautogui.mouseUp(button="left", _pause=False)
        self._dragging = False
        self._drag_anchor_landmark = None
        self._drag_anchor_cursor = None

    def _move_to_index(self, hand: HandLandmarks) -> None:
        # Normalized [0, 1] coords in the (possibly mirrored) frame.
        nx = hand.points[INDEX_TIP, 0]
        ny = hand.points[INDEX_TIP, 1]

        # Remap so the user can reach screen edges without putting their hand
        # fully out of the camera frame.
        nx = (nx - self._margin) / (1 - 2 * self._margin)
        ny = (ny - self._margin) / (1 - 2 * self._margin)
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        target_x = nx * self._screen_w
        target_y = ny * self._screen_h

        if self._last_x is None:
            self._last_x, self._last_y = target_x, target_y
        else:
            self._last_x = self._smoothing * self._last_x + (1 - self._smoothing) * target_x
            self._last_y = self._smoothing * self._last_y + (1 - self._smoothing) * target_y

        pyautogui.moveTo(self._last_x, self._last_y, _pause=False)
