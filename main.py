import argparse

from handgester import Camera, HandDetector, Pipeline, RuleBasedClassifier
from handgester.actions import MouseController


def main():
    parser = argparse.ArgumentParser(description="Real-time hand gesture recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                        help="Disable mirror/flip (default: mirrored)")
    parser.add_argument("--max-hands", type=int, default=2,
                        help="Maximum number of hands to track (default: 2)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum detection confidence (default: 0.7)")
    parser.add_argument("--control-mouse", action="store_true",
                        help="Map gestures to mouse actions (pointing=move, fist=click, OK=right-click). "
                             "macOS: requires Accessibility permission for your terminal.")
    parser.set_defaults(mirror=True)
    args = parser.parse_args()

    print("Press q or ESC to quit")
    if args.control_mouse:
        print("Mouse control: ON")
        print("  POINTING = move | FIST = grab/drag | OK = right-click | PEACE = scroll")
        print("  Press q or ESC in the preview window to quit.")

    camera = Camera(index=args.camera, width=args.width, height=args.height)
    detector = HandDetector(
        max_hands=args.max_hands,
        min_detection_confidence=args.min_confidence,
    )
    classifier = RuleBasedClassifier()
    mouse = MouseController() if args.control_mouse else None
    pipeline = Pipeline(camera, detector, classifier, mirror=args.mirror, mouse=mouse)

    try:
        pipeline.run()
    finally:
        camera.release()
        detector.close()


if __name__ == "__main__":
    main()
