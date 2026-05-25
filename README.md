# Hand Gester

A real-time hand gesture recognition tool for macOS using your webcam. It detects up to two hands using MediaPipe's 21-keypoint hand model and classifies each hand into one of seven gestures, drawing an annotated overlay with the gesture name, hand skeleton, bounding box, and a live FPS counter.

## Recognized gestures

| Gesture     | Description                                        |
|-------------|----------------------------------------------------|
| Open Palm   | All five fingers extended                          |
| Fist        | All fingers folded                                 |
| Thumbs Up   | Thumb pointing upward, other fingers folded        |
| Peace       | Index and middle fingers extended (V-sign)         |
| Pointing    | Index finger only extended                         |
| OK          | Middle/ring/pinky up, thumb pinched to index tip   |
| Unknown     | Anything that doesn't match the above              |

## macOS setup

> **Important:** MediaPipe 0.10.x requires Python **3.9–3.12**. macOS ships Python 3.13+ by default on newer systems, which is not yet supported by MediaPipe. Use Homebrew to install 3.12.

```bash
# 1. Install Python 3.12 via Homebrew
brew install python@3.12

# 2. Create a virtual environment with Python 3.12
python3.12 -m venv .venv

# 3. Activate the environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python main.py
```

## CLI flags

| Flag               | Default | Description                                  |
|--------------------|---------|----------------------------------------------|
| `--camera INT`     | `0`     | Camera index                                 |
| `--no-mirror`      | off     | Disable horizontal flip (selfie-view is on by default) |
| `--max-hands INT`  | `2`     | Maximum number of hands to track             |
| `--width INT`      | `1280`  | Capture width in pixels                      |
| `--height INT`     | `720`   | Capture height in pixels                     |
| `--min-confidence FLOAT` | `0.7` | Minimum detection confidence (0–1)       |

## Troubleshooting

**Black / blank window**
macOS blocks camera access until permission is granted. Go to **System Settings → Privacy & Security → Camera** and enable access for Terminal (or your IDE).

**`ModuleNotFoundError: No module named 'mediapipe'`**
You are likely using the wrong Python interpreter. Confirm your venv is active (`which python` should point inside `.venv/`) and that it uses Python 3.12.

**Wrong camera / Continuity Camera issues**
If you have an iPhone Continuity Camera it often appears as index 1 or 2. Try `python main.py --camera 1`.

**Low FPS / laggy**
Close other camera-using applications. Reduce `--width` and `--height` (e.g., `--width 640 --height 480`).

## Running tests

```bash
pip install pytest
pytest
```

## Future work

- Replace the rule-based classifier with a trained MLP or lightweight CNN on recorded landmark sequences for more robust gesture recognition.
- Map gestures to system actions (media controls, window management, mouse pointer).
- Add a recording mode to capture labeled landmark sequences for building a custom training dataset.
