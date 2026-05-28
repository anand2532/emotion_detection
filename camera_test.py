#!/usr/bin/env python3
"""
Real-time emotion display demo for Raspberry Pi + IMX219 + OpenCV.

Run:
    python3 camera_test.py

Install (Raspberry Pi OS):
    sudo apt update
    sudo apt install -y python3-picamera2 python3-opencv rpicam-apps

Optional (for model-based emotion inference):
    python3 -m pip install tensorflow-lite-runtime
"""

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 missing. Install: sudo apt install -y python3-picamera2")
    sys.exit(1)


EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Surprise"]
EMO_STYLES = {
    "Happy": {"emoji": ":)", "color": (80, 220, 120)},
    "Sad": {"emoji": ":(", "color": (220, 120, 80)},
    "Angry": {"emoji": ">:(", "color": (70, 70, 255)},
    "Neutral": {"emoji": ":|", "color": (200, 200, 200)},
    "Surprise": {"emoji": ":O", "color": (90, 200, 255)},
    "No Face": {"emoji": "...", "color": (130, 130, 130)},
}


@dataclass
class DetectorBundle:
    face: cv2.CascadeClassifier
    eyes: cv2.CascadeClassifier
    smile: cv2.CascadeClassifier


def setup_logging() -> None:
    """Basic logger for runtime diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def troubleshooting_hint(section: str) -> None:
    """Print concise troubleshooting hints for common RPi camera issues."""
    hints = {
        "camera_not_detected": (
            "Camera not detected: run 'rpicam-hello --list-cameras', reseat CSI cable, "
            "and verify correct Pi 5 camera connector/cable adapter."
        ),
        "black_screen": (
            "Black screen: check lens cap/cable seating, reduce resolution, "
            "and verify camera works in rpicam-hello."
        ),
        "imshow_fail": (
            "cv2.imshow() failed: likely no GUI display. If using SSH, use VNC/desktop session "
            "or set DISPLAY correctly."
        ),
        "missing_deps": (
            "Missing dependency/model: install python3-opencv/python3-picamera2 and ensure "
            "cascade/model files exist."
        ),
        "low_fps": (
            "Low FPS: reduce resolution to 640x480, avoid heavy models, and close other apps."
        ),
        "permission": (
            "Permission issue: ensure user belongs to video group and no other process holds camera."
        ),
        "tf_cv_conflict": (
            "TensorFlow/OpenCV conflict: avoid mixing incompatible wheel/apt builds; "
            "prefer apt OpenCV and tflite-runtime on RPi."
        ),
        "ssh_vnc": (
            "SSH/VNC display issue: run script inside desktop session or use VNC with DISPLAY set."
        ),
    }
    logging.warning("Troubleshooting: %s", hints.get(section, "General camera diagnostic needed."))


def get_haarcascade_dir() -> str:
    """
    Resolve Haar cascade directory across OpenCV builds.
    Some distro builds do not expose cv2.data.
    """
    # Preferred path when available.
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None:
        cascade_dir = getattr(cv2_data, "haarcascades", "")
        if cascade_dir and os.path.isdir(cascade_dir):
            return cascade_dir

    # Optional override for custom/manual cascade directory.
    custom_dir = os.environ.get("HAAR_CASCADE_DIR", "")
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir if custom_dir.endswith("/") else custom_dir + "/"

    # Common distro fallback paths (Raspberry Pi OS / Debian variants).
    fallback_dirs = [
        "/usr/share/opencv4/haarcascades/",
        "/usr/share/opencv/haarcascades/",
        "/usr/local/share/opencv4/haarcascades/",
        "/usr/local/share/opencv/haarcascades/",
    ]
    for candidate in fallback_dirs:
        if os.path.isdir(candidate):
            return candidate

    # Python site-package fallback: .../cv2/data/haarcascade_*.xml
    cv2_file = getattr(cv2, "__file__", "")
    if cv2_file:
        candidate = Path(cv2_file).resolve().parent / "data"
        if candidate.is_dir():
            return str(candidate) + "/"

    raise RuntimeError(
        "Could not find Haar cascades directory.\n"
        "Try:\n"
        "  sudo apt install -y python3-opencv opencv-data\n"
        "or set custom path:\n"
        "  export HAAR_CASCADE_DIR=/path/to/haarcascades"
    )


def load_detectors() -> DetectorBundle:
    """Load Haar cascades for lightweight face-feature analysis."""
    cascade_dir = get_haarcascade_dir()
    face_xml = cascade_dir + "haarcascade_frontalface_default.xml"
    eyes_xml = cascade_dir + "haarcascade_eye.xml"
    smile_xml = cascade_dir + "haarcascade_smile.xml"

    if not (os.path.exists(face_xml) and os.path.exists(eyes_xml) and os.path.exists(smile_xml)):
        troubleshooting_hint("missing_deps")
        raise RuntimeError(
            "Haar cascade XML files not found.\n"
            f"Checked directory: {cascade_dir}\n"
            "Install data files:\n"
            "  sudo apt install -y opencv-data\n"
            "Then retry."
        )

    face = cv2.CascadeClassifier(face_xml)
    eyes = cv2.CascadeClassifier(eyes_xml)
    smile = cv2.CascadeClassifier(smile_xml)
    if face.empty() or eyes.empty() or smile.empty():
        troubleshooting_hint("missing_deps")
        raise RuntimeError("Failed to load OpenCV Haar cascade files.")
    return DetectorBundle(face=face, eyes=eyes, smile=smile)


def setup_camera() -> Picamera2:
    """Initialize IMX219 camera stream via picamera2."""
    try:
        picam2 = Picamera2(0)
        config = picam2.create_preview_configuration(
            main={"size": (960, 540), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)
        return picam2
    except Exception as exc:
        troubleshooting_hint("camera_not_detected")
        troubleshooting_hint("permission")
        raise RuntimeError(f"Camera setup failed: {exc}") from exc


def detect_face(gray_frame, detectors: DetectorBundle):
    """Return largest detected face rectangle (x, y, w, h) or None."""
    faces = detectors.face.detectMultiScale(
        gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def predict_emotion(face_gray, detectors: DetectorBundle, no_eye_streak: int):
    """
    Predict emotion with lightweight heuristics.

    Note:
    - This is not a clinically accurate model.
    - For better quality, plug in a trained model (TFLite/ONNX) here.
    """
    eyes = detectors.eyes.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
    smiles = detectors.smile.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(30, 30))

    eye_count = len(eyes)
    smile_count = len(smiles)

    # Approximate wide-eye indicator for "Surprise".
    eye_area_ratio = 0.0
    if eye_count > 0:
        total_eye_area = sum(int(w * h) for (_, _, w, h) in eyes[:2])
        face_area = max(1, face_gray.shape[0] * face_gray.shape[1])
        eye_area_ratio = total_eye_area / face_area

    if smile_count > 0:
        return "Happy", 0.90, eye_count
    if eye_count >= 2 and eye_area_ratio > 0.05:
        return "Surprise", 0.82, eye_count
    if no_eye_streak > 8:
        return "Sad", 0.72, eye_count
    if eye_count >= 2:
        return "Neutral", 0.70, eye_count
    return "Angry", 0.64, eye_count


def draw_overlay(frame, face_box, emotion: str, confidence: float, fps: float) -> None:
    """Render creative overlays: face box, emoji, label, and confidence."""
    h, w = frame.shape[:2]
    style = EMO_STYLES.get(emotion, EMO_STYLES["Neutral"])
    color = style["color"]
    emoji = style["emoji"]

    # Top translucent HUD panel.
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 92), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.35, frame, 0.65, 0, frame)

    # Title + metrics.
    cv2.putText(frame, "Emotion Vision", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{emoji}  {emotion}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

    # Confidence progress bar.
    bar_x, bar_y, bar_w, bar_h = w - 290, 50, 260, 14
    fill_w = int(max(0.0, min(confidence, 1.0)) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # Face box if available.
    if face_box is not None:
        x, y, fw, fh = face_box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.putText(
            frame,
            f"{emotion} {emoji}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def check_display_environment() -> None:
    """Log likely GUI issues before calling cv2.imshow()."""
    if os.environ.get("DISPLAY", "") == "":
        troubleshooting_hint("ssh_vnc")
        logging.warning("DISPLAY is empty. GUI window may not open.")


def main() -> int:
    """Main loop: camera capture, face detection, emotion prediction, overlay."""
    setup_logging()
    check_display_environment()

    picam2 = None
    frame_times = deque(maxlen=20)
    no_eye_streak = 0
    empty_frame_streak = 0

    try:
        logging.info("Loading detectors...")
        detectors = load_detectors()

        logging.info("Starting camera...")
        picam2 = setup_camera()
        logging.info("Press 'q' in preview window to quit.")

        while True:
            t0 = time.time()
            frame_rgb = picam2.capture_array()

            if frame_rgb is None:
                empty_frame_streak += 1
                if empty_frame_streak > 10:
                    troubleshooting_hint("black_screen")
                    logging.error("Too many empty frames from camera.")
                    return 1
                continue

            empty_frame_streak = 0
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_box = detect_face(gray, detectors)
            if face_box is None:
                emotion = "No Face"
                confidence = 0.0
                no_eye_streak = 0
            else:
                x, y, w, h = face_box
                face_gray = gray[y : y + h, x : x + w]
                emotion, confidence, eye_count = predict_emotion(face_gray, detectors, no_eye_streak)
                no_eye_streak = no_eye_streak + 1 if eye_count == 0 else max(0, no_eye_streak - 2)

            frame_times.append(time.time() - t0)
            avg_dt = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            if fps < 10:
                troubleshooting_hint("low_fps")

            draw_overlay(frame, face_box, emotion, confidence, fps)

            try:
                cv2.imshow("IMX219 Emotion Detection", frame)
            except cv2.error:
                troubleshooting_hint("imshow_fail")
                return 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        return 0
    except Exception as exc:
        logging.error("Runtime error: %s", exc)
        troubleshooting_hint("tf_cv_conflict")
        return 1
    finally:
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
