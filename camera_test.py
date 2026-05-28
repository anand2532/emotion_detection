#!/usr/bin/env python3
"""
Robust real-time emotion detection for Raspberry Pi camera feed.

Run:
    python3 camera_test.py

Install (Raspberry Pi OS, recommended):
    sudo apt update
    sudo apt install -y python3-picamera2 python3-opencv python3-numpy rpicam-apps
    python3 -m pip install --upgrade tflite-runtime

If tflite-runtime is unavailable for your OS:
    python3 -m pip install --upgrade tensorflow
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: Missing picamera2. Install with: sudo apt install -y python3-picamera2")
    sys.exit(1)

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:  # pylint: disable=broad-except
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except Exception:  # pylint: disable=broad-except
        Interpreter = None  # type: ignore


# =========================
# Config / Constants
# =========================
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI = {"Angry": ">:(", "Happy": ":)", "Neutral": ":|", "Sad": ":(", "Surprise": ":O", "Unknown": "?"}
COLORS = {
    "Angry": (50, 50, 255),
    "Happy": (80, 230, 120),
    "Neutral": (220, 220, 220),
    "Sad": (230, 150, 60),
    "Surprise": (90, 210, 255),
    "Unknown": (140, 140, 140),
}

MODEL_DIR = Path.home() / ".cache" / "rpi_emotion_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FACE_PROTOTXT = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
EMOTION_MODEL = MODEL_DIR / "emotion_ferplus.tflite"

MODEL_URLS = {
    "face_prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "face_model": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    # Primary and backup URLs. If unavailable, script will fallback to heuristic mode.
    "emotion_tflite": [
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion_ferplus.tflite",
        "https://raw.githubusercontent.com/AmrElsersy/Emotions-Recognition/main/models/emotion_ferplus.tflite",
    ],
}

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
DETECT_WIDTH = 480
DETECT_HEIGHT = 270
FACE_CONF_THRESH = 0.35
MIN_FACE_SIZE = 24  # allows small/far faces
DETECTION_INTERVAL = 2  # run face detector every N frames
TRACK_DISTANCE_THRESH = 80.0
TRACK_MISS_LIMIT = 20
SMOOTHING_ALPHA = 0.28
LABEL_SWITCH_MARGIN = 0.08
LOW_CONF_THRESH = 0.45
FPS_WARN = 10.0


@dataclass
class FaceTrack:
    """State for each tracked face across frames."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    misses: int = 0
    stable_label: str = "Unknown"
    stable_conf: float = 0.0
    probs: np.ndarray = field(default_factory=lambda: np.zeros(len(EMOTIONS), dtype=np.float32))


class EmotionEngine:
    """Encapsulates model loading, detection, inference, tracking, and rendering."""

    def __init__(self) -> None:
        self.face_net = None
        self.emotion_interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = (1, 64, 64, 1)
        self.use_model = False
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 1
        self.last_boxes: List[Tuple[int, int, int, int]] = []
        self.frame_count = 0
        self.fps_samples: deque[float] = deque(maxlen=30)
        self.low_fps_warned = False
        self.haar_fallback = cv2.CascadeClassifier(self._resolve_haar_path("haarcascade_frontalface_default.xml"))

    @staticmethod
    def _resolve_haar_path(name: str) -> str:
        cv2_data = getattr(cv2, "data", None)
        if cv2_data and getattr(cv2_data, "haarcascades", ""):
            p = Path(cv2_data.haarcascades) / name
            if p.exists():
                return str(p)
        for base in ["/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades"]:
            p = Path(base) / name
            if p.exists():
                return str(p)
        return name

    @staticmethod
    def _download(url: str, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < 1024:
            raise RuntimeError(f"Downloaded file too small: {url}")
        tmp.replace(path)

    def ensure_models(self) -> None:
        """Auto-download required model files with graceful fallback behavior."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not FACE_PROTOTXT.exists():
            logging.info("Downloading face prototxt...")
            self._download(MODEL_URLS["face_prototxt"], FACE_PROTOTXT)
        if not FACE_MODEL.exists():
            logging.info("Downloading face detector model...")
            self._download(MODEL_URLS["face_model"], FACE_MODEL)

        if not EMOTION_MODEL.exists():
            for url in MODEL_URLS["emotion_tflite"]:
                try:
                    logging.info("Trying emotion model download: %s", url)
                    self._download(url, EMOTION_MODEL)
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("Emotion model download failed: %s", exc)

    def load_models(self) -> None:
        """Load OpenCV DNN face model and optional TFLite emotion model."""
        self.ensure_models()
        self.face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTOTXT), str(FACE_MODEL))
        # On Raspberry Pi, OpenCV default backend is often best with packaged builds.
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        if Interpreter is None:
            logging.warning("tflite-runtime/tensorflow not found. Using heuristic emotion fallback.")
            return
        if not EMOTION_MODEL.exists():
            logging.warning("Emotion model not found. Using heuristic emotion fallback.")
            return

        self.emotion_interpreter = Interpreter(model_path=str(EMOTION_MODEL))
        self.emotion_interpreter.allocate_tensors()
        self.input_details = self.emotion_interpreter.get_input_details()
        self.output_details = self.emotion_interpreter.get_output_details()
        self.input_shape = tuple(self.input_details[0]["shape"])
        self.use_model = True
        logging.info("Loaded TFLite emotion model.")

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """Improve low-light visibility and local contrast."""
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(yuv)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        y = clahe.apply(y)
        merged = cv2.merge([y, cr, cb])
        enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        gamma = 1.15
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(enhanced, lut)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        DNN face detection with small-face support.
        Uses resized detect frame for speed and maps boxes back.
        """
        if self.face_net is None:
            return self._haar_detect(frame)

        detect_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            detect_frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.face_net.setInput(blob)
        dets = self.face_net.forward()
        boxes: List[Tuple[int, int, int, int]] = []
        sx = frame.shape[1] / float(DETECT_WIDTH)
        sy = frame.shape[0] / float(DETECT_HEIGHT)

        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < FACE_CONF_THRESH:
                continue
            x1 = int(dets[0, 0, i, 3] * DETECT_WIDTH * sx)
            y1 = int(dets[0, 0, i, 4] * DETECT_HEIGHT * sy)
            x2 = int(dets[0, 0, i, 5] * DETECT_WIDTH * sx)
            y2 = int(dets[0, 0, i, 6] * DETECT_HEIGHT * sy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            w = x2 - x1
            h = y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            boxes.append((x1, y1, w, h))

        # Fallback for difficult small/low-light cases.
        if not boxes:
            boxes = self._haar_detect(frame)
        return boxes

    def _haar_detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_fallback.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(22, 22))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def _prepare_emotion_input(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face ROI to match TFLite model input shape and dtype."""
        _, h, w, c = self.input_shape
        target_h = int(h)
        target_w = int(w)
        if c == 1:
            roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_AREA)
            roi = roi.astype(np.float32) / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))
        else:
            roi = cv2.resize(face_roi, (target_w, target_h), interpolation=cv2.INTER_AREA)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            roi = np.expand_dims(roi, axis=0)
        return roi

    def _predict_heuristic(self, face_roi: np.ndarray) -> np.ndarray:
        """Fallback probability vector when emotion model is unavailable."""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        probs = np.array([0.20, 0.20, 0.30, 0.20, 0.10], dtype=np.float32)  # Angry,Happy,Neutral,Sad,Surprise
        if mean_intensity > 145 and std_intensity > 40:
            probs = np.array([0.08, 0.45, 0.20, 0.07, 0.20], dtype=np.float32)
        elif mean_intensity < 90:
            probs = np.array([0.16, 0.10, 0.24, 0.42, 0.08], dtype=np.float32)
        return probs / np.sum(probs)

    def predict_emotion_probs(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Return probability vector for one face."""
        x, y, w, h = bbox
        pad = int(min(w, h) * 0.18)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return np.zeros(len(EMOTIONS), dtype=np.float32)

        if not self.use_model or self.emotion_interpreter is None:
            return self._predict_heuristic(face_roi)

        try:
            inp = self._prepare_emotion_input(face_roi)
            input_index = int(self.input_details[0]["index"])
            output_index = int(self.output_details[0]["index"])
            in_dtype = self.input_details[0]["dtype"]
            if in_dtype == np.uint8:
                inp = (inp * 255.0).astype(np.uint8)
            self.emotion_interpreter.set_tensor(input_index, inp)
            self.emotion_interpreter.invoke()
            out = self.emotion_interpreter.get_tensor(output_index).squeeze().astype(np.float32)
            # Handle logits or arbitrary outputs.
            out = out - np.max(out)
            exp = np.exp(out)
            probs = exp / np.maximum(np.sum(exp), 1e-7)
            if probs.size != len(EMOTIONS):
                return self._predict_heuristic(face_roi)
            return probs
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Emotion inference error; using fallback: %s", exc)
            return self._predict_heuristic(face_roi)

    @staticmethod
    def _centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = box
        return (x + (w / 2.0), y + (h / 2.0))

    def update_tracks(self, boxes: List[Tuple[int, int, int, int]]) -> Dict[int, FaceTrack]:
        """Simple centroid-based matching for multi-face consistency."""
        used_track_ids = set()
        assignments: Dict[int, Tuple[int, int, int, int]] = {}

        for box in boxes:
            c = self._centroid(box)
            best_id = None
            best_dist = 1e9
            for tid, track in self.tracks.items():
                if tid in used_track_ids:
                    continue
                dist = math.hypot(c[0] - track.centroid[0], c[1] - track.centroid[1])
                if dist < best_dist and dist <= TRACK_DISTANCE_THRESH:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = FaceTrack(track_id=tid, bbox=box, centroid=c)
                used_track_ids.add(tid)
                assignments[tid] = box
            else:
                track = self.tracks[best_id]
                track.bbox = box
                track.centroid = c
                track.misses = 0
                used_track_ids.add(best_id)
                assignments[best_id] = box

        for tid, track in list(self.tracks.items()):
            if tid not in assignments:
                track.misses += 1
                if track.misses > TRACK_MISS_LIMIT:
                    del self.tracks[tid]

        return {tid: self.tracks[tid] for tid in assignments}

    def smooth_prediction(self, track: FaceTrack, probs: np.ndarray) -> Tuple[str, float]:
        """EMA + hysteresis to avoid flicker."""
        if track.probs.sum() == 0:
            track.probs = probs
        else:
            track.probs = (1.0 - SMOOTHING_ALPHA) * track.probs + SMOOTHING_ALPHA * probs

        best_idx = int(np.argmax(track.probs))
        best_conf = float(track.probs[best_idx])
        candidate = EMOTIONS[best_idx]

        if track.stable_label == "Unknown":
            if best_conf >= LOW_CONF_THRESH:
                track.stable_label = candidate
                track.stable_conf = best_conf
        elif candidate != track.stable_label:
            current_idx = EMOTIONS.index(track.stable_label) if track.stable_label in EMOTIONS else best_idx
            current_conf = float(track.probs[current_idx])
            if best_conf - current_conf > LABEL_SWITCH_MARGIN and best_conf >= LOW_CONF_THRESH:
                track.stable_label = candidate
                track.stable_conf = best_conf
        else:
            track.stable_conf = best_conf

        if best_conf < LOW_CONF_THRESH:
            return "Unknown", best_conf
        return track.stable_label, track.stable_conf

    def draw_ui(self, frame: np.ndarray, states: Dict[int, Tuple[str, float]]) -> None:
        """Draw creative overlay for all faces + global HUD."""
        h, w = frame.shape[:2]
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.35, frame, 0.65, 0, frame)

        fps = self.current_fps()
        cv2.putText(frame, "Emotion Vision RPi", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(states)}", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)

        for tid, track in self.tracks.items():
            if tid not in states:
                continue
            label, conf = states[tid]
            color = COLORS.get(label, COLORS["Unknown"])
            emoji = EMOJI.get(label, "?")
            x, y, bw, bh = track.bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            text = f"#{tid} {emoji} {label} {conf * 100:.1f}%"
            cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    def current_fps(self) -> float:
        if not self.fps_samples:
            return 0.0
        avg = sum(self.fps_samples) / len(self.fps_samples)
        return 1.0 / avg if avg > 0 else 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Main per-frame processing."""
        t0 = time.time()
        enhanced = self.preprocess_frame(frame)

        if self.frame_count % DETECTION_INTERVAL == 0:
            self.last_boxes = self.detect_faces(enhanced)
        self.frame_count += 1

        active_tracks = self.update_tracks(self.last_boxes)
        states: Dict[int, Tuple[str, float]] = {}
        for tid, track in active_tracks.items():
            probs = self.predict_emotion_probs(enhanced, track.bbox)
            states[tid] = self.smooth_prediction(track, probs)

        self.draw_ui(enhanced, states)
        self.fps_samples.append(time.time() - t0)
        fps = self.current_fps()
        if fps < FPS_WARN and len(self.fps_samples) == self.fps_samples.maxlen and not self.low_fps_warned:
            logging.warning("Low FPS detected (%.1f). Lower FRAME_* or increase DETECTION_INTERVAL.", fps)
            self.low_fps_warned = True
        return enhanced


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def troubleshooting_hint(code: str) -> None:
    hints = {
        "camera_not_opening": "Camera not opening: check cable, rpicam-hello --list-cameras, and camera permissions.",
        "model_loading_failure": "Model loading failure: verify internet for auto-download and tflite-runtime install.",
        "low_fps": "Low FPS: reduce FRAME_WIDTH/HEIGHT and increase DETECTION_INTERVAL.",
        "display_issue": "cv2 display issue: ensure GUI session; for SSH use VNC or set DISPLAY.",
        "tf_conflict": "TensorFlow/OpenCV conflict: prefer apt OpenCV + tflite-runtime. Avoid mixed ABI builds.",
    }
    logging.error("%s", hints.get(code, "Unknown issue"))


def check_environment() -> None:
    if not os.environ.get("DISPLAY"):
        logging.warning("DISPLAY not set. cv2.imshow may fail in SSH-only session.")


def setup_camera() -> Picamera2:
    try:
        cam = Picamera2(0)
        cfg = cam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
        cam.configure(cfg)
        cam.start()
        time.sleep(0.4)
        return cam
    except Exception as exc:  # pylint: disable=broad-except
        troubleshooting_hint("camera_not_opening")
        raise RuntimeError(f"Failed to open camera: {exc}") from exc


def main() -> int:
    setup_logging()
    check_environment()

    cam = None
    try:
        engine = EmotionEngine()
        try:
            engine.load_models()
        except Exception as exc:  # pylint: disable=broad-except
            troubleshooting_hint("model_loading_failure")
            logging.warning("Continuing with fallback logic: %s", exc)

        cam = setup_camera()
        logging.info("Running emotion detection. Press 'q' to exit.")

        empty_count = 0
        while True:
            frame_rgb = cam.capture_array()
            if frame_rgb is None:
                empty_count += 1
                if empty_count > 10:
                    logging.error("Black screen / empty frames from camera.")
                    break
                continue
            empty_count = 0

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            output = engine.process(frame_bgr)

            try:
                cv2.imshow("IMX219 Emotion Detection", output)
            except cv2.error:
                troubleshooting_hint("display_issue")
                return 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if engine.current_fps() < FPS_WARN:
            troubleshooting_hint("low_fps")
        return 0

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        troubleshooting_hint("tf_conflict")
        logging.exception("Fatal runtime error: %s", exc)
        return 1
    finally:
        if cam is not None:
            try:
                cam.stop()
                cam.close()
            except Exception:  # pylint: disable=broad-except
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
