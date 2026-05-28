#!/usr/bin/env python3
"""
Live camera test with lightweight emotion-style detection for IMX219.

Run:
    python3 camera_test.py
"""

import sys
import time
from collections import deque

import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 is not installed.")
    print("Install it with: sudo apt install -y python3-picamera2")
    sys.exit(1)


def classify_state(smile_count: int, eye_count: int, sleepy_frames: int) -> tuple[str, float, tuple[int, int, int]]:
    """
    Return a lightweight 'emotion-like' state.
    This is a heuristic (not a trained emotion model).
    """
    if smile_count > 0:
        return "HAPPY", 0.88, (60, 220, 120)
    if eye_count == 0 and sleepy_frames > 8:
        return "SLEEPY", 0.78, (70, 130, 255)
    if eye_count >= 2:
        return "FOCUSED", 0.74, (80, 200, 255)
    if eye_count == 1:
        return "THINKING", 0.66, (210, 170, 80)
    return "NEUTRAL", 0.60, (180, 180, 180)


def draw_hud(frame: cv2.UMat, label: str, confidence: float, color: tuple[int, int, int], fps: float) -> None:
    """Draw a small creative overlay for state and confidence."""
    h, w = frame.shape[:2]

    # Top translucent band.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 86), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Emotion badge and text.
    cv2.putText(frame, "EMOTION HUD", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"STATE: {label}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    # Confidence bar.
    bar_x, bar_y, bar_w, bar_h = 18, 66, 260, 12
    fill_w = int(bar_w * max(0.0, min(confidence, 1.0)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)


def main() -> int:
    """Initialize camera, run heuristic emotion display, exit on 'q'."""
    picam2 = None

    try:
        # Load built-in Haar cascades for face features.
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

        if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
            print("Error: Failed to load OpenCV Haar cascade files.")
            print("Please ensure python3-opencv is installed correctly.")
            return 1

        # Create camera object (camera index 0 is typical for a single IMX219).
        picam2 = Picamera2(0)

        # Configure a preview stream suitable for live display.
        # format='RGB888' gives 8-bit RGB data that OpenCV can display.
        config = picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()

        # Short warm-up delay lets auto-exposure/auto-white-balance settle.
        time.sleep(0.5)

        print("Camera started with emotion HUD. Press 'q' to quit.")

        frame_times = deque(maxlen=20)
        sleepy_counter = 0

        while True:
            frame_start = time.time()

            # Capture the latest frame as a NumPy array in RGB format.
            frame_rgb = picam2.capture_array()

            if frame_rgb is None:
                print("Warning: received empty frame from camera.")
                continue

            # OpenCV expects BGR for correct color display.
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90)
            )

            if len(faces) > 0:
                # Use the biggest face when multiple faces are present.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi_gray = gray[y : y + h, x : x + w]
                face_roi_bgr = frame_bgr[y : y + h, x : x + w]

                eyes = eye_cascade.detectMultiScale(
                    face_roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
                )
                smiles = smile_cascade.detectMultiScale(
                    face_roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(30, 30)
                )

                if len(eyes) == 0:
                    sleepy_counter += 1
                else:
                    sleepy_counter = max(0, sleepy_counter - 2)

                label, conf, color = classify_state(len(smiles), len(eyes), sleepy_counter)

                # Face box with colored border.
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

                # Draw eye/smile hints for visual feedback.
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(face_roi_bgr, (ex, ey), (ex + ew, ey + eh), (255, 200, 60), 1)
                for (sx, sy, sw, sh) in smiles[:1]:
                    cv2.rectangle(face_roi_bgr, (sx, sy), (sx + sw, sy + sh), (70, 255, 120), 1)
            else:
                label, conf, color = ("NO FACE", 0.0, (120, 120, 120))
                sleepy_counter = 0

            frame_times.append(time.time() - frame_start)
            avg_dt = sum(frame_times) / len(frame_times) if frame_times else 0.0
            fps = (1.0 / avg_dt) if avg_dt > 0 else 0.0

            draw_hud(frame_bgr, label, conf, color, fps)
            cv2.imshow("IMX219 Live Feed + Emotion", frame_bgr)

            # Exit when user presses 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 0
    except Exception as exc:
        print(f"Camera error: {exc}")
        print("Check camera cable, camera interface setting, and package installation.")
        return 1
    finally:
        # Always release resources cleanly.
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
