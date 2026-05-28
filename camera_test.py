#!/usr/bin/env python3
"""
Simple live camera test for Waveshare IMX219 on Raspberry Pi OS.

Run:
    python3 camera_test.py
"""

import sys
import time

import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 is not installed.")
    print("Install it with: sudo apt install -y python3-picamera2")
    sys.exit(1)


def main() -> int:
    """Initialize camera, show live preview, and exit on 'q'."""
    picam2 = None

    try:
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

        print("Camera started. Press 'q' in the preview window to quit.")

        while True:
            # Capture the latest frame as a NumPy array in RGB format.
            frame_rgb = picam2.capture_array()

            if frame_rgb is None:
                print("Warning: received empty frame from camera.")
                continue

            # OpenCV expects BGR for correct color display.
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("IMX219 Live Feed", frame_bgr)

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
