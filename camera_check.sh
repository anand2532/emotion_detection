#!/usr/bin/env bash

# Camera connection check script for Raspberry Pi camera modules (e.g., IMX219).
# - Detects available camera utility (rpicam-hello or libcamera-hello)
# - Lists connected cameras
# - Optionally opens a short preview test

set -u

echo "=== Raspberry Pi Camera Connection Check ==="

print_dmesg_diagnostics() {
  echo
  echo "=== dmesg camera diagnostics (latest matches) ==="
  echo "(If this is empty, try running with sudo for full kernel logs.)"
  dmesg | grep -Ei "imx219|camera|csi|rp1|unicam|libcamera" | tail -n 40 || true
  echo "=== end diagnostics ==="
}

# Pick camera utility based on what is installed.
CAM_CMD=""
if command -v rpicam-hello >/dev/null 2>&1; then
  CAM_CMD="rpicam-hello"
elif command -v libcamera-hello >/dev/null 2>&1; then
  CAM_CMD="libcamera-hello"
fi

if [[ -z "$CAM_CMD" ]]; then
  echo "ERROR: No camera utility found."
  echo "Install one of the following based on your OS:"
  echo "  sudo apt install -y rpicam-apps      # Trixie/Bookworm"
  echo "  sudo apt install -y libcamera-apps   # Bullseye"
  exit 1
fi

echo "Using command: $CAM_CMD"
echo
echo "Checking for connected cameras..."

LIST_OUTPUT="$($CAM_CMD --list-cameras 2>&1)"
LIST_CODE=$?

echo "$LIST_OUTPUT"
echo

if [[ $LIST_CODE -ne 0 ]]; then
  echo "ERROR: Camera command failed."
  echo "Try:"
  echo "  - Re-seat the CSI ribbon cable (power off first)"
  echo "  - Check cable orientation and correct CSI connector/adapter"
  echo "  - Update system packages"
  print_dmesg_diagnostics
  exit 1
fi

# Heuristic: if output contains "No cameras available", treat as failure.
if echo "$LIST_OUTPUT" | grep -qi "No cameras available"; then
  echo "FAIL: No camera detected."
  echo "Check cable orientation/port and reboot."
  print_dmesg_diagnostics
  exit 2
fi

echo "PASS: Camera appears to be detected."
echo
read -r -p "Run a 5-second preview test now? (y/N): " RUN_PREVIEW

if [[ "$RUN_PREVIEW" =~ ^[Yy]$ ]]; then
  echo "Starting 5-second preview..."
  # For both commands, -t expects milliseconds.
  $CAM_CMD -t 5000 >/dev/null 2>&1
  PREVIEW_CODE=$?

  if [[ $PREVIEW_CODE -eq 0 ]]; then
    echo "Preview test completed successfully."
    exit 0
  fi

  echo "Preview test failed (exit code: $PREVIEW_CODE)."
  exit 3
fi

echo "Done."
exit 0
