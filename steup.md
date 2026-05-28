# Camera Setup Guide (Waveshare IMX219)

## 1) Update system packages

```bash
sudo apt update
sudo apt upgrade -y
```

## 2) Install required Python packages

Use Raspberry Pi OS packages for best compatibility:

```bash
sudo apt install -y python3-picamera2 python3-opencv
```

## 3) Enable the camera interface

```bash
sudo raspi-config
```

Then go to:

- `Interface Options`
- `Camera`
- `Enable`

Reboot after enabling:

```bash
sudo reboot
```

## 4) Verify camera detection (optional but recommended)

```bash
libcamera-hello
```

If a preview opens, the camera is detected correctly.

## 5) Run the test script

From the project folder:

```bash
python3 camera_test.py
```

- A window named `IMX219 Live Feed` will open.
- Press `q` in the preview window to exit cleanly.
