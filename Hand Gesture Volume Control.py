import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity("error")
except Exception:
    pass

import sys
import time
import math
import cv2
import numpy as np
import mediapipe as mp

# ---------------- System volume (Windows via Pycaw) ----------------
WINDOWS = (sys.platform.startswith("win"))

_endpoint = None
if WINDOWS:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        _devices = AudioUtilities.GetSpeakers()
        _iface = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _endpoint = cast(_iface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        print("WARNING: Could not initialize Windows audio endpoint via Pycaw:", e)
        _endpoint = None

def set_volume_scalar(x: float):
    if _endpoint is None:
        return
    _endpoint.SetMasterVolumeLevelScalar(float(np.clip(x, 0.0, 1.0)), None)

def get_volume_scalar() -> float:
    if _endpoint is None:
        return 0.0
    return float(_endpoint.GetMasterVolumeLevelScalar())

def set_mute(flag: bool):
    if _endpoint is None:
        return
    _endpoint.SetMute(1 if flag else 0, None)

def is_muted() -> bool:
    if _endpoint is None:
        return False
    try:
        return bool(_endpoint.GetMute())
    except Exception:
        return False

# ---------------- MediaPipe Hands ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---------------- Helpers ----------------
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [2, 6, 10, 14, 18]

def get_handedness_label(results) -> str | None:
    try:
        return results.multi_handedness[0].classification[0].label  # "Left" or "Right"
    except Exception:
        return None

def fingers_up(landmarks, img_w, img_h, handed_label: str | None):
    """Return [thumb, index, middle, ring, pinky] flags and pixel points."""
    pts = [(int(l.x * img_w), int(l.y * img_h)) for l in landmarks]
    fingers = [0, 0, 0, 0, 0]

    # Thumb based on handedness for flipped image
    # If the image is flipped horizontally (selfie view), MediaPipe's handedness
    # still tells the visible hand. For Right hand, thumb tip should be to the RIGHT of thumb IP.
    if handed_label == "Right":
        fingers[0] = 1 if pts[4][0] > pts[2][0] else 0
    elif handed_label == "Left":
        fingers[0] = 1 if pts[4][0] < pts[2][0] else 0
    else:
        # Fallback: compare distance from wrist x
        wrist_x = pts[0][0]
        fingers[0] = 1 if abs(pts[4][0] - wrist_x) > abs(pts[2][0] - wrist_x) else 0

    # Other fingers: tip y < pip y => up
    for i in range(1, 5):
        tip = TIP_IDS[i]; pip = PIP_IDS[i]
        fingers[i] = 1 if pts[tip][1] < pts[pip][1] else 0

    return fingers, pts

def draw_volume_bar(img, vol_pct: int, status_text: str | None = None):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 50, 100, 80, h - 100
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 2)
    vol_pct = int(np.clip(vol_pct, 0, 100))
    fill_h = int((y2 - y1) * vol_pct / 100.0)
    cv2.rectangle(img, (x1+3, y2 - fill_h), (x2-3, y2), (255, 255, 255), -1)
    cv2.putText(img, f"{vol_pct}%", (40, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if status_text:
        cv2.putText(img, status_text, (20, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2, cv2.LINE_AA)

def draw_info(img, top_px, bot_px):
    h, w = img.shape[:2]
    cv2.line(img, (0, top_px), (w, top_px), (200, 200, 200), 1)
    cv2.line(img, (0, bot_px), (w, bot_px), (200, 200, 200), 1)
    cv2.putText(img, "100%", (w - 60, top_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, "0%",   (w - 60, bot_px + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    y = 28
    for ln in ["Finger+Palm Volume",
               "Index-only up → height = volume",
               "Fist: -10%  |  Open palm: +10%",
               "Press 'c' to calibrate top/bottom  |  Esc to quit"]:
        cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
        y += 24

# ---------------- Calibration & smoothing ----------------
TOP_ZONE = 0.15   # relative (0..1) of frame height = 100% volume
BOT_ZONE = 0.85   # relative (0..1) = 0% volume
alpha = 0.15      # smoothing for index-follow mode
smoothed = 0.0

# Step control settings
STEP = 0.10              # 10% per step
STEP_DEBOUNCE = 0.40     # seconds
last_step_time = 0.0

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if WINDOWS else cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

if _endpoint is None and WINDOWS:
    print("WARNING: Volume endpoint not available; UI will show but system volume won't change.")

print("Starting… Index-only = continuous volume; Fist: -10%; Open palm: +10%. 'c' calibrates, Esc quits.")

# Initialize smoothed to current system volume if available
try:
    smoothed = get_volume_scalar()
except Exception:
    smoothed = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    status_text = None
    curr_scalar = get_volume_scalar() if _endpoint is not None else smoothed

    if results.multi_hand_landmarks:
        handed = get_handedness_label(results)
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(120, 255, 120), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(120, 120, 255), thickness=2),
        )
        fup, pts = fingers_up(lm, w, h, handed)
        total_up = sum(fup)
        idx_tip = pts[8]
        cv2.circle(frame, idx_tip, 7, (0, 255, 0), -1)

        now = time.time()

        # Strict fist: all 0
        if total_up == 0:
            if now - last_step_time > STEP_DEBOUNCE:
                new_scalar = max(0.0, curr_scalar - STEP)
                set_volume_scalar(new_scalar)
                smoothed = new_scalar
                last_step_time = now
                status_text = f"Step Down → {int(new_scalar*100)}%"

        # Strict open palm: all four (index..pinky) up (thumb optional)
        elif fup[1] == fup[2] == fup[3] == fup[4] == 1:
            if now - last_step_time > STEP_DEBOUNCE:
                new_scalar = min(1.0, curr_scalar + STEP)
                set_volume_scalar(new_scalar)
                smoothed = new_scalar
                last_step_time = now
                status_text = f"Step Up → {int(new_scalar*100)}%"

        else:
            # Index-only up → continuous control
            only_index_up = (fup[1] == 1) and (fup[0] + fup[2] + fup[3] + fup[4] == 0)
            if only_index_up:
                top_px = int(TOP_ZONE * h)
                bot_px = int(BOT_ZONE * h)
                y = int(np.clip(idx_tip[1], top_px, bot_px))
                # map y -> [1.0 .. 0.0]
                target = 1.0 - (y - top_px) / max(1, (bot_px - top_px))
                smoothed = float(np.clip(alpha * target + (1 - alpha) * smoothed, 0.0, 1.0))
                set_volume_scalar(smoothed)
                status_text = f"Index Control → {int(smoothed*100)}%"

    # Guides & bar
    draw_info(frame, top_px=int(TOP_ZONE * h), bot_px=int(BOT_ZONE * h))
    ui_vol = int((get_volume_scalar() if _endpoint is not None else smoothed) * 100)
    draw_volume_bar(frame, ui_vol, status_text=status_text)

    cv2.imshow("Hand Gesture Volume Control", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and results and results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        idx_y = int(lm[8].y * h)
        # Snap whichever guide is closer
        if abs(idx_y - int(TOP_ZONE * h)) <= abs(idx_y - int(BOT_ZONE * h)):
            TOP_ZONE = float(np.clip((idx_y / h), 0.02, 0.45))
            print(f"Calibrated TOP_ZONE → {TOP_ZONE:.2f} (y={idx_y})")
        else:
            BOT_ZONE = float(np.clip((idx_y / h), 0.55, 0.98))
            print(f"Calibrated BOT_ZONE → {BOT_ZONE:.2f} (y={idx_y})")

    elif key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
