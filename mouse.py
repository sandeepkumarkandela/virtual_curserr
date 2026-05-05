import cv2
import numpy as np
import time
import os
import sys
import pyautogui
import ctypes
from ctypes import wintypes
from datetime import datetime
from math import hypot

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

_FINGER_TIPS = (8, 12, 16, 20)
_FINGER_PIPS = (6, 10, 14, 18)

VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT)]
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]

_SendInput = ctypes.windll.user32.SendInput
_SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
_SendInput.restype = wintypes.UINT


def _send_key(vk_code):
    inp_down = _INPUT()
    inp_down.type = INPUT_KEYBOARD
    inp_down.union.ki.wVk = vk_code
    inp_down.union.ki.dwFlags = 0

    inp_up = _INPUT()
    inp_up.type = INPUT_KEYBOARD
    inp_up.union.ki.wVk = vk_code
    inp_up.union.ki.dwFlags = KEYEVENTF_KEYUP

    arr = (_INPUT * 2)(inp_down, inp_up)
    _SendInput(2, arr, ctypes.sizeof(_INPUT))


def _fast_interp(val, in_low, in_high, out_low, out_high):
    if val <= in_low:
        return out_low
    if val >= in_high:
        return out_high
    return out_low + (val - in_low) * (out_high - out_low) / (in_high - in_low)


def download_model():
    if os.path.exists(MODEL_PATH):
        return True
    print("Downloading hand_landmarker.task model...")
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        print("Please download manually from:")
        print("  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"And place it at: {MODEL_PATH}")
        return False


def find_distance(lmList, p1, p2, img):
    x1, y1 = lmList[p1][1], lmList[p1][2]
    x2, y2 = lmList[p2][1], lmList[p2][2]
    cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1

    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

    length = hypot(x2 - x1, y2 - y1)
    return length, img, (x1, y1, x2, y2, cx, cy)


def fingers_up(lmList):
    thumb_tip_x = lmList[4][1]
    thumb_ip_x = lmList[3][1]
    thumb_mcp_x = lmList[2][1]
    index_mcp_x = lmList[5][1]
    if thumb_mcp_x < index_mcp_x:
        thumb = 1 if thumb_tip_x < thumb_ip_x else 0
    else:
        thumb = 1 if thumb_tip_x > thumb_ip_x else 0

    index  = 1 if lmList[8][2]  < lmList[6][2]  else 0
    middle = 1 if lmList[12][2] < lmList[10][2] else 0
    ring   = 1 if lmList[16][2] < lmList[14][2] else 0
    pinky  = 1 if lmList[20][2] < lmList[18][2] else 0

    return (thumb, index, middle, ring, pinky)


def set_brightness(level):
    try:
        import subprocess
        level = max(0, min(100, int(level)))
        cmd = f'(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})'
        subprocess.run(['powershell', '-Command', cmd],
                       capture_output=True, timeout=3)
        return True
    except Exception:
        return False


SCREENSHOT_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "GestureScreenshots")


def take_screenshot(save_dir=None):
    if save_dir is None:
        save_dir = SCREENSHOT_FOLDER
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"gesture_screenshot_{timestamp}.png")
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        print(f"[OK] Screenshot saved to: {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to take screenshot: {e}")
        return None
    return filepath


def draw_hud(img, mode_text, color=(0, 255, 255), extra_info=None):
    h, w = img.shape[:2]

    roi = img[h - 80:h, 0:w]
    dark = np.full_like(roi, 30, dtype=np.uint8)
    cv2.addWeighted(dark, 0.6, roi, 0.4, 0, roi)

    cv2.circle(img, (30, h - 40), 14, color, cv2.FILLED)
    cv2.circle(img, (30, h - 40), 14, (255, 255, 255), 2)

    cv2.putText(img, mode_text, (55, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if extra_info:
        cv2.putText(img, extra_info, (55, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


def draw_cooldown_indicator(img, x, y, progress, radius=20):
    angle = int(progress * 360)
    cv2.ellipse(img, (x, y), (radius, radius), -90, 0, angle, (0, 255, 0), 3)
    if progress >= 1.0:
        cv2.circle(img, (x, y), radius - 5, (0, 255, 0), cv2.FILLED)


def main():
    if not download_model():
        sys.exit(1)

    latest_landmarks = []
    _frame_ts = [0]

    def _on_result(result, output_image, timestamp_ms):
        nonlocal latest_landmarks
        if result.hand_landmarks:
            latest_landmarks = result.hand_landmarks
        else:
            latest_landmarks = []

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=_on_result,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    wCam, hCam = 640, 480
    DETECT_EVERY_N = 2
    frameR = 40
    smoothening = 5
    inv_smooth = 1.0 / smoothening

    pTime = 0
    plocX, plocY = 0.0, 0.0
    clocX, clocY = 0.0, 0.0

    previous_vol = 50

    previous_brightness = 50

    scroll_prev_y = 0
    scroll_cooldown = 0

    click_cooldown = 0.0
    right_click_cooldown = 0.0
    CLICK_COOLDOWN_TIME = 0.4
    RIGHT_CLICK_COOLDOWN_TIME = 0.5

    last_click_time = 0.0
    DOUBLE_CLICK_THRESHOLD = 0.5

    screenshot_cooldown = 0.0
    SCREENSHOT_COOLDOWN_TIME = 2.0

    is_dragging = False
    drag_cooldown = 0.0
    DRAG_COOLDOWN_TIME = 0.3

    current_mode = "IDLE"
    mode_color = (128, 128, 128)
    extra_info_text = None

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    print("Opening webcam...")
    cap = None
    for attempt in range(3):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        print(f"  Attempt {attempt + 1}/3 failed, retrying...")
        time.sleep(1)

    if cap is None or not cap.isOpened():
        print("ERROR: Could not open webcam after 3 attempts.")
        print("Make sure no other application is using the camera.")
        detector.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Webcam opened successfully!")

    wScr, hScr = pyautogui.size()

    edge_overshoot = 20
    x_map_in_low = frameR
    x_map_in_high = wCam - frameR
    y_map_in_low = frameR
    y_map_in_high = hCam - frameR
    x_map_out_low = -edge_overshoot
    x_map_out_high = wScr + edge_overshoot
    y_map_out_low = -edge_overshoot
    y_map_out_high = hScr + edge_overshoot

    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    _finger_names = ('T', 'I', 'M', 'R', 'P')

    print("=" * 50)
    print("      AI Virtual Mouse - Optimized Edition")
    print("=" * 50)
    print()
    print("  GESTURE CONTROLS:")
    print("  " + "─" * 41)
    print("  Index finger ONLY          → Move mouse")
    print("  Index + Middle UP          → Left click (close together)")
    print("  Middle finger ONLY         → Right click")
    print("  Index + Middle (tap twice) → Double click")
    print("  Thumb + Index (pinch)      → Volume control")
    print("  Thumb + Pinky UP           → Brightness control")
    print("  Pinky finger ONLY          → Scroll (move hand up/down)")
    print("  All 5 fingers UP (palm)    → Screenshot")
    print("  Ring + Pinky UP            → Drag & Drop toggle")
    print("  " + "─" * 41)
    print("  Press ESC to quit")
    print("=" * 50)

    lmList = [[i, 0, 0] for i in range(21)]

    frame_count = 0

    while cap.isOpened():
        current_time = time.time()

        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        h, w, c = img.shape

        if frame_count % DETECT_EVERY_N == 0:
            small = cv2.resize(img, (320, 240))
            frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            _frame_ts[0] += 1
            try:
                detector.detect_async(mp_image, _frame_ts[0])
            except Exception:
                pass

        has_hand = False
        if latest_landmarks:
            hand_landmarks = latest_landmarks[0]
            for idx, lm in enumerate(hand_landmarks):
                lmList[idx][1] = int(lm.x * w)
                lmList[idx][2] = int(lm.y * h)
            has_hand = True

            for start, end in HAND_CONNECTIONS:
                pt1 = (lmList[start][1], lmList[start][2])
                pt2 = (lmList[end][1], lmList[end][2])
                cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            for pt in lmList:
                cv2.circle(img, (pt[1], pt[2]), 3, (255, 0, 0), cv2.FILLED)

        if has_hand:
            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]

            fings = fingers_up(lmList)

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            if fings == (1, 1, 1, 1, 1):
                current_mode = "SCREENSHOT"
                mode_color = (0, 200, 255)
                print(f"[GESTURE] Screenshot gesture detected! Fingers: {fings}")

                if current_time - screenshot_cooldown > SCREENSHOT_COOLDOWN_TIME:
                    filepath = take_screenshot()
                    screenshot_cooldown = current_time
                    if filepath:
                        extra_info_text = f"Saved: {os.path.basename(filepath)}"
                    else:
                        extra_info_text = "Screenshot FAILED - check console"

                    cv2.addWeighted(np.full_like(img, 255, dtype=np.uint8), 0.5, img, 0.5, 0, img)
                else:
                    remaining = SCREENSHOT_COOLDOWN_TIME - (current_time - screenshot_cooldown)
                    extra_info_text = f"Cooldown: {remaining:.1f}s"

                progress = min(1.0, (current_time - screenshot_cooldown) / SCREENSHOT_COOLDOWN_TIME)
                draw_cooldown_indicator(img, w - 40, h - 40, progress)

            elif fings[3] == 1 and fings[4] == 1 and fings[1] == 0 and fings[2] == 0:
                current_mode = "DRAG & DROP"
                mode_color = (255, 165, 0)

                if current_time - drag_cooldown > DRAG_COOLDOWN_TIME:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                        extra_info_text = "Dragging started - move with index finger"
                        print("Drag started!")
                    else:
                        pyautogui.mouseUp()
                        is_dragging = False
                        extra_info_text = "Dropped!"
                        print("Drop completed!")
                    drag_cooldown = current_time
                else:
                    extra_info_text = "Dragging..." if is_dragging else "Ready (ring+pinky to start drag)"

            elif fings[1] == 1 and fings[2] == 0 and fings[3] == 0 and fings[4] == 0:
                current_mode = "MOVE"
                mode_color = (255, 0, 255)

                x3 = _fast_interp(x1, x_map_in_low, x_map_in_high, x_map_out_low, x_map_out_high)
                y3 = _fast_interp(y1, y_map_in_low, y_map_in_high, y_map_out_low, y_map_out_high)

                clocX = plocX + (x3 - plocX) * inv_smooth
                clocY = plocY + (y3 - plocY) * inv_smooth

                clocX = max(0, min(wScr - 1, clocX))
                clocY = max(0, min(hScr - 1, clocY))

                extra_info_text = f"Pos: ({int(clocX)}, {int(clocY)})"

                try:
                    pyautogui.moveTo(int(clocX), int(clocY))
                except Exception:
                    pass

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            elif fings[1] == 1 and fings[2] == 1 and fings[3] == 0 and fings[4] == 0 and fings[0] == 0:
                length, img, lineInfo = find_distance(lmList, 8, 12, img)

                if length < 40:
                    if current_time - click_cooldown > CLICK_COOLDOWN_TIME:
                        if current_time - last_click_time < DOUBLE_CLICK_THRESHOLD:
                            pyautogui.doubleClick()
                            current_mode = "DOUBLE CLICK"
                            mode_color = (0, 255, 255)
                            extra_info_text = "Double clicked!"
                            print("Double click!")
                            last_click_time = 0.0
                        else:
                            pyautogui.click()
                            current_mode = "LEFT CLICK"
                            mode_color = (0, 255, 0)
                            extra_info_text = "Click! (tap again quickly for double-click)"
                            last_click_time = current_time

                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        click_cooldown = current_time
                    else:
                        current_mode = "CLICK"
                        mode_color = (0, 255, 0)
                        remaining = CLICK_COOLDOWN_TIME - (current_time - click_cooldown)
                        extra_info_text = f"Cooldown: {remaining:.1f}s"
                else:
                    current_mode = "CLICK READY"
                    mode_color = (0, 200, 0)
                    extra_info_text = f"Bring fingers closer to click (dist: {int(length)})"

                plocX, plocY = 0, 0

            elif fings[2] == 1 and fings[1] == 0 and fings[0] == 0 and fings[3] == 0 and fings[4] == 0:
                current_mode = "RIGHT CLICK"
                mode_color = (0, 165, 255)

                if current_time - right_click_cooldown > RIGHT_CLICK_COOLDOWN_TIME:
                    pyautogui.rightClick()
                    right_click_cooldown = current_time
                    extra_info_text = "Right clicked!"
                    print("Right click!")
                    cv2.circle(img, (x2, y2), 18, (0, 165, 255), cv2.FILLED)
                else:
                    remaining = RIGHT_CLICK_COOLDOWN_TIME - (current_time - right_click_cooldown)
                    extra_info_text = f"Cooldown: {remaining:.1f}s"

            elif fings[0] == 1 and fings[1] == 1 and fings[2] == 0 and fings[3] == 0 and fings[4] == 0:
                current_mode = "VOLUME"
                mode_color = (255, 255, 0)

                length, img, lineInfo = find_distance(lmList, 4, 8, img)

                vol = int(_fast_interp(length, 20, 200, 0, 100))

                diff = vol - previous_vol
                if diff < 0:
                    vk = VK_VOLUME_DOWN
                    count = min(-diff, 10)
                elif diff > 0:
                    vk = VK_VOLUME_UP
                    count = min(diff, 10)
                else:
                    count = 0
                    vk = 0

                for _ in range(count):
                    _send_key(vk)
                if count > 0:
                    previous_vol = vol

                extra_info_text = f"Volume: {vol}%"

                bar_x, bar_y, bar_w, bar_h = 20, 130, 30, 200
                vol_height = int(_fast_interp(vol, 0, 100, 0, bar_h))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), cv2.FILLED)
                cv2.rectangle(img, (bar_x, bar_y + bar_h - vol_height),
                              (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.putText(img, f'{vol}%', (bar_x - 5, bar_y + bar_h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, 'VOL', (bar_x, bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            elif fings[0] == 1 and fings[4] == 1 and fings[1] == 0 and fings[2] == 0 and fings[3] == 0:
                current_mode = "BRIGHTNESS"
                mode_color = (0, 255, 255)

                length, img, lineInfo = find_distance(lmList, 4, 20, img)

                brightness = int(_fast_interp(length, 30, 250, 0, 100))

                if abs(brightness - previous_brightness) > 3:
                    set_brightness(brightness)
                    previous_brightness = brightness

                extra_info_text = f"Brightness: {brightness}%"

                bar_x, bar_y, bar_w, bar_h = w - 50, 130, 30, 200
                bright_height = int(_fast_interp(brightness, 0, 100, 0, bar_h))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), cv2.FILLED)
                cv2.rectangle(img, (bar_x, bar_y + bar_h - bright_height),
                              (bar_x + bar_w, bar_y + bar_h), (0, 255, 255), cv2.FILLED)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.putText(img, f'{brightness}%', (bar_x - 10, bar_y + bar_h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(img, 'BRT', (bar_x, bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            elif fings[4] == 1 and fings[0] == 0 and fings[1] == 0 and fings[2] == 0 and fings[3] == 0:
                current_mode = "SCROLL"
                mode_color = (255, 128, 0)

                pinky_y = lmList[20][2]

                if scroll_prev_y == 0:
                    scroll_prev_y = pinky_y
                else:
                    delta_y = scroll_prev_y - pinky_y

                    if abs(delta_y) > 8:
                        scroll_amount = delta_y // 3
                        pyautogui.scroll(scroll_amount)
                        scroll_prev_y = pinky_y

                        if delta_y > 0:
                            extra_info_text = "Scrolling UP ↑"
                            cv2.arrowedLine(img, (w // 2, h // 2 + 30),
                                            (w // 2, h // 2 - 30), (255, 128, 0), 3)
                        else:
                            extra_info_text = "Scrolling DOWN ↓"
                            cv2.arrowedLine(img, (w // 2, h // 2 - 30),
                                            (w // 2, h // 2 + 30), (255, 128, 0), 3)
                    else:
                        extra_info_text = "Move hand up/down to scroll"

                cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (255, 128, 0), cv2.FILLED)

            else:
                current_mode = "IDLE"
                mode_color = (128, 128, 128)
                extra_info_text = None
                scroll_prev_y = 0

                if is_dragging and fings == (0, 0, 0, 0, 0):
                    current_mode = "DRAGGING"
                    mode_color = (255, 165, 0)
                    extra_info_text = "Dragging... (ring+pinky to drop)"

        else:
            current_mode = "NO HAND"
            mode_color = (0, 0, 255)
            extra_info_text = "Show your hand to the camera"
            scroll_prev_y = 0

            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                print("Drag cancelled - hand lost")

        img = draw_hud(img, current_mode, mode_color, extra_info_text)

        if is_dragging:
            cv2.rectangle(img, (w - 120, 10), (w - 10, 40), (0, 100, 255), cv2.FILLED)
            cv2.putText(img, "DRAG ON", (w - 115, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cTime = current_time
        if pTime > 0:
            fps = 1.0 / (cTime - pTime)
            cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        pTime = cTime

        if has_hand:
            for i, (name, up) in enumerate(zip(_finger_names, fings)):
                color = (0, 255, 0) if up else (0, 0, 200)
                cx = w - 150 + i * 28
                cy = 70
                cv2.circle(img, (cx, cy), 12, color, cv2.FILLED)
                cv2.putText(img, name, (cx - 5, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("AI Virtual Mouse - Optimized", img)
        if cv2.waitKey(1) == 27:
            break

        frame_count += 1

    if is_dragging:
        pyautogui.mouseUp()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()