# MonitorTracking.py (Pupil-only gaze tracking with calibration)

import cv2
import numpy as np
import time, math, threading
import pyautogui

# =========================
# Screen and mouse control
# =========================
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X, CENTER_Y = MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2
mouse_control_enabled = False
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

def mouse_mover():
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)

threading.Thread(target=mouse_mover, daemon=True).start()

# ============
# Calibration
# ============
calibration_offset_yaw = 0.0
calibration_offset_pitch = 0.0
latest_gaze_vec = None
FOCAL_Z = -500.0

def build_gaze_vector(model_center, pupil_center, focal_z=FOCAL_Z):
    dx = float(pupil_center[0] - model_center[0])
    dy = float(pupil_center[1] - model_center[1])
    return np.array([dx, dy, focal_z], dtype=np.float32)

def convert_gaze_to_screen_coordinates(gaze_vec, offset_yaw, offset_pitch):
    reference_forward = np.array([0, 0, -1], dtype=np.float32)
    gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)

    # yaw
    xz_proj = np.array([gaze_vec[0], 0, gaze_vec[2]])
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if gaze_vec[0] < 0: yaw_rad = -yaw_rad

    # pitch
    yz_proj = np.array([0, gaze_vec[1], gaze_vec[2]])
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if gaze_vec[1] > 0: pitch_rad = -pitch_rad

    yaw_deg   = math.degrees(yaw_rad) + offset_yaw
    pitch_deg = math.degrees(pitch_rad) + offset_pitch

    yaw_range, pitch_range = 15, 10
    screen_x = int(((yaw_deg + yaw_range) / (2 * yaw_range)) * MONITOR_WIDTH)
    screen_y = int(((pitch_range - pitch_deg) / (2 * pitch_range)) * MONITOR_HEIGHT)

    return max(0, min(screen_x, MONITOR_WIDTH-1)), max(0, min(screen_y, MONITOR_HEIGHT-1)), yaw_deg, pitch_deg

# ============================
# Pupil detection (simplified)
# ============================
ray_lines, stored_intersections, model_centers = [], [], []
prev_model_center_avg = (CENTER_X//2, CENTER_Y//2)

def crop_to_aspect_ratio(image, width=640, height=480):
    h, w = image.shape[:2]
    desired = width / height
    current = w / h
    if current > desired:
        new_w = int(desired * h)
        offset = (w - new_w) // 2
        cropped = image[:, offset:offset + new_w]
    else:
        new_h = int(w / desired)
        offset = (h - new_h) // 2
        cropped = image[offset:offset + new_h, :]
    return cv2.resize(cropped, (width, height))

def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, _, min_loc, _ = cv2.minMaxLoc(gray)
    return min_loc

def process_frame(frame, render=True):
    global latest_gaze_vec, prev_model_center_avg

    frame = crop_to_aspect_ratio(frame, 640, 480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest = get_darkest_area(frame)
    _, thresh = cv2.threshold(gray, gray[darkest[1], darkest[0]]+15, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            pupil_center = tuple(map(int, ellipse[0]))
            model_center = prev_model_center_avg  # fallback: use last avg
            prev_model_center_avg = model_center

            # Build gaze vector
            gaze_vec = build_gaze_vector(model_center, pupil_center)
            latest_gaze_vec = gaze_vec

            # Map to screen
            screen_x, screen_y, yaw, pitch = convert_gaze_to_screen_coordinates(
                gaze_vec, calibration_offset_yaw, calibration_offset_pitch
            )

            if mouse_control_enabled:
                with mouse_lock:
                    mouse_target[0], mouse_target[1] = screen_x, screen_y

            if render:
                cv2.ellipse(frame, ellipse, (0,255,255), 2)
                cv2.circle(frame, model_center, 5, (255,0,0), -1)
                cv2.circle(frame, pupil_center, 5, (0,0,255), -1)
                cv2.putText(frame, f"({screen_x},{screen_y})", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame

# ============
# Main loop
# ============
def main():
    global mouse_control_enabled, calibration_offset_yaw, calibration_offset_pitch
    stream_url = "http://192.168.104.16:81/stream"  # Change to your camera stream URL
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'f' to toggle mouse control, 's' to calibrate center, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_out = process_frame(frame, render=True)
        cv2.imshow("Pupil Gaze Tracking", frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            mouse_control_enabled = not mouse_control_enabled
            print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
            time.sleep(0.2)
        elif key == ord('s'):
            if latest_gaze_vec is not None:
                _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(latest_gaze_vec, 0, 0)
                calibration_offset_yaw   = -raw_yaw
                calibration_offset_pitch = -raw_pitch
                print(f"[Calibrated] yaw_offset={calibration_offset_yaw:.2f}, pitch_offset={calibration_offset_pitch:.2f}")
            else:
                print("[Calibration] No gaze vector available.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()