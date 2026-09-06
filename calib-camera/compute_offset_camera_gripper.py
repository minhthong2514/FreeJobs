import os
import sys
import threading
import time
import numpy as np
import cv2
import serial

# Add path to src directory to use uart_protocol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from uart_protocol import UART

# Path to calibration data files (located in calib-camera/)
PARAMS_PATH = "camera_params.npz"
OFFSET_SAVE_PATH = "camera_offset.npz"

# Global shared variables protected by a Lock
frame_lock = threading.Lock()
latest_frame = None
clicked_pixel_x = None
clicked_pixel_y = None
is_running = True


def click_event(event, x, y, flags, params):
    """Callback function to record mouse click coordinates."""
    global clicked_pixel_x, clicked_pixel_y
    if event == cv2.EVENT_LBUTTONDOWN:
        with frame_lock:
            clicked_pixel_x = x
            clicked_pixel_y = y
        print(f"\n[CAMERA CLICKED] Selected Pixel: X={x}, Y={y}")


def reset_clicked_pixel():
    """Reset clicked pixel coordinates."""
    global clicked_pixel_x, clicked_pixel_y
    with frame_lock:
        clicked_pixel_x = None
        clicked_pixel_y = None
    print("\n[CAMERA] Reset clicked pixel coordinates.")


class CameraStreamThread(threading.Thread):
    """Separate thread for continuously reading camera frames, applying full undistort, and handling GUI display."""

    def __init__(self, K=None, dist=None, device="/dev/video0", width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.pipeline = (
            f"v4l2src device={device} ! "
            f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        
        # Camera intrinsics for full-frame undistort
        self.K = K
        self.dist = dist
        self.newK = None
        if self.K is not None and self.dist is not None:
            self.newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (width, height), 0)

    def run(self):
        global latest_frame, is_running, clicked_pixel_x, clicked_pixel_y

        if not self.cap.isOpened():
            print("\n[CAMERA ERROR] Could not open camera via GStreamer pipeline.")
            print("[CAMERA INFO] Fallback to standard V4L2 device...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[CAMERA FATAL] Failed to initialize camera.")
            return

        window_name = "Live Camera Feed (Undistorted | Click Target Point P | Press 'r' to Reset)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)
        print("\n[CAMERA] GStreamer camera stream thread started successfully.")
        print("[CAMERA INSTRUCTION] Focus on camera window and press 'r' to reset selected point.")

        while is_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # --- UNDISTORT FULL FRAME ---
            if self.K is not None and self.dist is not None:
                frame = cv2.undistort(frame, self.K, self.dist, None, self.newK)

            with frame_lock:
                latest_frame = frame.copy()
                curr_x = clicked_pixel_x
                curr_y = clicked_pixel_y

            # Visual overlay for selected point
            if curr_x is not None and curr_y is not None:
                cv2.circle(frame, (curr_x, curr_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"P({curr_x},{curr_y})",
                            (curr_x + 10, curr_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(window_name, frame)
            
            # Non-blocking key check for OpenCV GUI thread
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):  # Reset clicked point on pressing 'r'
                reset_clicked_pixel()
            elif key == 27 or key == ord('q'):     # ESC or q to exit
                is_running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("[CAMERA] Camera thread stopped.")


def init_serial_uart(port="/dev/ttyUSB0", baudrate=115200):
    """Initialize Serial UART connection."""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        uart_dev = UART(ser, incoming_mailbox=None)
        print(f"[UART] Connected successfully to {port}")
        return uart_dev
    except Exception as e:
        print(f"[UART WARNING] Could not open serial port {port}: {e}")
        print("[UART WARNING] Running in offline mode.")
        return None


def load_camera_parameters(file_path=PARAMS_PATH):
    """Load K, dist, H_tissue, H_bag safely from camera_params.npz."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Parameter file not found at: {file_path}")
        return None

    try:
        data = np.load(file_path)
        keys = list(data.keys())
        print(f"[DEBUG] Available keys in '{file_path}': {keys}")

        K = data["K"] if "K" in data else (data["mtx"] if "mtx" in data else None)
        dist = data["dist"] if "dist" in data else (data["dist_coeff"] if "dist_coeff" in data else None)
        H_tissue = data["H_tissue"] if "H_tissue" in data else (data["H"] if "H" in data else None)
        H_bag = data["H_bag"] if "H_bag" in data else None

        if K is None or dist is None:
            print("[ERROR] Missing intrinsic K or distortion parameters in npz file!")
            return None

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        print(f"[LOAD SUCCESS] Loaded '{file_path}'")
        print(f" -> Focal Length (fx, fy) : ({fx:.2f}, {fy:.2f})")
        print(f" -> Optical Center (cx, cy): ({cx:.2f}, {cy:.2f}) [Calibrated]")

        return {
            "K": K,
            "dist": dist,
            "H_tissue": H_tissue,
            "H_bag": H_bag,
            "cx": cx,
            "cy": cy
        }
    except Exception as e:
        print(f"[ERROR] Failed to load camera parameters: {e}")
        return None


def convert_pixel_to_mm(pixel_x, pixel_y, K, dist, H_matrix, cx, cy):
    """Convert Pixel coordinates to camera space mm using undistortPoints and Homography H."""
    if H_matrix is None:
        print("[ERROR] Homography matrix is None!")
        return None, None

    # 1. Undistort single point
    src_pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    dst_pt = cv2.undistortPoints(src_pt, K, dist, P=K)
    u_clean = dst_pt[0][0][0]
    v_clean = dst_pt[0][0][1]

    # 2. Convert clean pixel to mm using Homography H
    p = np.array([u_clean, v_clean, 1.0], dtype=np.float32)
    P_mm = H_matrix @ p
    P_mm = P_mm[:2] / P_mm[2]

    # 3. Calculate calibrated image center in mm
    center_p = np.array([cx, cy, 1.0], dtype=np.float32)
    center_mm = H_matrix @ center_p
    center_mm = center_mm[:2] / center_mm[2]

    # 4. Relative offset to optical center
    target_mm = P_mm - center_mm
    return float(target_mm[0]), float(target_mm[1])


def compute_single_offset(cam_params):
    """Compute camera-to-gripper offset for a single position/plane."""
    if cam_params is None:
        print("[ERROR] Cannot calculate without valid camera parameters!")
        return

    K = cam_params["K"]
    dist = cam_params["dist"]
    cx = cam_params["cx"]
    cy = cam_params["cy"]

    print("\n==============================================")
    print("   COMPUTE SINGLE CAMERA OFFSET (ONE PLANE)   ")
    print("==============================================")
    print("Select Homography plane to use:")
    print(" 1. Tissue Plane (H_tissue)")
    print(" 2. Bag Plane (H_bag)")
    h_choice = input("Select plane (1-2, default=1): ").strip()

    if h_choice == '2':
        H_mat = cam_params.get("H_bag")
        plane_name = "BAG"
    else:
        H_mat = cam_params.get("H_tissue")
        plane_name = "TISSUE"

    if H_mat is None:
        print(f"[ERROR] Selected Homography matrix for {plane_name} is missing!")
        return

    try:
        # 1. Physical Gripper Position touching reference point P
        print("\n--- [STEP 1: PHYSICAL REFERENCE POINT P] ---")
        grip_x = float(input("Enter Gripper Touch Robot X (mm): "))
        grip_y = float(input("Enter Gripper Touch Robot Y (mm): "))

        # 2. Camera View Position
        print(f"\n--- [STEP 2: CAMERA VIEW AT {plane_name} PLANE] ---")
        cam_bot_x = float(input("Enter Camera View Robot X (mm): "))
        cam_bot_y = float(input("Enter Camera View Robot Y (mm): "))

        with frame_lock:
            cur_x, cur_y = clicked_pixel_x, clicked_pixel_y

        if cur_x is not None and cur_y is not None:
            use_last = input(f"Use last clicked pixel ({cur_x}, {cur_y})? (y/n): ").strip().lower()
            if use_last == 'y':
                pix_x, pix_y = cur_x, cur_y
            else:
                pix_x = float(input("Enter Pixel X: "))
                pix_y = float(input("Enter Pixel Y: "))
        else:
            pix_x = float(input("Enter Pixel X: "))
            pix_y = float(input("Enter Pixel Y: "))

        # 3. Convert pixel to mm
        cam_dx, cam_dy = convert_pixel_to_mm(pix_x, pix_y, K, dist, H_mat, cx, cy)

        if cam_dx is None:
            print("[ERROR] Conversion failed!")
            return

        # 4. Calculate Offset
        offset_x = grip_x - (cam_bot_x + cam_dx)
        offset_y = grip_y - (cam_bot_y + cam_dy)

        # 5. Display result
        print("\n==================================================")
        print(f"       SINGLE OFFSET RESULT ({plane_name} PLANE)   ")
        print("==================================================")
        print(f" Gripper Position : X = {grip_x:.2f} mm, Y = {grip_y:.2f} mm")
        print(f" Camera Position  : X = {cam_bot_x:.2f} mm, Y = {cam_bot_y:.2f} mm")
        print(f" Target Pixel     : ({pix_x}, {pix_y}) -> Cam dx = {cam_dx:.2f} mm, dy = {cam_dy:.2f} mm")
        print("--------------------------------------------------")
        print(f" Calculated Offset X : {offset_x:.2f} mm")
        print(f" Calculated Offset Y : {offset_y:.2f} mm")
        print("==================================================\n")

    except ValueError:
        print("[ERROR] Invalid numerical input!")


def compute_both_offsets_and_save(cam_params, uart_dev=None):
    """Calculate both offsets (Tissue & Bag) with proper robot alignment sequence."""
    global clicked_pixel_x, clicked_pixel_y

    if cam_params is None:
        print("[ERROR] Cannot calculate without valid camera parameters!")
        return

    K = cam_params["K"]
    dist = cam_params["dist"]
    cx = cam_params["cx"]
    cy = cam_params["cy"]
    H_tissue = cam_params.get("H_tissue")
    H_bag = cam_params.get("H_bag")

    print("H_tissue:\n", H_tissue)
    print("H_bag:\n", H_bag)
    if H_tissue is None or H_bag is None:
        print("[ERROR] Missing H_tissue or H_bag in camera_params.npz!")
        return

    print("\n==============================================")
    print("   COMPUTE DUAL CAMERA OFFSETS (TISSUE & BAG) ")
    print("==============================================")

    try:
        # STEP 1: Touch physical reference point P with Gripper
        print("\n--- [STEP 1: PHYSICAL REFERENCE POINT P] ---")
        grip_x = float(input("Enter Gripper Touch Robot X (mm): "))
        grip_y = float(input("Enter Gripper Touch Robot Y (mm): "))

        # STEP 2: Measurement at Z_TISSUE plane (Low Z)
        print("\n--- [STEP 2: MEASUREMENT AT Z_TISSUE (LOW Z)] ---")
        print(">> Move Robot to Camera View position for Tissue plane.")
        if uart_dev:
            move_tissue = input("Do you need to move Robot via UART now? (y/n, default=n): ").strip().lower()
            if move_tissue == 'y':
                uart_dev.request_move_xyz(request=2)

        cam_bot_x_tissue = float(input("Enter Camera View Robot X at Z_tissue (mm): "))
        cam_bot_y_tissue = float(input("Enter Camera View Robot Y at Z_tissue (mm): "))
        
        reset_clicked_pixel()
        input(">> CLICK on point P on live camera stream, then press ENTER in terminal...")
        with frame_lock:
            pix_x_tissue, pix_y_tissue = clicked_pixel_x, clicked_pixel_y

        if pix_x_tissue is None or pix_y_tissue is None:
            print("[WARNING] No pixel clicked! Entering manual fallback:")
            pix_x_tissue = float(input(" Enter Pixel X for Tissue: "))
            pix_y_tissue = float(input(" Enter Pixel Y for Tissue: "))
        else:
            print(f"[SUCCESS] Tissue Pixel Captured: ({pix_x_tissue}, {pix_y_tissue})")

        # STEP 3: Measurement at Z_BAG plane (High Z)
        print("\n--- [STEP 3: MEASUREMENT AT Z_BAG (HIGH Z)] ---")
        print(">> LOGIC: You MUST move the Robot to Z_bag height & position FIRST so camera can see point P!")
        
        if uart_dev:
            print("\n[ROBOT CONTROL] Moving Robot to Z_bag position...")
            uart_dev.request_move_xyz(request=2)
        else:
            input(">> Please manually move Robot to Z_bag height & position, then press ENTER...")

        cam_bot_x_bag = float(input("\nEnter Camera View Robot X at Z_bag (mm): "))
        cam_bot_y_bag = float(input("Enter Camera View Robot Y at Z_bag (mm): "))
        
        reset_clicked_pixel()
        input(">> NOW click on point P on the live camera stream, then press ENTER in terminal...")
        
        with frame_lock:
            pix_x_bag, pix_y_bag = clicked_pixel_x, clicked_pixel_y

        if pix_x_bag is None or pix_y_bag is None:
            print("[WARNING] No pixel clicked! Entering manual fallback:")
            pix_x_bag = float(input(" Enter Pixel X for Bag: "))
            pix_y_bag = float(input(" Enter Pixel Y for Bag: "))
        else:
            print(f"[SUCCESS] Bag Pixel Captured: ({pix_x_bag}, {pix_y_bag})")

        # STEP 4: Convert Pixels to mm & Calculate Offsets
        cam_dx_t, cam_dy_t = convert_pixel_to_mm(pix_x_tissue, pix_y_tissue, K, dist, H_tissue, cx, cy)
        cam_dx_b, cam_dy_b = convert_pixel_to_mm(pix_x_bag, pix_y_bag, K, dist, H_bag, cx, cy)

        if cam_dx_t is None or cam_dx_b is None:
            print("[ERROR] Homography conversion failed!")
            return

        offset_tissue_x = grip_x - (cam_bot_x_tissue + cam_dx_t)
        offset_tissue_y = grip_y - (cam_bot_y_tissue + cam_dy_t)

        offset_bag_x = grip_x - (cam_bot_x_bag + cam_dx_b)
        offset_bag_y = grip_y - (cam_bot_y_bag + cam_dy_b)

        # STEP 5: Display Results & Save Prompt
        print("\n==================================================")
        print("          CALCULATED OFFSETS RESULT               ")
        print("==================================================")
        print(f" Reference Point P (Gripper Touch) : X = {grip_x:.2f} mm, Y = {grip_y:.2f} mm")
        print("--------------------------------------------------")
        print(f" [1] OFFSET_TISSUE (Z_low)  :")
        print(f"     -> Offset X : {offset_tissue_x:.2f} mm | Offset Y : {offset_tissue_y:.2f} mm")
        print(f"     -> Cam delta: dx={cam_dx_t:.2f} mm, dy={cam_dy_t:.2f} mm")
        print("--------------------------------------------------")
        print(f" [2] OFFSET_BAG (Z_high)   :")
        print(f"     -> Offset X : {offset_bag_x:.2f} mm | Offset Y : {offset_bag_y:.2f} mm")
        print(f"     -> Cam delta: dx={cam_dx_b:.2f} mm, dy={cam_dy_b:.2f} mm")
        print("==================================================\n")

        save_choice = input(f"Save BOTH offsets into '{OFFSET_SAVE_PATH}'? (y/n): ").strip().lower()
        if save_choice == 'y':
            np.savez(
                OFFSET_SAVE_PATH,
                offset_tissue=np.array([offset_tissue_x, offset_tissue_y], dtype=np.float32),
                offset_bag=np.array([offset_bag_x, offset_bag_y], dtype=np.float32)
            )
            print(f"[SUCCESS] Saved 'offset_tissue' and 'offset_bag' into '{OFFSET_SAVE_PATH}'")

    except ValueError:
        print("[ERROR] Invalid numerical input!")


def main():
    global is_running

    uart_dev = init_serial_uart(port="/dev/ttyUSB0")
    cam_params = load_camera_parameters(PARAMS_PATH)

    # Extract K and dist if available to pass into camera thread
    K = cam_params["K"] if cam_params else None
    dist = cam_params["dist"] if cam_params else None

    # Start Camera Stream with full frame undistort enabled
    cam_thread = CameraStreamThread(K=K, dist=dist, device="/dev/video0", width=640, height=480, fps=30)
    cam_thread.start()

    # Main control menu loop
    try:
        while is_running:
            print("\n=== CAMERA CALIBRATION & UART CONTROL MENU ===")
            print("1. Request Homing (request_homing - Mode 5)")
            print("2. Move XYZ (request_move_xyz - Mode 2)")
            print("3. Compute SINGLE Offset (One Position/Plane)")
            print("4. Compute DUAL Offsets (Tissue & Bag) & Save")
            print("5. Exit")

            choice = input("Select Option (1-5): ").strip()

            if choice == '1':
                if uart_dev:
                    uart_dev.request_homing(request=5)
                else:
                    print("[UART] Serial not connected!")

            elif choice == '2':
                if uart_dev:
                    uart_dev.request_move_xyz(request=2)
                else:
                    print("[UART] Serial not connected!")

            elif choice == '3':
                compute_single_offset(cam_params)

            elif choice == '4':
                compute_both_offsets_and_save(cam_params, uart_dev)

            elif choice == '5':
                print("Exiting application...")
                is_running = False
                break

    except KeyboardInterrupt:
        print("\nTerminating by user command...")
        is_running = False

    cam_thread.join(timeout=2.0)
    print("Application terminated successfully.")


if __name__ == "__main__":
    main()