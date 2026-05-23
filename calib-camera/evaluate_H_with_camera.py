import cv2
import numpy as np

# ==============================
# Load Camera params
# ==============================
Camera_params = np.load(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz"
)
H = Camera_params["H"]
K = Camera_params["K"]
newK = Camera_params["newK"]
dist = Camera_params["dist"]


# ==============================
# Global Variables
# ==============================
origin_world = None        # (X0, Y0) in mm
display_img = None         # image shown on screen
base_img = None            # clean image for reset

# ==============================
# Pixel -> World using Homography
# ==============================
def pixel_to_world(x, y, H):
    p = np.array([x, y, 1.0])
    P = H @ p
    return P[0] / P[2], P[1] / P[2]

# ==============================
# Mouse Callback
# ==============================
def mouse_callback(event, x, y, flags, param):
    global origin_world, clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert clicked pixel to world mm
        X, Y = pixel_to_world(x, y, H)

        if origin_world is None:
            origin_world = (X, Y)
            clicked_points.append({
                "pixel": (x, y),
                "world": (0.0, 0.0), # Origin is 0,0
                "is_origin": True
            })
            print(f"Origin set at: ({X:.2f}, {Y:.2f}) mm")
        else:
            Xr = X - origin_world[0]
            Yr = origin_world[1] - Y  
            clicked_points.append({
                "pixel": (x, y),
                "world": (Xr, Yr),
                "is_origin": False
            })
            print(f"Pixel ({x},{y}) -> World relative: ({Xr:.2f}, {Yr:.2f}) mm")
def draw_points(img):
    for pt in clicked_points:
        x, y = pt["pixel"]
        X, Y = pt["world"]

        if pt["is_origin"]:
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                img,
                "ORIGIN (0,0) mm",
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        else:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"({X:.2f},{Y:.2f})mm",
                (x + 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

# ==============================
# Open Camera
# ==============================
pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Pixel to World")
cv2.setMouseCallback("Pixel to World", mouse_callback)
clicked_points = []   # list of dict

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Undistort the live frame
    undist_img = cv2.undistort(frame, K, dist, None, newK)
    
    # 2. Create a copy for display so we don't mess up the original coordinates
    display_img = undist_img.copy()

    # 3. Draw all stored points on the current frame
    draw_points(display_img)

    cv2.imshow("Pixel to World", display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        origin_world = None
        clicked_points.clear()
        print("Origin and points reset")


cap.release()
cv2.destroyAllWindows()
