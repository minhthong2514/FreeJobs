import cv2
import numpy as np

H_data = np.load(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/Homoraphy_value.npz"
)
H = H_data["H"]
params = np.load(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz"
)
K = params["K"]
dist = params["dist"]

origin_world = None        # (X0, Y0) in mm
display_img = None         # image shown on screen
base_img = None            # clean image for reset

clicked_points = []
# ==============================
# Pixel -> World using Homography
# ==============================
def pixel_to_world(x, y, H):
    """
    Convert pixel coordinate (x, y)
    to world coordinate (X, Y) in mm
    using homography matrix H
    """
    p = np.array([x, y, 1.0])
    P = H @ p
    return P[0] / P[2], P[1] / P[2]

# ==============================
# Mouse Callback
# ==============================
def mouse_callback(event, x, y, flags, param):
    global origin_world, clicked_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    X, Y = pixel_to_world(x, y, H)

    # First click → set origin
    if origin_world is None:
        origin_world = (X, Y)

        clicked_points.append({
            "pixel": (x, y),
            "world": (0.0, 0.0),
            "is_origin": True
        })

        print(f"Origin set at: ({X:.2f}, {Y:.2f})mm")
        return

    # Other points
    Xr = X - origin_world[0]
    Yr = origin_world[1] - Y

    clicked_points.append({
        "pixel": (x, y),
        "world": (Xr, Yr),
        "is_origin": False
    })

    print(f"Pixel ({x},{y}) -> World ({Xr:.2f},{Yr:.2f})mm")

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Convert pixel to world coordinate
    X, Y = pixel_to_world(x, y, H)

    # ------------------------------
    # First click: set origin
    # ------------------------------
    if origin_world is None:
        origin_world = (X, Y)
        print(f"Origin set at: ({X:.2f}, {Y:.2f})mm")

        cv2.circle(display_img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            display_img,
            "ORIGIN (0, 0) mm",
            (x + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        return

    # ------------------------------
    # Other points: relative to origin
    # ------------------------------
    Xr = X - origin_world[0]
    Yr = origin_world[1] - Y    # Cartesian coordinate (Y up)

    cv2.circle(display_img, (x, y), 4, (0, 0, 255), -1)
    cv2.putText(
        display_img,
        f"({Xr:.2f}, {Yr:.2f})mm",
        (x + 8, y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

def draw_points(img):
    for pt in clicked_points:
        x, y = pt["pixel"]
        X, Y = pt["world"]

        if pt["is_origin"]:
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                img,
                "ORIGIN (0,0)mm",
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
                0.5,
                (0, 0, 255),
                1
            )