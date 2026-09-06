import cv2
import numpy as np

# Get calibrate params
params = np.load("/home/minhthong/Desktop/code/farmbot/calib-camera/camera_intrinsic.npz")
K = params['K']
dist = params['dist']
rvecs = params['rvecs']
tvecs = params['tvecs']

# Display value of K and dist
print(f"\nK: \n{K}")
print(f"\nDistortion: \n{dist}")

# Helper function to compute Homography matrix
def compute_homography_single(img_path, checkerboard, squares_w, squares_h, label="Target"):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load image from: {img_path}")
        return None, None, None

    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)

    # Undistort the image
    undist_img = cv2.undistort(img, K, dist, None, newK)

    cols, rows = checkerboard
    # Create world points
    world_pts = np.zeros((cols * rows, 2), dtype=np.float32)
    world_pts[:, 0] = np.tile(np.arange(cols), rows) * squares_w
    world_pts[:, 1] = np.repeat(np.arange(rows), cols) * squares_h

    # Find corners
    ret, corners = cv2.findChessboardCorners(undist_img, checkerboard)

    if ret:
        # Refine corner coordinates for sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Reshape from (N,1,2) to (N,2) 
        pixels_pts = corners2.reshape(-1, 2)

        # Calculate HOMOGRAPHY matrix
        H, mask = cv2.findHomography(pixels_pts, world_pts, cv2.RANSAC, 5.0)

        if H[0][0] < 0:
            print(f"[FIX] Đảo ngược thứ tự corners cho {label} để đồng nhất hệ tọa độ...")
            pixels_pts = pixels_pts[::-1] 
            H, mask = cv2.findHomography(pixels_pts, world_pts, cv2.RANSAC, 5.0)
        
        # Verify calibration accuracy
        errors = []
        for i in range(len(pixels_pts)):
            p = np.array([pixels_pts[i][0], pixels_pts[i][1], 1.0])
            P = H @ p
            P /= P[2]
            err = np.linalg.norm(P[:2] - world_pts[i])
            errors.append(err)
        print(f"\n--- Accuracy for {label} ---")
        print(f"Mean error: {np.mean(errors):.4f} mm")
        print(f"Max error: {np.max(errors):.4f} mm")
        
        return H, undist_img, newK
    else:
        print(f"[ERROR] Chessboard corners NOT found in: {label}")
        return None, None, None

# Checkerboard info for BAG (Z = 0)
CHECKERBOARD_BAG = (14, 8)
squares_width_size_bag = 197.4 / 15
squares_height_size_bag = 118.3 / 9

# Checkerboard info for TISSUE (Z = 140)
CHECKERBOARD_TISSUE = (15, 10)
squares_width_size_tissue = 85.8 / 16.0
squares_height_size_tissue = 59.0 / 11.0

# Load images and calculate Homography for both levels
img_bag_path = '/home/minhthong/Desktop/code/farmbot/calib-camera/images/homo/homo-bag.jpg'
img_tissue_path = '/home/minhthong/Desktop/code/farmbot/calib-camera/images/homo/homo-tissue.jpg'

H_bag, undist_img_bag, newK_bag = compute_homography_single(
    img_bag_path, CHECKERBOARD_BAG, squares_width_size_bag, squares_height_size_bag, label="BAG (High Z)"
)

H_tissue, undist_img_tissue, newK_tissue = compute_homography_single(
    img_tissue_path, CHECKERBOARD_TISSUE, squares_width_size_tissue, squares_height_size_tissue, label="TISSUE (Low Z)"
)

if H_bag is None or H_tissue is None:
    print("[CRITICAL] Failed to compute homography matrices. Exiting.")
    exit()

# Function to convert Pixel (u, v) to World (X, Y)
def pixel_to_world(x, y, H):
    p = np.array([x, y, 1.0])
    P = H @ p
    return P[0] / P[2], P[1] / P[2]

# --- 1. VERIFY CALIBRATION FOR BAG (High Z) ---
def mouse_callback_bag(event, x, y, flags, param):
    global origin_world_bag, undist_img_bag

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert clicked pixel to world mm
        X, Y = pixel_to_world(x, y, H_bag)

        # Set the origin point (0,0) on first click
        if origin_world_bag is None:
            origin_world_bag = (X, Y)
            print(f"[BAG] Origin set at: ({X:.2f}, {Y:.2f}) mm")

            cv2.circle(undist_img_bag, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(undist_img_bag, "ORIGIN(0,0)mm", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        else:
            # Calculate relative distance from the established origin            
            Xr = X - origin_world_bag[0]
            Yr = origin_world_bag[1] - Y  
            print(f"[BAG] Pixel ({x},{y}) -> World relative: ({Xr:.2f}, {Yr:.2f}) mm")

            # Draw point and labels on the image
            cv2.circle(undist_img_bag, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(undist_img_bag, f"({Xr:.2f},{Yr:.2f}) mm", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.namedWindow("Pixel to World - BAG")
cv2.setMouseCallback("Pixel to World - BAG", mouse_callback_bag)
origin_world_bag = None
base_img_bag = undist_img_bag.copy()

print("\n>>> Testing BAG (High Z) Homography Matrix. Click to test, press 'q' to proceed...")
while True:
    cv2.imshow("Pixel to World - BAG", undist_img_bag)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):   # press q to exit
        break

    if key == ord('r'):
        origin_world_bag = None
        undist_img_bag = base_img_bag.copy() 
        print("[BAG] Reset Origin + points.")

cv2.destroyAllWindows()


# --- 2. VERIFY CALIBRATION FOR TISSUE (Low Z) ---
def mouse_callback_tissue(event, x, y, flags, param):
    global origin_world_tissue, undist_img_tissue

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert clicked pixel to world mm
        X, Y = pixel_to_world(x, y, H_tissue)

        # Set the origin point (0,0) on first click
        if origin_world_tissue is None:
            origin_world_tissue = (X, Y)
            print(f"[TISSUE] Origin set at: ({X:.2f}, {Y:.2f}) mm")

            cv2.circle(undist_img_tissue, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(undist_img_tissue, "ORIGIN(0,0)mm", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        else:
            # Calculate relative distance from the established origin            
            Xr = X - origin_world_tissue[0]
            Yr = origin_world_tissue[1] - Y  
            print(f"[TISSUE] Pixel ({x},{y}) -> World relative: ({Xr:.2f}, {Yr:.2f}) mm")

            # Draw point and labels on the image
            cv2.circle(undist_img_tissue, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(undist_img_tissue, f"({Xr:.2f},{Yr:.2f}) mm", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.namedWindow("Pixel to World - TISSUE")
cv2.setMouseCallback("Pixel to World - TISSUE", mouse_callback_tissue)
origin_world_tissue = None
base_img_tissue = undist_img_tissue.copy()

print("\n>>> Testing TISSUE (Low Z) Homography Matrix. Click to test, press 'q' to finish...")
while True:
    cv2.imshow("Pixel to World - TISSUE", undist_img_tissue)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):   # press q to exit
        break

    if key == ord('r'):
        origin_world_tissue = None
        undist_img_tissue = base_img_tissue.copy() 
        print("[TISSUE] Reset Origin + points.")

cv2.destroyAllWindows()


# Save parameters
np.savez("/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz",
    H_bag = H_bag,
    H_tissue = H_tissue,
    K=K,
    newK = newK_tissue,
    dist=dist,
    rvecs=rvecs,
    tvecs=tvecs,
)
print("\nFinal parameters saved successfully.")