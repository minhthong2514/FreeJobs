import cv2
import numpy as np
import glob

# Checkerboard size (number of inner corners)
CHECKERBOARD = (15, 10)  # change if your pattern is different

# Square size in mm
square_size = 5.3333  # 5.3333 mm

# Create 3D points for the real-world plane (Z = 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Scale by square size -> real-world coordinates
objp *= square_size  # mm

# Arrays to store 3D (world) points and 2D (image) points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Load chessboard images
images = glob.glob(r'/home/minhthong/Desktop/code/farmbot/calib-camera/images/*.jpg')

img_shape = None
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_shape is None:
        img_shape = gray.shape[::-1]  # (width, height)

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)

        # Refine corner detection (subpixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Draw corners only if found
        img_draw = img.copy()
        cv2.drawChessboardCorners(img_draw, CHECKERBOARD, corners2, ret)
        
        # Show successful images
        cv2.imshow('Corners Found', img_draw)
        print(f"OK: Corners found in {fname}")
        cv2.waitKey(0) 
    else:
        # Print failed images to terminal
        print(f"FAILED: Could not find corners in {fname}")

cv2.destroyAllWindows()

# Camera calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

# ---- COMPUTE REPROJECTION ERROR ----
total_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print("Re-projection error:", mean_error)

# ------------------------------------

# Print results
print("Camera matrix (K):", K)
print("Distortion coefficients:", dist.ravel())

np.savez(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_intrinsic.npz",
    K=K,
    dist=dist,
    rvecs=rvecs,
    tvecs=tvecs,
    reprojection_error=mean_error
)

