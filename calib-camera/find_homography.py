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

# Load image from dataset
img = cv2.imread('/home/minhthong/Desktop/code/farmbot/calib-camera/result_imgs/homo_img.jpg')
h, w = img.shape[:2]
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0)

# Undistort the image
undist_img = cv2.undistort(img, K, dist, None, newK)

# Checkerboard info
CHECKERBOARD = (15, 10)
cols, rows = CHECKERBOARD
squares_size = 5.3333               # 5.3333mm

# Create world points
world_pts = np.zeros((cols * rows, 2), dtype=np.float32)
world_pts[:, 0] = np.tile(np.arange(cols), rows) * squares_size
world_pts[:, 1] = np.repeat(np.arange(rows), cols) * squares_size

# Find corners
ret, corners = cv2.findChessboardCorners(undist_img, CHECKERBOARD)
# print(corners.shape)

if ret:
    # Refine corner coordinates for sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Reshape from (N,1,2) to (N,2) 
    pixels_pts = corners2.reshape(-1, 2)
    # print(pixels_pts.shape)

    # Calculate HOMOGRAPHY matrix
    H, mask = cv2.findHomography(pixels_pts, world_pts, cv2.RANSAC, 5.0)
    
    # Verify calibration accuracy
    errors = []
    for i in range(len(pixels_pts)):
        p = np.array([pixels_pts[i][0], pixels_pts[i][1], 1.0])
        P = H @ p
        P /= P[2]
        err = np.linalg.norm(P[:2] - world_pts[i])
        errors.append(err)
    print(f"Mean error: {np.mean(errors):.4f} mm")
    print(f"Max error: {np.max(errors):.4f} mm") 

# Function to convert Pixel (u, v) to World (X, Y)
def pixel_to_world(x, y, H):
    p = np.array([x, y, 1.0])
    P = H @ p
    return P[0] / P[2], P[1] / P[2]

# Mouse callback function for interactive measurement
def mouse_callback(event, x, y, flags, param):
    global origin_world, undist_img

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert clicked pixel to world mm
        X, Y = pixel_to_world(x, y, H)

        # Set the origin point (0,0) on first click
        if origin_world is None:
            origin_world = (X, Y)
            print(f"Origin set at: ({X:.2f}, {Y:.2f}) mm")

            cv2.circle(undist_img, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(undist_img, "ORIGIN(0,0)mm", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        else:
            # Calculate relative distance from the established origin            
            Xr = X - origin_world[0]
            Yr = origin_world[1] - Y  
            print(f"Pixel ({x},{y}) -> World relative: ({Xr:.2f}, {Yr:.2f}) mm")

            # Draw point and labels on the image
            cv2.circle(undist_img, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(undist_img, f"({Xr:.2f},{Yr:.2f}) mm", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


cv2.namedWindow("Pixel to World")
cv2.setMouseCallback("Pixel to World", mouse_callback)
origin_world = None
base_img = undist_img.copy() 

while True:
    h,w ,_ = undist_img.shape
    # print(h,w)
    cv2.imshow("Pixel to World", undist_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):   # press q to exit
        break

    if key == ord('r'):
        origin_world = None
        undist_img = base_img.copy() 
        print("Reset Origin + points.")


cv2.destroyAllWindows()

np.savez("/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz",
    H = H,
    K=K,
    newK = newK,
    dist=dist,
    rvecs=rvecs,
    tvecs=tvecs,
)
print("\nFinal parameters saved successfully.")