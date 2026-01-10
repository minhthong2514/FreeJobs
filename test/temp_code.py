import cv2
import numpy as np

# Load calibration
K_data = np.load("/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz")
K = K_data["K"]
dist = K_data["dist"]

# Optical center
cx = int(K[0, 2])
cy = int(K[1, 2])
print(cx)
print(cy)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Undistort
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, K, dist, None, new_K)

    # Draw optical center
    cv2.circle(undistorted, (cx, cy), 6, (0, 0, 255), -1)

    cv2.imshow("Undistorted - Optical Center", undistorted)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
