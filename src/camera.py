import cv2
import numpy as np

Camera_params = np.load(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz"
)

class Mapping:
    def __init__(self):
        self.H = Camera_params["H"]
        self.K = Camera_params["K"]
        self.dist = Camera_params["dist"]

        self.cam_gripper_mm = np.array([1.0, 1.0], dtype=np.float32)

        self.base_camera_mm = np.array([np.nan, np.nan], dtype=np.float32)
        self.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)

        # Optical center
        self.cx = self.K[0, 2]  
        self.cy = self.K[1, 2]  
        p_center = np.array([self.cx, self.cy, 1.0], dtype=np.float32).reshape(3, 1)
        P_center_mm = self.H @ p_center
        self.center_mm = P_center_mm[:2, 0] / P_center_mm[2, 0]
    
    def pixel_to_world(self, u, v):
        # Convert current point to mm
        p = np.array([u, v, 1.0], dtype=np.float32)
        P_mm = self.H @ p
        P_mm = P_mm[:2] / P_mm[2]

        relative_mm = P_mm - self.center_mm
        
        return relative_mm
    
    def return_cx_cy(self):
        return self.cx, self.cy
