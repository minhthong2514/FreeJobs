import cv2
import numpy as np
import threading
import onnxruntime as ort
import time
import queue
# Load calibration data
Camera_params = np.load(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz"
)

# =========================
# Mapping class (math only)
# =========================
class Mapping:
    def __init__(self, uart):
        self.uart = uart
        self.wp_x, self.wp_y = np.array([60, 80], dtype=np.int32)
        self.step_move = 20
        self.dir_move = 1
        self.numbers_of_bag = 0
        # self.positions_lst = []
        # Homography and intrinsic matrix
        self.H = Camera_params["H"]
        self.K = Camera_params["K"]
        
        # Fixed mechanical offset (camera -> gripper), measured manually
        self.camera_gripper_mm = np.array([16.0, -24.25], dtype=np.float32)

        # Current camera position in base coordinate (updated externally)
        self.base_camera_mm = np.array([0.0, 0.0], dtype=np.float32)

        # Latest detected bag position in camera coordinate
        self.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)

        # Thread lock
        self.lock = threading.Lock()

        # Optical center in pixel
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        p_center = np.array([self.cx, self.cy, 1.0], dtype=np.float32)

        # Optical center mapped to mm
        P_center_mm = self.H @ p_center
        self.center_mm = P_center_mm[:2] / P_center_mm[2]

    def pixel_to_world(self, u, v):
        # Convert pixel to mm using homography
        p = np.array([u, v, 1.0], dtype=np.float32)
        P_mm = self.H @ p
        P_mm = P_mm[:2] / P_mm[2]

        # Relative to image center
        return P_mm - self.center_mm

    def update_bag_from_pixel(self, u, v):
        # Update bag position in camera frame
        with self.lock:
            self.camera_bag_mm = self.pixel_to_world(u, v)

    def update_base_camera_position(self, x_mm, y_mm):
        # Update camera position from motion system
        with self.lock:
            self.base_camera_mm[:] = [x_mm, y_mm]

    def compute_final_base_position(self):
        # Compute final gripper position in base frame
        with self.lock:
            if (self.camera_bag_mm is None or np.isnan(self.camera_bag_mm).any() or np.isnan(self.base_camera_mm).any()):
                return None

            bag_base_mm = self.base_camera_mm + self.camera_bag_mm
            final_position = bag_base_mm - self.camera_gripper_mm
            
            # SAFETY BOUNDARY CHECK            
            target_x = final_position[0]
            if target_x < 0:
                return None # Unreachable on the left
            if target_x > self.wp_x:
                return None
            return final_position

    def get_camera_bag_mm(self):
        with self.lock:
            return self.camera_bag_mm.copy()        
    
    def moving(self, request=2):
        positions_lst = []
        while True:
            waiting_for_base_camera = True
            frame = [
                request,
                self.uart.axes["X"],
                self.uart.axes["Y"],
                self.uart.axes["Z"],
                self.uart.gripper
            ]

            self.uart.send_data(frame)
            time.sleep(0.5)
            self.uart.request_ask_current_position(request=0)
            try:
                result = self.uart.incoming_mailbox.get()
                if waiting_for_base_camera and result["type"] == 6:
                    time.sleep(2)
                    current_x = result["Current_X"]
                    current_y = result["Current_Y"]
                    self.update_base_camera_position(current_x, current_y)
                    waiting_for_base_camera = False
                    final_position = self.compute_final_base_position()
                    if final_position is not None:
                        positions_lst.append(final_position)

                print(f"\nPhản hồi nhận được: X={current_x}, Y={current_y}")
                
                
            except queue.Empty:
                print("Lỗi: Không nhận được phản hồi từ Robot!")


            if self.uart.axes["Y"] >= self.wp_y:
                if (self.dir_move == 1 and self.uart.axes["X"] >= self.wp_x) or (self.dir_move == -1 and self.uart.axes["X"] <= 0):
                    print(f"\nList postions of object: {positions_lst}")
                    return positions_lst
                    # break 

            # Move X
            self.uart.axes["X"] += self.dir_move * self.step_move

            if self.uart.axes["X"] > self.wp_x:
                self.uart.axes["X"] = self.wp_x
                self.uart.axes["Y"] += self.step_move
                self.dir_move = -1                                                                                                                                                                                                                                                  
            elif self.uart.axes["X"] < 0:
                self.uart.axes["X"] = 0
                self.uart.axes["Y"] += self.step_move
                self.dir_move = 1
            time.sleep(0.1)

    def process_noise_positions(self, raw_list):
        # Convert the Python List to a Numpy Array before processing
        if not raw_list:
            print("Error: No positions were collected!")
            return []
            
        position_lst = np.array(raw_list) 

        # Now this line will work perfectly
        position_lst_sorted = position_lst[position_lst[:, 0].argsort()]

        # --- STEP 1: Preliminary Clustering (Absolute threshold for X and Y) ---
        pre_clusters = []
        threshold_val = 2.0  # 2mm Threshold

        if len(position_lst_sorted) > 0:
            curr_group = [position_lst_sorted[0]]
        
            for i in range(1, len(position_lst_sorted)):
                prev_point = curr_group[-1]
                curr_point = position_lst_sorted[i]
            
                # Check thresholds for both X and Y axes
                # Join group if both X and Y distances are within 2mm
                if abs(curr_point[0] - prev_point[0]) <= threshold_val and \
                abs(curr_point[1] - prev_point[1]) <= threshold_val:
                    curr_group.append(curr_point)
                else:
                    pre_clusters.append(np.array(curr_group))
                    curr_group = [curr_point]
                
            pre_clusters.append(np.array(curr_group))

        # --- STEP 2: Fine Filtering around Centroid ---
        final_clusters = []
        for cluster in pre_clusters:
            center = np.mean(cluster, axis=0)
        
            # Filter: Points must stay within (Centroid ± Threshold) for both X and Y
            refined_group = cluster[
                (np.abs(cluster[:, 0] - center[0]) <= threshold_val) &
                (np.abs(cluster[:, 1] - center[1]) <= threshold_val)
            ]
        
            if len(refined_group) > 0:
                final_clusters.append(refined_group)

        # --- STEP 3: Sort clusters by size (Descending: Highest point count first) ---
        sorted_clusters = sorted(final_clusters, key=len, reverse=True)

        # --- STEP 4: Select n + 1 groups for analysis ---
        num_to_keep = self.numbers_of_bag + 1
        final_selection = sorted_clusters[:num_to_keep]

        # --- STEP 5: Delta Logic and Coordinate Extraction ---
        should_remap = False
        final_positions = []
        warning_msg = ""

        if len(sorted_clusters) >= self.numbers_of_bag:
            if len(sorted_clusters) > self.numbers_of_bag:
                count_n = len(sorted_clusters[self.numbers_of_bag-1])      # Last valid object count
                count_n_plus_1 = len(sorted_clusters[self.numbers_of_bag]) # First noise/suspected count
                delta = count_n - count_n_plus_1
            
                if delta < 1:
                    should_remap = True
                    warning_msg = f"WARNING: Low Delta ({delta}). Group {self.numbers_of_bag+1} is too similar to actual objects!"
                else:
                    warning_msg = f"SAFE: Valid Delta ({delta})."
            else:
                warning_msg = "SAFE: No significant noise detected."

            if not should_remap:
                for i in range(self.numbers_of_bag):
                    avg_pos = np.mean(sorted_clusters[i], axis=0)
                    final_positions.append(np.round(avg_pos, 2).tolist())
        else:
            warning_msg = "ERROR: Failed to find the required number of clusters."

        # --- DEBUG & RESULTS ---
        print(f"\n--- 2D SYSTEM ANALYSIS (n={self.numbers_of_bag}) ---")
        for i, g in enumerate(final_selection):
            label = "[VALID OBJECT]" if i < self.numbers_of_bag else "[SUSPECTED NOISE]"
            print(f"Rank {i+1} {label}: {len(g)} points | Centroid: {np.round(np.mean(g, axis=0), 1)}")

        print("-" * 50)
        if should_remap:
            print(f"CONCLUSION: {warning_msg} -> RE-MAPPING REQUIRED!")
            return []
        print(f"CONCLUSION: {warning_msg} -> Ready.")
            
        return np.floor(np.array(final_positions) + 0.5).astype(int).tolist()
        
    def mapping(self):
        # Reset coordinate to NaN before starting a new scan
        with self.lock:
            self.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)

        try:
            self.numbers_of_bag = int(input("Enter the actual number of bags (n): "))
        except ValueError:
            print("Invalid input! Please enter an integer.")
            self.numbers_of_bag = 0
        
        print("\n[MAPPING] Starting scanning process...")

        if self.numbers_of_bag != 0:
            # Step 1: Execute the movement and collect raw points
            raw_list = self.moving()
            
            if not raw_list:
                print("[MAPPING] Error: No data collected during scan.")
                return []

            # Step 2: Pass raw data to processing logic
            print("[MAPPING] Scanning finished. Analyzing data...")
            final_coords = self.process_noise_positions(raw_list)
            print(f"Final position is: {final_coords}")
            return final_coords
            
# ===================================
# Camera detection thread (vision only)
# ===================================
class CameraDetect(threading.Thread):
    def __init__(self, model_onnx_path, mapping, enable_display=True):
        super().__init__(daemon=True)

        # ---------------- Camera init ----------------
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ---------------- External objects ----------------
        self.mapping = mapping
        self.enable_display = enable_display

        # ---------------- Detection params ----------------
        self.INPUT_SIZE = 640
        self.IOU_THRESH = 0.45
        self.CONF_THRESH = 0.9
        self.classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']

        # ---------------- Camera calibration ----------------
        self.K = Camera_params["K"]
        self.dist = Camera_params["dist"]
        self.newK = Camera_params["newK"]

        # ---------------- Thread control ----------------
        self.running = True
        self.lock = threading.Lock()

        # Shared frame for display
        self.det_frame = None

        # ---------------- ONNX Runtime (init once) ----------------
        providers = ort.get_available_providers()
        print("Available providers:", providers)

        if "CUDAExecutionProvider" in providers:
            self.model = ort.InferenceSession(
                model_onnx_path,
                providers=["CUDAExecutionProvider"]
            )
            print("Using GPU for inference")
        else:
            self.model = ort.InferenceSession(
                model_onnx_path,
                providers=["CPUExecutionProvider"]
            )
            print("Using CPU for inference")

        self.input_name = self.model.get_inputs()[0].name

        # Display thread
        self.display_thread = None

        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
    # ==========================================================
    # Public control
    # ==========================================================
    def stop(self):
        self.running = False
        self.cap.release()

    def start_display(self):
        if not self.enable_display:
            return

        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True
        )
        self.display_thread.start()

    # ==========================================================
    # Thread 1: Capture + Detect
    # ==========================================================
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Undistort
            frame = cv2.undistort(
                frame, self.K, self.dist, None, self.newK
            )

            # Detect
            draw = self.infer_and_detect(frame)

            # Share frame for display
            with self.lock:
                self.det_frame = draw

            time.sleep(0.001)

    # ==========================================================
    # Thread 2: Display only
    # ==========================================================
    def _display_loop(self):
        while self.running:
            with self.lock:
                if self.det_frame is not None:
                    cv2.imshow("CameraDetect", self.det_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        cv2.destroyAllWindows()

    # ==========================================================
    # Detection pipeline
    # ==========================================================
    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (nw, nh))
        pad_x = (self.INPUT_SIZE - nw) // 2
        pad_y = (self.INPUT_SIZE - nh) // 2

        canvas = np.full(
            (self.INPUT_SIZE, self.INPUT_SIZE, 3),
            114, dtype=np.uint8
        )
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgb = np.transpose(img_rgb, (2, 0, 1))
        img_rgb = np.expand_dims(img_rgb, axis=0)

        return img_rgb, scale, pad_x, pad_y

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= self.IOU_THRESH]

        return keep

    def is_real_bbox(self, x, y, w, h, frame_w=640, frame_h=480, margin=15):

        # Calculate max boundaries
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        
        # Check if any side is within the margin of the frame edges
        if (x_min <= margin or 
            y_min <= margin or 
            x_max >= (frame_w - margin) or 
            y_max >= (frame_h - margin)):
            return False # Touching edge (Unsafe)
            
        return True # Inside safe zone

    def infer_and_detect(self, frame):
        img_input, scale, pad_x, pad_y = self.preprocess(frame)
        outputs = self.model.run(None, {self.input_name: img_input})
        pred = outputs[0][0]

        boxes, scores, class_ids = [], [], []

        for det in pred:
            conf = det[4]
            if conf < self.CONF_THRESH:
                continue

            class_prob = det[5:]
            class_id = int(np.argmax(class_prob))
            score = conf * class_prob[class_id]
            if score < self.CONF_THRESH:
                continue

            cx, cy, w, h = det[:4]

            x1 = int((cx - w / 2 - pad_x) / scale)
            y1 = int((cy - h / 2 - pad_y) / scale)
            x2 = int((cx + w / 2 - pad_x) / scale)
            y2 = int((cy + h / 2 - pad_y) / scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        idxs = self.nms(boxes, scores)
        draw = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        is_real_bbox = False

        for i in idxs:
            x, y, w, h = boxes[i]
            u = (x * 2 + w) // 2
            v = (y * 2 + h) // 2

            # cam_bag_mm_x = self.camera_bag_mm[0]
            # cam_bag_mm_y = self.camera_bag_mm[1]

            label = self.classes[class_ids[i]]

            
            # Update mapping (example: stop = bag)
            color_box = (0, 255, 0)
            if label == "stop":
                if self.is_real_bbox(x, y, w, h, w_frame, h_frame):
                    self.mapping.update_bag_from_pixel(u, v)
                    is_real_bbox = True

                    camera_bag_mm = self.mapping.get_camera_bag_mm()
                    final_position = self.mapping.compute_final_base_position()

                    if final_position is not None:
                        cv2.putText(
                            draw,
                            f"Final: X={final_position[0]:.2f}, Y={final_position[1]:.2f} mm",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        ) 
                    cv2.circle(draw, (u, v), 5, (0, 0, 255), -1)
                    cv2.circle(draw, (int(self.cx), int(self.cy)), 5, (255,0,0), -1)
                    cv2.line(draw, (int(self.cx), int(self.cy)), (u,v), (0,255,255), 2)
                    cv2.putText(draw, f"({camera_bag_mm[0]:.2f}, {camera_bag_mm[1]:.2f})mm",
                    (u+8,v+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                else:
                    color_box = (0, 0, 255)
                    
            cv2.rectangle(draw, (x, y), (x + w, y + h), color_box, 2)
            cv2.putText(draw, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not is_real_bbox:
            with self.mapping.lock:
                self.mapping.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)
        
        return draw