import cv2
import numpy as np
import threading
import time
import queue
from yolov5_trt import YoLov5TRT

# Load calibration data
Camera_params = np.load("/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz")

# =========================
# Mapping class
# =========================
class Mapping:
    def __init__(self, uart):
        self.uart = uart
        self.camera = None
        self.limit_x_wp, self.limit_y_wp = np.array([200, 80], dtype=np.int32)
        self.source_pickup_pos = np.array([30, 50], dtype=np.int32)
        self.step_move = 20
        self.dir_move = 1
        self.numbers_of_bag = 0
        self.tissues_per_bag = None
        self.final_positions_lst = []
        # Homography and intrinsic matrix
        self.H = Camera_params["H"]
        self.K = Camera_params["K"]
        
        # Fixed mechanical offset (camera -> gripper), measured manually
        # self.camera_gripper_mm = np.array([16.0, -24.25], dtype=np.float32)
        self.camera_gripper_mm = np.array([11.0, -29.25], dtype=np.float32)

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

    def import_camera_to_mapping(self, camera):
        self.camera = camera

    def compute_euclidean_dist(self, pos1, pos2=np.array([0, 0])):
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        return np.sqrt(np.sum((p1 - p2)**2))

    def pixel_to_world(self, u, v):
        # Undistort single point using camera's dist coefficients
        src_pt = np.array([[[u, v]]], dtype=np.float32)
        
        # Access dist directly from the imported camera object
        dst_pt = cv2.undistortPoints(src_pt, self.K, self.camera.dist, P=self.K)
        u_clean = dst_pt[0][0][0]
        v_clean = dst_pt[0][0][1]

        # Convert clean pixel to mm using homography
        p = np.array([u_clean, v_clean, 1.0], dtype=np.float32)
        P_mm = self.H @ p
        P_mm = P_mm[:2] / P_mm[2]

        # Relative to image center
        return P_mm - self.center_mm
    
    def update_object_from_pixel(self, all_object_pixels):
        # Update bag position in camera frame
        temp_list = []

        # Get value of the each position
        for u, v in all_object_pixels:
            mm_offset = self.pixel_to_world(u, v)
            temp_list.append(mm_offset)
            
        with self.lock:
            if temp_list:
                self.camera_bag_mm = np.array(temp_list, dtype=np.float32)
                # print(self.camera_bag_mm)
            else:
                self.camera_bag_mm = None
                
    def update_base_camera_position(self, x_mm, y_mm):
        # Update camera position from motion system
        with self.lock:
            self.base_camera_mm[:] = [x_mm, y_mm]

    def compute_final_base_position(self, custom_objects=None):
        list_final_positions = []

        with self.lock:
            target_objects = custom_objects if custom_objects is not None else self.camera_bag_mm
            # Check if data is missing or if robot position is invalid
            if (target_objects is None or np.isnan(self.base_camera_mm).any()):
                return None
             
            # Ensure bags is treated as a 2D array (N, 2)
            bags = target_objects
            if bags.ndim == 1:
                bags = [bags]

            for bag_offset in bags:
                # Skip if this specific bag coordinate is invalid
                if np.isnan(bag_offset).any():
                    continue  

                # Calculate position: Base = Current_Robot + Offset_from_Camera - Mechanical_Offset
                bag_base_mm = self.base_camera_mm + bag_offset
                final_position = bag_base_mm - self.camera_gripper_mm

                target_x = final_position[0]
                target_y = final_position[1]

                # SAFETY BOUNDARY CHECK: Only add if within Workspace limits
                if 0 <= target_x <= self.limit_x_wp and 0 <= target_y <= self.limit_y_wp:
                    list_final_positions.append(final_position.copy())

        if not list_final_positions:
            return None
        # Returns a list (can be empty, have 1 object, or multiple objects)
        return list_final_positions
    
    def get_camera_bag_mm(self):
        with self.lock:
            return self.camera_bag_mm.copy()     

    def get_final_position(self, result, object_label="sweet potato"):
        # Wait in 1s
        time.sleep(1)

        # Update axes from result
        self.uart.axes["X"] = result["Current_X"]
        self.uart.axes["Y"] = result["Current_Y"]

        # Update current position for calculating
        self.update_base_camera_position(self.uart.axes["X"], self.uart.axes["Y"])   
        
        if self.camera is not None:
            if object_label == "strawberry" or object_label == "sweet potato":
                pixels = [t['center'] for t in self.camera.last_detected_tissues]
                self.update_object_from_pixel(pixels) # Use the same mapping logic
            elif object_label == "bag":
                pixels = self.camera.all_bag_pixels
                self.update_object_from_pixel(pixels)
        # Calculate final position
        final_position = self.compute_final_base_position()

        return final_position

    def wait_for_arrival(self, request):
        arrival_flag = False
        while not arrival_flag:
            try:
                result = self.uart.incoming_mailbox.get(timeout=2.0)
                if result["type"] == 6:
                    dist_x = abs(result["Current_X"] - self.uart.axes["X"])
                    dist_y = abs(result["Current_Y"] - self.uart.axes["Y"])
                    if dist_x < 1.0 and dist_y < 1.0:
                        arrival_flag = True
            except queue.Empty:
                frame = [request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper]
                self.uart.send_data(frame)
                
    def moveZ_Up_Down(self, request=2):
        # --- PHASE 1: MOVE GRIPPER DOWN ---
        # Open gripper firstly
        self.uart.gripper = 90
        self.uart.send_data([3, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper])
        time.sleep(0.5)

        # Move Z after
        self.uart.axes["Z"] = 50  
        frame_down = [request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper]
        self.uart.send_data(frame_down)
        print(f"\nMoving Z DOWN to: [{self.uart.axes['X'], self.uart.axes['Y'], self.uart.axes['Z']}]")
        # Wait for Z to reach the DOWN position
        arrival_z_down = False
        while not arrival_z_down:
            try:
                result = self.uart.incoming_mailbox.get(timeout=2)
                if result["type"] == 6:
                    # Verify if current Z is within 1.0mm of target
                    if abs(result["Current_Z"] - self.uart.axes["Z"]) < 1.0:
                        # print("Z reached target: DOWN")
                        arrival_z_down = True
                        time.sleep(0.5)             # Stop in 0.5s
            except queue.Empty:
                self.uart.send_data(frame_down)

        # Close gripper in 0.5s
        self.uart.gripper = 20
        self.uart.send_data([3, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper])
        time.sleep(0.5)

        # --- PHASE 2: MOVE GRIPPER UP ---
        self.uart.axes["Z"] = 0  # Set target to home position
        frame_up = [request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper]
        
        print(f"Moving Z UP to: [{self.uart.axes['X'], self.uart.axes['Y'], self.uart.axes['Z']}]")
        self.uart.send_data(frame_up)

        # Wait for Z to reach the UP position
        arrival_z_up = False
        while not arrival_z_up:
            try:
                result = self.uart.incoming_mailbox.get(timeout=2)
                if result["type"] == 6:
                    # Verify if current Z returned to 0
                    if abs(result["Current_Z"] - self.uart.axes["Z"]) < 1.0:
                        # print("Z reached target: UP")
                        arrival_z_up = True
            except queue.Empty:
                self.uart.send_data(frame_up)
        
    def moving(self, request=2):
        list_raw_final_position = []
        print("\n[MAPPING] Starting scanning process...")

        while True:
            # Init target coordinate frame
            frame = [request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper]

            # CLEAR MAILBOX
            while not self.uart.incoming_mailbox.empty():
                try: self.uart.incoming_mailbox.get_nowait()
                except: break

            # SEND MOVE COMMAND
            self.uart.send_data(frame)

            # Init flag and counting
            arrival_flag = False            
            start_time = time.time()

            while not arrival_flag:
                try:
                    # Wait for Motion Complete (Type 6) from Move Command
                    result = self.uart.incoming_mailbox.get(timeout=2.0)
                    
                    if result["type"] == 6:
                        # time.sleep(1)           # Sleep for waiting camera
                        # NOW ASK FOR ACTUAL POSITION
                        # self.uart.request_ask_current_position(request=0)
                        # result = self.uart.incoming_mailbox.get(timeout=1.0)
                        print("\n--- RESPONSE FROM MCU ---")
                        print(f"Type : {result['type']}")
                        print(f"X    : {result['Current_X']} mm")
                        print(f"Y    : {result['Current_Y']} mm")
                        print(f"Z    : {result['Current_Z']} mm")
                        print(f"Grip : {result['gripper']} deg")

                        # Update axes with current position from ASK command
                        # self.uart.axes["X"] = result["Current_X"]
                        # self.uart.axes["Y"] = result["Current_Y"]
                        
                        # self.update_base_camera_position(self.uart.axes["X"], self.uart.axes["Y"])

                        # Computing final position now
                        # raw_final_position = self.compute_final_base_position()
                        raw_final_position = self.get_final_position(result)
                        print(f"Raw final position: {raw_final_position}")
                        if raw_final_position is not None:
                            list_raw_final_position.append(raw_final_position)
                                
                        arrival_flag = True 
                    
                except queue.Empty:
                    print(f"[!] Timeout waiting for Type 6 at ({self.uart.axes['X']})...")
                    self.uart.request_ask_current_position(request=0)
                    if time.time() - start_time > 15.0: 
                        return list_raw_final_position

            # CHECK FINISH CONDITION
            if self.uart.axes["Y"] >= self.limit_y_wp:
                if (self.dir_move == 1 and self.uart.axes["X"] >= self.limit_x_wp) or \
                   (self.dir_move == -1 and self.uart.axes["X"] <= 0):
                    return list_raw_final_position

            # CALCULATE NEXT STEP TO MOVE
            self.uart.axes["X"] += self.dir_move * self.step_move
            
            # Row switching logic
            if self.uart.axes["X"] > self.limit_x_wp:
                self.uart.axes["X"] = self.limit_x_wp
                self.uart.axes["Y"] += self.step_move
                self.dir_move = -1
            elif self.uart.axes["X"] < 0:
                self.uart.axes["X"] = 0
                self.uart.axes["Y"] += self.step_move
                self.dir_move = 1
            
            time.sleep(0.05)

    def filtering_positions_lst(self, raw_list):
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
                # --- NEW LOGIC: Re-sort valid clusters by Euclidean distance to (0,0) ---
                valid_clusters = sorted_clusters[:self.numbers_of_bag]
                
                # The closest cluster to (0,0) will now be Rank 1
                valid_clusters.sort(key=self.compute_euclidean_dist)
                
                #Extract final coordinates in the new optimized order
                for cluster in valid_clusters:
                    avg_pos = np.mean(cluster, axis=0)
                    final_positions.append(np.round(avg_pos, 2).tolist())
        else:
            warning_msg = "ERROR: Failed to find the required number of clusters."

        # --- DEBUG & RESULTS ---
        print(f"\n--- 2D SYSTEM ANALYSIS (n={self.numbers_of_bag}) ---")
        # Logic display adjusted to show actual pick order if not re-mapping
        display_list = valid_clusters if (not should_remap and len(final_positions) > 0) else final_selection
        
        for i, g in enumerate(display_list):
            label = "[VALID OBJECT]" if i < self.numbers_of_bag else "[SUSPECTED NOISE]"
            centroid = np.mean(g, axis=0)
            dist = np.sqrt(centroid[0]**2 + centroid[1]**2)
            print(f"Rank {i+1} {label}: {len(g)} points | Dist: {dist:.1f}mm | Centroid: {np.round(centroid, 1)}")

        print("-" * 50)
        if should_remap:
            print(f"CONCLUSION: {warning_msg} -> RE-MAPPING REQUIRED!")
            return []
        print(f"CONCLUSION: {warning_msg} -> Ready.")
            
        return np.floor(np.array(final_positions) + 0.5).astype(int).tolist()
        
    def mapping(self, request=2):
        # Reset final positions list for mapping
        self.final_positions_lst = []

        # Reset coordinate to NaN before starting a new scan
        with self.lock:
            self.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)

        try:
            self.numbers_of_bag = int(input("Enter the actual number of bags (n): "))
        except ValueError:
            print("Invalid input! Please enter an integer.")
            self.numbers_of_bag = 0
        
        # Request homing before mapping
        self.uart.request_homing(request=5)
        homing_flag = False
        print("\n[HOMING] Robot is homing...")

        # Start homing
        while not homing_flag:
            try:
                result = self.uart.incoming_mailbox.get()
                # print(result)
                if result["type"] == 6:
                    print("\n[HOMING] Homing is done!")
                    homing_flag = True

            except queue.Empty:
                print("[HOMING] Error!")
                return []
        
        time.sleep(2)

        if self.numbers_of_bag != 0:
            # Step 1: Execute the movement and collect raw points
            raw_list = self.moving()
            if not raw_list:
                print("[MAPPING] Error: No data collected during scan.")
                return []
            
            # Flattend from 2D list to 1D list for before push in filtering function
            flattened_list = []
            for frame_data in raw_list:
                if frame_data is not None: 
                    for bag in frame_data: 
                        flattened_list.append(bag)

            print("[MAPPING] Scanning finished. Analyzing data...")
            self.final_positions_lst = self.filtering_positions_lst(flattened_list)
            print(f"Final positions are: {self.final_positions_lst}")
            time.sleep(2)
            # Go to base
            self.uart.axes = {"X": 0, "Y": 0, "Z": 0}
            frame = [request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper]
            self.uart.send_data(frame)
    
    def calculate_tissues_drop_pos(self, bag_base_pos, dropped_count=0, margin=5, gap_dist=10):
        # Default drop position is the center of the bag
        final_x = bag_base_pos[0]
        final_y = bag_base_pos[1]

        with self.camera.lock:
            detected_at_bag = list(self.camera.last_detected_tissues)
            detected_bags = list(self.camera.last_detected_bags)

        if not detected_bags: 
            return int(final_x), int(final_y)

        bag_w, bag_h = detected_bags[0]['size']
        w2, h2 = self.current_tissue_size 

        # Define priority patterns based on margin
        r_x, r_y = (bag_w / 2 - margin), (bag_h / 2 - margin)
        
        if self.tissues_per_bag == 1:
            priority_points = [(0, 0)]
            
        elif self.tissues_per_bag == 2:
            # For 2 tissues: Always use Diagonal corners first to avoid center
            priority_points = [(-r_x, -r_y), (r_x, r_y), (0, 0)]
        else:
            # For 3+ tissues: 4 corners (Skip center to keep space open)
            priority_points = [
                (-r_x, -r_y), # Corner 1 (Top-Left)
                (r_x, r_y),   # Corner 2 (Bottom-Right)
                (r_x, -r_y),  # Corner 3 (Top-Right)
                (-r_x, r_y)   # Corner 4 (Bottom-Left)
            ]

        # Use dropped_count to determine the starting priority point
        # This ensures we don't pick the same corner twice even if AI misses previous tissues
        start_idx = dropped_count % len(priority_points)
        
        # Create a search list starting from the intended priority index
        search_points = priority_points[start_idx:] + priority_points[:start_idx]

        chosen_offset = None
        for target_x, target_y in search_points:
            is_valid = True
            for tissue in detected_at_bag:
                w1, h1 = tissue['size']
                x1, y1 = self.pixel_to_world(tissue['center'][0], tissue['center'][1])
                
                req_dist_x = (w1 / 2) + (w2 / 2) + gap_dist
                req_dist_y = (h1 / 2) + (h2 / 2) + gap_dist

                if abs(target_x - x1) < req_dist_x and abs(target_y - y1) < req_dist_y:
                    is_valid = False
                    break
            
            if is_valid:
                chosen_offset = (target_x, target_y)
                break
        
        # Final position assignment and logging
        if chosen_offset is not None:
            final_x += chosen_offset[0]
            final_y += chosen_offset[1]
            
            # Specific log for the first tissue placement
            if dropped_count == 0:
                print(f"[LOGIC] First Tissue: Placed at {chosen_offset}")
            else:
                print(f"[LOGIC] Tissue {dropped_count + 1}/{self.tissues_per_bag}: Found spot at {chosen_offset}")
        else:
            print(f"[LOGIC] Warning: All priority spots blocked for Tissue {dropped_count + 1}")

        return int(final_x), int(final_y)
    
    # def calculate_tissues_drop_pos(self, bag_base_pos):
    #     # Default drop position is the center of the bag
    #     final_x = bag_base_pos[0]
    #     final_y = bag_base_pos[1]

    #     with self.camera.lock:
    #         detected_at_bag = list(self.camera.last_detected_tissues)
    #         detected_bags = list(self.camera.last_detected_bags)

    #     # Return to center if no bag is detected
    #     if not detected_bags: 
    #         return int(final_x), int(final_y)

    #     # Get the real-world dimensions of the bag
    #     bag_w, bag_h = detected_bags[0]['size']

    #     # Start at the first bag 
    #     if len(detected_at_bag) == 0:
    #         # Position seed at Top-Left corner with 5mm margin from edges
    #         edge_offset_x = -(bag_w / 2 - 5)
    #         edge_offset_y = -(bag_h / 2 - 5)
    #         final_x += edge_offset_x
    #         final_y += edge_offset_y
    #         print(f"[LOGIC] First Seed: Placed at top-left corner")

    #     # Start at the second bag
    #     else:
    #         # Get size of current bag
    #         w1, h1 = detected_at_bag[0]['size']
    #         # Convert pixel of current bag to mm
    #         x1, y1 = self.pixel_to_world(detected_at_bag[0]['center'][0], 
    #                                     detected_at_bag[0]['center'][1])
    #         # Get data of tissue is detected from thread process camera
    #         w2, h2 = self.current_tissue_size 

    #         # Calculate required center-to-center distance for a 10mm gap between BBox edges
    #         # Formula: (Width1/2) + (Width2/2) + 10mm
    #         req_dist_x = (w1 / 2) + (w2 / 2) + 10
    #         req_dist_y = (h1 / 2) + (h2 / 2) + 10

    #         # Check if placing Seed 2 at (0,0) center violates the 10mm gap
    #         if abs(x1) < req_dist_x and abs(y1) < req_dist_y:
    #             # Calculate necessary shift to reach the 10mm safety threshold
    #             shift_x = req_dist_x - abs(x1)
    #             # Shift in the opposite direction of Seed 1
    #             avoid_x = shift_x if x1 < 0 else -shift_x
                
    #             # Boundary check: Ensure Seed 2 center stays within bag limits (5mm from edge)
    #             limit_x = bag_w / 2 - 5
    #             if abs(final_x + avoid_x - bag_base_pos[0]) > limit_x:
    #                 # If X-axis shift exceeds bag boundary, try Y-axis shift instead
    #                 shift_y = req_dist_y - abs(y1)
    #                 avoid_y = shift_y if y1 < 0 else -shift_y
    #                 final_y += avoid_y
    #                 print(f"[LOGIC] Avoidance: Shifted along Y-axis")
    #             else:
    #                 final_x += avoid_x
    #                 print(f"[LOGIC] Avoidance: Shifted along X-axis")
    #         else:
    #             print("[LOGIC] Center is safe: Placing at bag center")
            
    #     return int(final_x), int(final_y)
    
    def run(self, request=2):
        if len(self.final_positions_lst) == 0:
            print("[RUN] No bags mapped. Please run mapping first!")
            return

        try:
            self.tissues_per_bag = int(input("Enter number of tissues per bag: "))
        except ValueError:
            print("[RUN] Input must be an integer!")
            return
        
        if self.tissues_per_bag >= 0:
            # Initialize task management: [ [x, y], number of tissues ]
            current_tasks = [[list(pos), self.tissues_per_bag] for pos in self.final_positions_lst]

            while current_tasks:
                # --- STEP 1: MOVE TO FIXED SOURCE STATION ---
                # Clear previous detection data before arriving at source
                with self.camera.lock:
                    self.camera.last_detected_tissues = []

                self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"] = self.source_pickup_pos[0], self.source_pickup_pos[1], 0
                frame_source = [request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper]
                self.uart.send_data(frame_source)
                
                # Wait for Type 6 response to get actual source coordinates
                arrival_source = None
                list_tissues = None
                while arrival_source is None:
                    try:
                        result = self.uart.incoming_mailbox.get(timeout=2.0)
                        if result["type"] == 6:
                            # Use target_type="tissue" to fetch seedlings from self.camera
                            list_tissues = self.get_final_position(result, object_label="tissue")
                            with self.camera.lock:
                                if self.camera.last_detected_tissues:
                                    self.current_tissue_size = self.camera.last_detected_tissues[0]['size']

                            arrival_source = result
                    except queue.Empty:
                        self.uart.send_data(frame_source)
                # --- STEP 2: PICKING PROCESS ---
                if list_tissues and len(list_tissues) > 0:
                    seed_pos = list_tissues[0] # Get the first position of tissue
                    
                    # Move to precise tissue coordinates
                    self.uart.axes["X"], self.uart.axes["Y"] = int(seed_pos[0]), int(seed_pos[1])
                    self.uart.send_data([request, self.uart.axes["X"], self.uart.axes["Y"], 0, self.uart.gripper])
                    self.wait_for_arrival(request) 
                    
                    # Execute picking (Lifts Z back to 0 while keeping gripper closed)
                    self.moveZ_Up_Down() 

                    # Clear source detection data after picking is done
                    with self.camera.lock:
                        self.camera.last_detected_tissues = []
                else:
                    print("[!] No tissue detected. Retrying source cycle...")
                    continue

                # --- STEP 3: FIND NEAREST BAG (EUCLIDEAN DISTANCE) ---
                current_robot_pos = [self.uart.axes["X"], self.uart.axes["Y"]]
                nearest_idx = min(
                    range(len(current_tasks)), 
                    key=lambda i: self.compute_euclidean_dist(current_tasks[i][0], current_robot_pos)
                )
                target_bag_pos = current_tasks[nearest_idx][0]

                # --- STEP 4: MOVE TO BAG AND RELEASE ---
                # 4.1 MOVE CAMERA TO BAG CENTER (Inverse offset)
                view_x = target_bag_pos[0] + self.camera_gripper_mm[0]
                view_y = target_bag_pos[1] + self.camera_gripper_mm[1]

                self.uart.axes["X"], self.uart.axes["Y"] = int(view_x), int(view_y)
                self.uart.send_data([request, self.uart.axes["X"], self.uart.axes["Y"], 0, self.uart.gripper])
                self.wait_for_arrival(request)

                # 4.2 WAIT FOR AI TO SCAN THE BAG
                time.sleep(1.5) 

                # 4.3 CALCULATE FINAL DROP POSITION (Including Avoidance + Gripper Offset)
                # Calculate how many tissues have already been dropped in this specific bag
                # Calculation: Goal amount - Remaining amount in the task list
                dropped_in_bag = self.tissues_per_bag - current_tasks[nearest_idx][1]
                
                # Pass the count into the logic to ensure we pick a NEW corner
                drop_x, drop_y = self.calculate_tissues_drop_pos(target_bag_pos, dropped_count=dropped_in_bag)

                # 4.4 MOVE GRIPPER TO TARGET
                self.uart.axes["X"], self.uart.axes["Y"] = drop_x, drop_y
                self.uart.send_data([request,self.uart.axes["X"], self.uart.axes["Y"], 0, self.uart.gripper])
                self.wait_for_arrival(request)

                # 4.5 LOWER Z-AXIS INTO THE BAG (Go down to Z=50)
                self.uart.axes["Z"] = 50
                self.uart.send_data([request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper])
                self.wait_for_arrival(request)

                # 4.6 OPEN GRIPPER TO RELEASE (Change gripper angle to 90)
                self.uart.gripper = 90 
                self.uart.send_data([3, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper])
                time.sleep(0.5) 

                # 4.7 LIFT Z-AXIS BACK TO SAFE HEIGHT (Back to Z=0)
                self.uart.axes["Z"] = 0
                self.uart.send_data([request, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper])
                self.wait_for_arrival(request)

                # Clear bag detection data to prepare for the next cycle
                with self.camera.lock:
                    self.camera.last_detected_tissues = []

                # --- STEP 5: UPDATE TASK STATUS ---
                current_tasks[nearest_idx][1] -= 1
                if current_tasks[nearest_idx][1] <= 0:
                    print(f"[INFO] Bag at {target_bag_pos} is complete.")
                    current_tasks.pop(nearest_idx)

            print("\n[RUN] All bags filled. Returning home...")
            self.uart.send_data([request, 0, 0, 0, 90])
        
# ===================================
# Camera detection thread
# ===================================
class CameraDetect(threading.Thread):
    def __init__(self, engine_path, mapping, enable_display=True):
        super().__init__(daemon=True)
        
        # ---------------- Camera init ----------------
        self.pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        print("Hardware Driver successfully opened.")

        # ---------------- External objects ----------------
        self.mapping = mapping
        self.enable_display = enable_display
        self.all_bag_pixels = None
        self.last_detected_bags = []
        self.last_detected_tissues = []

        # ---------------- Detection params ----------------
        self.INPUT_SIZE = 640
        self.IOU_THRESH = 0.45
        self.CONF_THRESH = 0.90
        self.classes = ["strawberry", "sweet potato"]

        # ---------------- Camera calibration ----------------
        self.K = Camera_params["K"]
        self.dist = Camera_params["dist"]
        self.newK = Camera_params["newK"]

        # ---------------- Thread control ----------------
        # self.running = True
        self.running = False    
        self.draw_mask = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.lock = threading.Lock()

        # Shared frame for display
        self.det_frame = None

        # ---------------- TensorRT Inference Init ----------------
        # Initialize the engine model
        self.model = YoLov5TRT(engine_path, self.classes, self.CONF_THRESH, self.IOU_THRESH)
        
        # Initialize time tracker for FPS calculation
        self.prev_time = time.time()

        # Display thread
        self.display_thread = None
        self.capture_thread = None

        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
    # ==========================================================
    # Public control
    # ==========================================================
    def stop_camera(self):
        self.running = False
        self.cap.release()
        self.model.destroy()

    def start_camera(self):
        """Starts the log threads for capturing frames from the hardware"""
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        print("[CAMERA] Capture loop started.")
        while self.running:
            if self.frame_queue.full():
                self.cap.grab()
                time.sleep(0.005)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            
            self.frame_queue.put(frame)
            
        self.cap.release()
        print("[CAMERA] Capture loop stopped.")

    def start_display(self):
        if not self.enable_display:
            return
        
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True
        )
        self.display_thread.start()

    def _display_loop(self):
        print("[DISPLAY] Display loop started.")
        if self.enable_display:
            cv2.namedWindow("FarmBot Vision", cv2.WINDOW_NORMAL)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Undistort
            # frame = cv2.undistort(frame, self.K, self.dist, None, self.newK)

            # Detect
            draw = self.infer_and_detect(frame)

            if self.enable_display and draw is not None:
                cv2.imshow("FarmBot Vision", draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
        if self.enable_display:
            cv2.destroyAllWindows()
        print("[DISPLAY] Display loop stopped.")

    # ==========================================================
    # Detection pipeline
    # ==========================================================
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

    def get_object_size_mm(self, bbox):
        if bbox is None or len(bbox) < 4:
            print("[SIZE] Invalid bbox provided.")
            return 0, 0

        # Convert the two diagonal corners of the bbox to world coordinates (mm)
        # Using your existing pixel_to_world logic
        point_min = self.mapping.pixel_to_world(bbox[0], bbox[1]) # Top-left
        point_max = self.mapping.pixel_to_world(bbox[2], bbox[3]) # Bottom-right

        # The size is the absolute difference between these world coordinates
        # point_min and point_max are already (x, y) relative to center_mm
        width_mm = abs(point_max[0] - point_min[0])
        height_mm = abs(point_max[1] - point_min[1])

        # Return rounded values for easier calculation
        return round(float(width_mm), 2), round(float(height_mm), 2)
    
    def infer_and_detect(self, frame):
        # The external model automatically draws masks and bboxes onto the raw frame
        batch_results, infer_time, boxes = self.model.infer([frame], self.draw_mask)
        
        # Extract the processed frame containing standard AI overlays
        draw = batch_results[0]

        h_frame, w_frame = frame.shape[:2]
        is_real_bbox = False
        self.all_bag_pixels = []    # List of all bag positions (pixel)
        bags_bboxes_mm = []         # List for separated bag data
        tissues_bboxes_mm = []      # List for separated tissue data
        
        # Structure of each row in boxes: [x1, y1, x2, y2, confidence, class_id, ...]
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1_b, y1_b, x2_b, y2_b = box[:4]
                class_id = int(box[5])
                
                # Boundary safety check for class indices
                if class_id >= len(self.classes):
                    continue
                    
                label = self.classes[class_id]

                # Reconstruct standard width, height, and center coordinate configurations
                x, y, w, h = int(x1_b), int(y1_b), int(x2_b - x1_b), int(y2_b - y1_b)
                u = (x * 2 + w) // 2
                v = (y * 2 + h) // 2

                # Calculate real-world dimensional metrics (mm) utilizing the preserved function
                obj_w_mm, obj_h_mm = self.get_object_size_mm([x1_b, y1_b, x2_b, y2_b])
                size_text = f"{obj_w_mm}x{obj_h_mm}mm"

                # Render object physical dimension values on the display interface
                cv2.putText(draw, size_text, (x, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Process the growth container unit ("sweet potato" represents the bag target)
                if label == "bag":
                    if self.is_real_bbox(x, y, w, h, w_frame, h_frame):
                        self.all_bag_pixels.append((u, v))
                        bags_bboxes_mm.append({
                            'center': (u, v), 
                            'bbox': [x1_b, y1_b, x2_b, y2_b], 
                            'size': (obj_w_mm, obj_h_mm)
                        })
                        is_real_bbox = True
                        
                        # Render target reference tracking lines for hardware alignment verification
                        cv2.circle(draw, (u, v), 5, (0, 0, 255), -1)        # Target center anchor
                        cv2.circle(draw, (int(self.cx), int(self.cy)), 5, (255, 0, 0), -1)  # Optical center point
                        cv2.line(draw, (int(self.cx), int(self.cy)), (u, v), (0, 255, 255), 2) # Offset vector
                
                # Process the biological seedling unit ("strawberry" represents the tissue sprout)
                elif label == "strawberry" or label == "sweet potato":
                    tissues_bboxes_mm.append({
                        'center': (u, v), 
                        'bbox': [x1_b, y1_b, x2_b, y2_b], 
                        'size': (obj_w_mm, obj_h_mm)
                    })
                    cv2.circle(draw, (u, v), 5, (0, 255, 0), -1) # Sprout focal spot
                    cv2.circle(draw, (int(self.cx), int(self.cy)), 5, (255, 0, 0), -1)  # Optical center point
                    cv2.line(draw, (int(self.cx), int(self.cy)), (u, v), (0, 255, 255), 2) # Offset vector
                
        if is_real_bbox and self.all_bag_pixels:
            # Inject raw pixels into homography calculation arrays
            self.mapping.update_object_from_pixel(self.all_bag_pixels)          
            camera_bag_mm_list = self.mapping.get_camera_bag_mm()       
            list_final_positions = self.mapping.compute_final_base_position()   

            # Display calculated absolute robot base destination targets (Gantry Frame coordinates)
            if list_final_positions is not None:
                for i, pos in enumerate(list_final_positions):
                    y_offset = 30 + (i * 30)
                    cv2.putText(draw, f"Bag-{i+1}: X={pos[0]:.1f}, Y={pos[1]:.1f} mm",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display relative camera lens offset dimensions alongside individual objects
            for j, (u_p, v_p) in enumerate(self.all_bag_pixels):
                if j < len(camera_bag_mm_list):
                    mm_val = camera_bag_mm_list[j]
                    cv2.putText(draw, f"({mm_val[0]:.1f}, {mm_val[1]:.1f})mm", 
                                (u_p + 8, v_p + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Safely pipe structural frames data to global thread monitoring attributes
        with self.lock:
            self.last_detected_bags = bags_bboxes_mm
            self.last_detected_tissues = tissues_bboxes_mm
        
        # Reset relative tracking coordinate storage if data becomes unstable
        if not is_real_bbox:
            with self.mapping.lock:
                self.mapping.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)

        # Calculate System FPS and Overlay performance text onto the frame
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        self.prev_time = curr_time

        # Display time of FPS and latency
        cv2.putText(draw, f"FPS: {fps:.1f}", (20, 40), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(draw, f"Inference time: {infer_time*1000:.1f}ms", (20, 80), 1, 1.5, (0, 255, 0), 2)
        return draw
