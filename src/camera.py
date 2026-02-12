import cv2
import numpy as np
import threading
import onnxruntime as ort
import time
import queue

# Load calibration data
Camera_params = np.load("/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz")

# =========================
# Mapping class
# =========================
class Mapping:
    def __init__(self, uart):
        self.uart = uart
        self.camera = None
        self.wp_x, self.wp_y = np.array([200, 100], dtype=np.int32)
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

    def import_camera_to_mapping(self, camera):
        self.camera = camera

    def compute_euclidean_dist(self, pos1, pos2=np.array([0, 0])):
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        return np.sqrt(np.sum((p1 - p2)**2))
    
    def pixel_to_world(self, u, v):
        # Convert pixel to mm using homography
        p = np.array([u, v, 1.0], dtype=np.float32)
        P_mm = self.H @ p
        P_mm = P_mm[:2] / P_mm[2]

        # Relative to image center
        return P_mm - self.center_mm

    def update_bag_from_pixel(self, all_bag_pixels):
        # Update bag position in camera frame
        temp_list = []

        # Get value of the each position
        for u, v in all_bag_pixels:
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

    def compute_final_base_position(self):
        list_final_positions = []

        with self.lock:
            # Check if data is missing or if robot position is invalid
            if (self.camera_bag_mm is None or np.isnan(self.base_camera_mm).any()):
                return None
             
            # Ensure bags is treated as a 2D array (N, 2)
            bags = self.camera_bag_mm
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
                if 0 <= target_x <= self.wp_x and 0 <= target_y <= self.wp_y:
                    list_final_positions.append(final_position.copy())

        if not list_final_positions:
            return None
        # Returns a list (can be empty, have 1 object, or multiple objects)
        return list_final_positions
    
    def get_camera_bag_mm(self):
        with self.lock:
            return self.camera_bag_mm.copy()     

    def get_final_position(self, result, object_label="bag"):
        # Wait in 1s
        time.sleep(1)

        # Update axes from result
        self.uart.axes["X"] = result["Current_X"]
        self.uart.axes["Y"] = result["Current_Y"]

        # Update current position for calculating
        self.update_base_camera_position(self.uart.axes["X"], self.uart.axes["Y"])   
        
        if self.camera is not None:
            if object_label == "tissue":
                pixels = [t['center'] for t in self.camera.last_detected_tissues]
                self.update_bag_from_pixel(pixels) # Use the same mapping logic
            else:
                pixels = self.camera.all_bag_pixels
                self.update_bag_from_pixel(pixels)
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
            if self.uart.axes["Y"] >= self.wp_y:
                if (self.dir_move == 1 and self.uart.axes["X"] >= self.wp_x) or \
                   (self.dir_move == -1 and self.uart.axes["X"] <= 0):
                    return list_raw_final_position

            # CALCULATE NEXT STEP TO MOVE
            self.uart.axes["X"] += self.dir_move * self.step_move
            
            # Row switching logic
            if self.uart.axes["X"] > self.wp_x:
                self.uart.axes["X"] = self.wp_x
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
    
    # def calculate_tissues_drop_pos(self, bag_base_pos):
    #     """
    #     Calculates the final robot world coordinates (X, Y) to drop a seedling,
    #     implementing an avoidance strategy if a seedling already exists in the bag.
        
    #     Args:
    #         bag_base_pos (list/tuple): The base coordinates [x, y] of the target bag center.
            
    #     Returns:
    #         tuple: (final_x, final_y) in integer mm for robot movement.
    #     """
    #     # 1. Initialize default drop position (Bag center adjusted by fixed Camera-to-Gripper offset)
    #     final_x = bag_base_pos[0] - self.camera_gripper_mm[0]
    #     final_y = bag_base_pos[1] - self.camera_gripper_mm[1]

    #     # # 2. Thread-safe access to the latest detection data from Camera class
    #     # with self.camera.lock:
    #     #     # Create a local copy to avoid "dictionary changed size during iteration" errors
    #     #     detected_tissues = list(self.camera.last_detected_tissues)

    #     # # 3. Check if the bag is already occupied (Avoidance Logic)
    #     # if len(detected_tissues) > 0:
    #     #     # Analyze the first detected seedling (Seedling 1)
    #     #     existing_seed = detected_tissues[0]
    #     #     u, v = existing_seed['center']
    #     #     w1, _ = existing_seed['size']
            
    #     #     # 4. Validity Check: Ensure the detected seedling is within a realistic radius of the bag center
    #     #     # This prevents using stale data from the Source Station if AI hasn't updated yet.
    #     #     if abs(u - self.cx) > 180 or abs(v - self.cy) > 180:
    #     #         print("[LOGIC] Seedling detected is too far from center, likely stale data. Using center drop.")
    #     #         return int(final_x), int(final_y)

    #     #     # 5. Calculate Seedling 1's horizontal offset relative to the Camera center (mm)
    #     #     offset_mm = self.pixel_to_world(u, v)
            
    #     #     # 6. Safety Margin: Define distance to move away from Seedling 1
    #     #     # Combined width + extra clearance (e.g., 15mm)
    #     #     safe_margin = w1 + 15 
            
    #     #     # 7. Directional Avoidance: If Seedling 1 is on the Right, move Left; otherwise move Right
    #     #     if offset_mm[0] > 0: # Seedling 1 is in the right-half of the bag
    #     #         avoid_x = -safe_margin
    #     #     else: # Seedling 1 is in the left-half or dead center
    #     #         avoid_x = safe_margin
            
    #     #     # 8. Boundary Constraint: Prevent the robot from dropping the seedling outside the bag radius
    #     #     # Assuming a standard bag radius (e.g., 35mm), we clip the offset.
    #     #     MAX_ALLOWED_OFFSET = 35
    #     #     avoid_x = max(min(avoid_x, MAX_ALLOWED_OFFSET), -MAX_ALLOWED_OFFSET)
                
    #     #     final_x += avoid_x
    #     #     print(f"[LOGIC] Avoidance triggered. Offset applied: {avoid_x:.1f}mm")

    #     return int(final_x), int(final_y)
    def calculate_tissues_drop_pos(self, bag_base_pos):
        """
        TEMPORARY DEBUG VERSION: 
        Returns the mapped gripper position directly to test the 'run' flow.
        """
        # Just return the mapped coordinates without any AI adjustment
        final_x = bag_base_pos[0]
        final_y = bag_base_pos[1]
        
        print(f"[DEBUG-FLOW] Target Bag: {bag_base_pos} -> Gripper will move to: {final_x, final_y}")
        
        return int(final_x), int(final_y)
    
    def run(self, request=2):
        if len(self.final_positions_lst) == 0:
            print("[RUN] No bags mapped. Please run mapping first!")
            return

        try:
            self.tissues_per_bag = int(input("Enter number of seedlings per bag: "))
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
                    print("  [!] No tissue detected. Retrying source cycle...")
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
                # We calculate where the camera needs to be so its optical center is over the bag
                view_x = target_bag_pos[0] + self.camera_gripper_mm[0]
                view_y = target_bag_pos[1] + self.camera_gripper_mm[1]

                self.uart.axes["X"], self.uart.axes["Y"] = int(view_x), int(view_y)
                self.uart.send_data([request, self.uart.axes["X"], self.uart.axes["Y"], 0, self.uart.gripper])
                self.wait_for_arrival(request)

                # 4.2 WAIT FOR AI TO SCAN THE BAG
                time.sleep(1.5) 

                # 4.3 CALCULATE FINAL DROP POSITION (Including Avoidance + Gripper Offset)
                # This function will now take the Mapped Bag Position and return the Gripper Drop Position
                drop_x, drop_y = self.calculate_tissues_drop_pos(target_bag_pos)

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
                # Use request 3 if your protocol defines it as Gripper Command, or just 2
                self.uart.send_data([3, self.uart.axes["X"], self.uart.axes["Y"], self.uart.axes["Z"], self.uart.gripper])
                time.sleep(0.5) # Small delay to ensure gripper is fully open

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
                    print(f"  [INFO] Bag at {target_bag_pos} is complete.")
                    current_tasks.pop(nearest_idx)

            print("\n[RUN] All bags filled. Returning home...")
            self.uart.send_data([request, 0, 0, 0, 90])
        
# ===================================
# Camera detection thread
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
        self.CONF_THRESH = 0.6
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
            frame = cv2.undistort(frame, self.K, self.dist, None, self.newK)

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

            # boxes.append([x1, y1, x2 - x1, y2 - y1])
            boxes.append([x1, y1, x2, y2]) # Changed to store full coordinates for size calculation
            scores.append(score)
            class_ids.append(class_id)

        idxs = self.nms(boxes, scores)
        draw = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        is_real_bbox = False
        self.all_bag_pixels = []         # List of all bag positions (pixel)
        all_tissue_data = []        # List to store tissue data

        bags_bboxes_mm = []         # List for separated bag data
        tissues_bboxes_mm = []      # List for separated tissue data
        
        for i in idxs:
            x1_b, y1_b, x2_b, y2_b = boxes[i] # Get full coordinates
            x, y, w, h = x1_b, y1_b, x2_b - x1_b, y2_b - y1_b # Keep existing x, y, w, h logic
            
            u = (x * 2 + w) // 2
            v = (y * 2 + h) // 2

            label = self.classes[class_ids[i]]
            color_box = (0, 255, 0)
            
            # Calculate real-world size in mm for the current object
            obj_w_mm, obj_h_mm = self.get_object_size_mm([x1_b, y1_b, x2_b, y2_b])
            size_text = f"{obj_w_mm}x{obj_h_mm}mm"

            # Process the bag
            if label == "stop":
                if self.is_real_bbox(x, y, w, h, w_frame, h_frame):
                    self.all_bag_pixels.append((u, v))
                    # Store separated bag data
                    bags_bboxes_mm.append({'center': (u, v), 'bbox': [x1_b, y1_b, x2_b, y2_b], 'size': (obj_w_mm, obj_h_mm)})
                    is_real_bbox = True
                    cv2.circle(draw, (u, v), 5, (0, 0, 255), -1)        # Draw circle of bbox centers
                    cv2.circle(draw, (int(self.cx), int(self.cy)), 5, (255, 0, 0), -1)        # Draw circle of image center
                    cv2.line(draw, (int(self.cx), int(self.cy)), (u, v), (0, 255, 255), 2)
                else:
                    color_box = (0, 0, 255)
            # Process the tissue
            elif label == "go-ahead":
                # Store separated tissue data
                tissues_bboxes_mm.append({'center': (u, v), 'bbox': [x1_b, y1_b, x2_b, y2_b], 'size': (obj_w_mm, obj_h_mm)})
                # Store center and full bbox for mm size calculation later
                all_tissue_data.append({'center': (u, v), 'bbox': [x1_b, y1_b, x2_b, y2_b]})
                cv2.circle(draw, (u, v), 5, (0, 255, 0), -1)

            cv2.rectangle(draw, (x, y), (x + w, y + h), color_box, 2)
            # Display label and real size next to the bbox
            cv2.putText(draw, f"{label} {size_text}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if is_real_bbox and self.all_bag_pixels:
            # print(f"--- Frame Debug ---")
            # print(f"Pixels detected: {all_bag_pixels}")
            
            self.mapping.update_bag_from_pixel(self.all_bag_pixels)          # Update position of bag in frame (pixel) 
            camera_bag_mm_list = self.mapping.get_camera_bag_mm()       # Calculate distance between camera and bag (mm)
            list_final_positions = self.mapping.compute_final_base_position()   # Calculate final position for moving gripper to that

            if list_final_positions is not None:
                # print(f"Final Positions (Base Frame): {list_final_positions}")
                for i, pos in enumerate(list_final_positions):
                    y_offset = 30 + (i * 30)
                    cv2.putText(draw, f"Bag-{i+1}: X={pos[0]:.1f}, Y={pos[1]:.1f} mm",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            for j, (u_p, v_p) in enumerate(self.all_bag_pixels):
                if j < len(camera_bag_mm_list):
                    mm_val = camera_bag_mm_list[j]
                    cv2.putText(draw, f"({mm_val[0]:.1f}, {mm_val[1]:.1f})mm", 
                                (u_p + 8, v_p + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Store separated results to class attributes for global access
        self.last_detected_bags = bags_bboxes_mm
        self.last_detected_tissues = tissues_bboxes_mm
        
        if not is_real_bbox:
            with self.mapping.lock:
                self.mapping.camera_bag_mm = np.array([np.nan, np.nan], dtype=np.float32)
        
        return draw