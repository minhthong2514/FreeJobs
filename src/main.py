import serial
from uart_protocol import UART
import logging
import threading
import queue
from camera import Mapping, CameraDetect
import ctypes

class FarmBotSystem:
    def __init__(self):
        # --- 1. Initialize Shared Resources ---
        self.incoming_mailbox = queue.Queue()
        self.REQUEST_TYPES = {
            0: "ASK_POSITION", 1: "RUN_SEQ", 2: "MOVE_XYZ", 
            3: "GRIPPER", 4: "SETUP", 5: "HOMING", 
            6: "MAPPING", 7: "RUN", 8: "RUN_DEMO"
        }
        
        # --- 2. Initialize Hardware & Software Modules ---
        # Setup Serial communication with MCU (ESP32)
        self.ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, timeout=1)
        self.uart = UART(ser=self.ser, incoming_mailbox=self.incoming_mailbox)
        
        # Initialize Coordinate Mapping logic
        self.mapping = Mapping(uart=self.uart)
        
        # # Load plugin file
        try:
            ctypes.CDLL("libmyplugins.so")
        except Exception as e:
            print(f"Warning: Could not load libmyplugins.so. Error: {e}")

        # Initialize Computer Vision module (engine model)
        ENGINE_MODEL_PATH = "../models/nano/farmbot_seg_model.engine"

        self.camera = CameraDetect(
            engine_path=ENGINE_MODEL_PATH,
            mapping=self.mapping,
            enable_display=True
        )
        # # Connect class camera to class mapping
        self.mapping.import_camera_to_mapping(self.camera)
        
        self.is_running = True

    def flush_mailbox(self):
        while not self.incoming_mailbox.empty():
            try:
                self.incoming_mailbox.get_nowait()
            except queue.Empty:
                break

    def menu_thread(self):
        while self.is_running:
            print("\n" + "="*40)
            print(f"AVAILABLE COMMANDS: {self.REQUEST_TYPES}")
            user_input = input("ENTER COMMAND (or 's' to stop): ").lower().strip()

            if user_input == 's':
                self.is_running = False
                break
            
            if user_input == '':
                continue

            try:
                cmd = int(user_input)
                self.handle_logic(cmd)
            except ValueError:
                print("[!] Invalid input. Please enter a numeric command.")

    def handle_logic(self, cmd):
        self.flush_mailbox() # Clear queue before sending a new request

        if cmd == 0:
            # Request current coordinates from MCU
            self.uart.request_ask_current_position(request=0)
            try:
                # Wait up to 2 seconds for a response
                result = self.incoming_mailbox.get(timeout=2)
                # Type 6 represents a Motion/Position update packet
                if result["type"] == 6:
                    print(result)
                    self.mapping.update_base_camera_position(result["Current_X"], result["Current_Y"])
                    target_label = self.camera.get_current_detected_label()
                    raw_final_position = self.mapping.get_final_position(result, object_label=target_label)
                    print(f"\nRaw final positions: {raw_final_position}")
                    # print(f"\n[OK] Updated Base Pos: X={result['Current_X']}, Y={result['Current_Y']}")
            except queue.Empty:
                print("\n[TIMEOUT] No response from MCU for command 0")

        elif cmd == 2:
            self.uart.request_move_xyz(request=2)
        elif cmd == 3:
            self.uart.request_controll_gripper(request=3)
        elif cmd == 5:
            self.uart.request_homing(request=5)
        elif cmd == 6:
            self.mapping.mapping()
        elif cmd == 7:
            self.mapping.run()
        elif cmd == 8:
            self.mapping.run_demo()
        else:
            print(f"\n[!] Command {cmd} is not yet implemented or invalid.")

    def run_system(self):
        print("--- System Starting ---")

        # 1. UART Listener Thread (Daemon: closes automatically when main thread exits)
        t_uart = threading.Thread(
            target=self.uart.serial_listener, 
            args=(self.REQUEST_TYPES, self.incoming_mailbox), 
            daemon=True
        )
        
        # 2. Menu Interface Thread
        t_menu = threading.Thread(target=self.menu_thread)

        # # 3. Start Vision Processing (CameraDetect handles its own internal threading)
        self.camera.start_camera()
        self.camera.start_display()

        # Launch threads
        t_uart.start()
        t_menu.start()

        # Keep main thread alive until user stops the menu
        t_menu.join()
        
        # Cleanup
        self.camera.stop_camera()
        print("--- System Shutdown Successful ---")

if __name__ == "__main__":
    # Setup logging format
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    farmbot = FarmBotSystem()
    farmbot.run_system()
