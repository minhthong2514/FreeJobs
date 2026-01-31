import serial
import time
from uart_protocol import UART
import logging
import threading
import queue
from camera import Mapping, CameraDetect

#This Queue will hold the response forever til need
incoming_mailbox = queue.Queue()

# Initializ serial port
ser = serial.Serial(port= "/dev/ttyUSB0", baudrate= 115200, timeout=1)
uart = UART(ser=ser, incoming_mailbox=incoming_mailbox)

# Init Mapping
mapping = Mapping(uart=uart)
ONNX_MODEL_PATH = "/home/minhthong/Desktop/code/farmbot/src/traffic_sign_model.onnx"

# Init camera thread
camera = CameraDetect(
    model_onnx_path=ONNX_MODEL_PATH,
    mapping=mapping,
    enable_display=True
)

# Start threads
camera.start()          # detect thread
camera.start_display()  # display thread

# --- Enum Mapping (For easy reading) ---
REQUEST_TYPES = {
    0: "ASK_CURRENT_POSITION",
    1: "RUN_SEQUENCE",
    2: "MOVE_XYZ",
    3: "CONTROL_GRIPPER",
    4: "SETUP_MATERIAL",
    5: "HOMING",
    6: "MOTION_COMPLETE",
    7: "MAPPING",
    8: "RUN"
}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Start the listener
t1 = threading.Thread(
    target=uart.serial_listener,
    args=(REQUEST_TYPES, incoming_mailbox),
    daemon=True
)
t1.start()

waiting_for_base_camera = False

while True:
    print("\n" + str(REQUEST_TYPES))
    cmd = input("\nCHOOSE REQUEST OR PRESS 's' TO STOP PROGRAM: ").lower()

    if cmd == "s":
        break

    if cmd == "":
        continue

    try:
        cmd = int(cmd)
        if cmd == 0:
            waiting_for_base_camera = True
            uart.request_ask_current_position(request=cmd)
            try:
                result = incoming_mailbox.get()

                # Nếu đang chờ mode 0 → update base_camera
                if waiting_for_base_camera:
                    x = result["Current_X"]
                    y = result["Current_Y"]

                    mapping.update_base_camera_position(x, y)
                    waiting_for_base_camera = False

            except queue.Empty:
                pass
        elif cmd == 1:
            uart.request_run_sequence(request=cmd)
            
        elif cmd == 2:
            uart.request_move_xyz(request=cmd) 

        elif cmd == 3:
            uart.request_controll_gripper(request=cmd)

        elif cmd == 4: #Package 27 Bytes
            bags = [1, 2, 60, 30, 50, 30,]
            foils = [2, 60, 3, 10, 10]
            holder = [100, 60, 30]

            uart.send_setup_cmd(bags, foils, holder)

        elif cmd == 5:
            uart.request_homing(request=cmd)

        elif cmd == 6:
            continue
        elif cmd == 7:
            mapping.mapping()
        elif cmd == 8:
            mapping.run()
        else:
            print("\nInvalid command!\n")
            
    except ValueError:
        print("\n[!] Please enter a valid number or 's'.")

    # ---------- HANDLE RESPONSE FROM ESP ----------
    
    # time.sleep(1)

    # uart.send_data(4, 40, 30, 0, ser) # run sequency automatically
    # time.sleep(1)

    # uart.send_data(4, 40, 30, 10, ser) # run sequency automatically
    # time.sleep(1)



# Check the mailbox
# block=True means "Wait here until a data arrives"
# timeout=10 means "Wait 10 second, then give up"

# try:
#     response = incoming_mailbox.get(block=True, timeout=10)

#     if response['type'] == 4: # Assuing 5 is MOTION_COMPLETE
#         print(f"Success! Motor Finished at X: {response['x']}")
#     else:
#         print("Got some other msg")
# except queue.Empty:
#     print("Timeout! Motor took too long or MCU is crashed")
