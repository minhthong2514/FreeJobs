import serial
import time
from uart_protocol import UART
import struct
import logging
import threading
import queue

# Format String:
# <  = Little Endian (Standard for ESP32/Jetson)
# B = 1 byte (uint8)
# b = 1 byte (int8)
# H = 2 byte (uint16)
# h = 2 byte (int16)
# I = 4 byte (uint32)
# i = 4 byte (int32)

STRUCT_FORMAT = "<BHHHHHH" #13 BYTE
PACKET_SIZE = struct.calcsize(STRUCT_FORMAT) # Should be 11 bytes
#This Queue will hold the response forever til need
incoming_mailbox = queue.Queue()


# int value
HEADER_1st = 0xAA       # 170
HEADER_2st = 0X55       # 85
ser = serial.Serial(port= "/dev/ttyUSB0", baudrate= 115200, timeout= 1)
uart = UART(HEADER_1st, HEADER_2st, ser=ser)

# --- Enum Mapping (For easy reading) ---
REQUEST_TYPES = {
    0: "ASK_CURRENT_POSITION",
    1: "RUN_SEQUENCE",
    2: "MOVE_XYZ",
    3: "CONTROL_GRIPPER",
    4: "MOTION_COMPLETE" 
}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# This functions run in the background of the time
def serial_listener():
    while True:
        try:
            # Check if data exist
            if ser.in_waiting > 0:
                 if ser.read(1) == bytes([HEADER_1st]):
                    print('header 1 ok')
                    if ser.read(1) == bytes([HEADER_2st]): 
                        print('header 2 ok')
                        print(f"Waiting for packets (Size: {PACKET_SIZE} bytes)...")
                        payload = ser.read(PACKET_SIZE)
                        req_val, x_pos_mm, y_pos_mm, z_pos_mm, gripper, current_foil, current_bag = uart.get_data(payload, STRUCT_FORMAT, PACKET_SIZE)
                        req_name = REQUEST_TYPES.get(req_val, "UNKNOWN")
                        
                        print("-" * 30)
                        print(f"Request Type : {req_name} ({req_val})")
                        print(f"X Position   : {x_pos_mm} mm")
                        print(f"Y Position   : {y_pos_mm} mm")
                        print(f"Z Position   : {z_pos_mm} mm")
                        print(f"Current Gripper: {gripper} deg")
                        print(f"Current Foils: {current_foil} pcs")
                        print(f"Current Bags : {current_bag} pcs")
                        # Put it Mailbox(Thread-safe)
                        # This saves the data in RAM safely
                        result = {"type": req_val, "x": x_pos_mm, "y": y_pos_mm, "z": z_pos_mm, "gripper": gripper,"foil": current_foil, "bag": current_bag}
                        incoming_mailbox.put(result)
                        
                    else:
                        print('ERROR: Incomplete packet received.')
        except:
            pass


# Start the listener
t1 = threading.Thread(target=serial_listener, daemon=True)
t1.start()

while True:
    print("\n" + str(REQUEST_TYPES))
    cmd = input("\nCHOOSE REQUEST OR PRESS 's' TO STOP PROGRAM: ").lower()

    if cmd == "s":
        break
    else:
        cmd = int(cmd)
        if cmd == 0:
            uart.request_ask_current_position(request=cmd)
        elif cmd == 1:
            uart.request_run_sequence(request=cmd)
        elif cmd == 2:
            uart.request_move_xyz(request=cmd) 

        elif cmd == 3:
            uart.request_controll_gripper(request=cmd)

        elif cmd == 4:
            continue
        else:
            print("\nInvalid command!\n")


    # time.sleep(1)

    # uart.send_data(4, 40, 30, 0, ser) # run sequency automatically
    # time.sleep(1)

    # uart.send_data(4, 40, 30, 10, ser) # run sequency automatically
    # time.sleep(1)

# Check the mailbox
# block=True means "Wait here until a data arrives"
# timeout=10 means "Wait 10 second, then give up"

try:
    response = incoming_mailbox.get(block=True, tineout=10)

    if response['type'] == 4: # Assuing 5 is MOTION_COMPLETE
        print(f"Success! Motor Finished at X: {response['x']}")
    else:
        print("Got some other msg")
except queue.Empty:
    print("Timeout! Motor took too long or MCU is crashed")


    
