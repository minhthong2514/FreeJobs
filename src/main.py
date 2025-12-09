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

STRUCT_FORMAT = "<BHHH"
PACKET_SIZE = struct.calcsize(STRUCT_FORMAT) # Should be 7 bytes
#This Queue will hold the response forever til need
incoming_mailbox = queue.Queue()


# int value
HEADER_1st = 0xAA       # 170
HEADER_2st = 0X55       # 85
uart = UART(HEADER_1st, HEADER_2st)
ser = serial.Serial(port= "/dev/ttyUSB0", baudrate= 115200, timeout= 1)

# --- Enum Mapping (For easy reading) ---
REQUEST_TYPES = {
    0: "ASK_CURRENT_POSITION",
    1: "CMD_CURRENT_POSITION",
    2: "MOVE_X",
    3: "MOVE_Y",
    4: "MOVE_Z"
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
                        req_val, x_pos_mm, y_pos_mm, z_pos_mm = uart.get_data(payload, STRUCT_FORMAT, PACKET_SIZE)
                        req_name = REQUEST_TYPES.get(req_val, "UNKNOWN")
                        
                        print("-" * 30)
                        print(f"Request Type : {req_name} ({req_val})")
                        print(f"X Position   : {x_pos_mm} mm")
                        print(f"Y Position   : {y_pos_mm} mm")
                        print(f"Z Position   : {z_pos_mm} mm")

                        # Put it Mailbox(Thread-safe)
                        # This saves the data in RAM safely
                        result = {"type": req_val, "x": x_pos_mm, "y": y_pos_mm, "z": z_pos_mm}
                        incoming_mailbox.put(result)
                    else:
                        print('ERROR: Incomplete packet received.')
        except:
            pass

# Start the listener
t = threading.Thread(target=serial_listener, daemon=True)
t.start()

while True:
    # if ser.read(1) == bytes([HEADER_1st]):
    #     print('header 1 ok')
    #     if ser.read(1) == bytes([HEADER_2st]): 
    #         print('header 2 ok')
    #         print(f"Waiting for packets (Size: {PACKET_SIZE} bytes)...")
    #         payload = ser.read(PACKET_SIZE)
    #         req_val, x_pos_mm, y_pos_mm, z_pos_mm = uart.get_data(payload, STRUCT_FORMAT, PACKET_SIZE)
    #         req_name = REQUEST_TYPES.get(req_val, "UNKNOWN")
               
    #         print("-" * 30)
    #         print(f"Request Type : {req_name} ({req_val})")
    #         print(f"X Position   : {x_pos_mm} mm")
    #         print(f"Y Position   : {y_pos_mm} mm")
    #         print(f"Z Position   : {z_pos_mm} mm")
    #     else:
    #         print('ERROR: Incomplete packet received.')
    
    # time.sleep(3)

    uart.send_data(1, 40, 30, 50, ser) #move X
    time.sleep(1)

    uart.send_data(2, 40, 30, 50, ser) #move Y
    time.sleep(1)

    uart.send_data(3, 40, 30, 50, ser) #move Z
    time.sleep(3)

    uart.send_data(1, 80, 30, 50, ser) #move X
    time.sleep(1)

    uart.send_data(2, 80, 60, 50, ser) #move Y
    time.sleep(1)

    uart.send_data(3, 40, 30, 10, ser) #move Z
    time.sleep(3)

    # uart.send_data(0, 0, 0, 0, ser) # Ask current position
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


    
