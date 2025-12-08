import serial
import time
from uart_protocol import UART
import struct
import logging

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

while True:
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
        else:
            print('ERROR: Incomplete packet received.')
    
    time.sleep(3)

    uart.send_data(2, 40, 30, 50, ser) #move X
    time.sleep(3)

    uart.send_data(3, 40, 30, 50, ser) #move Y
    time.sleep(3)

    uart.send_data(4, 40, 30, 50, ser) #move Z
    time.sleep(3)

    uart.send_data(0, 0, 0, 0, ser) # Ask current position
    time.sleep(0.01)

    # if ser.read(1) == bytes([HEADER_1st]):
    #     logging.info('header 1 ok')
    #     if ser.read(1) == bytes([HEADER_2st]): 
    #         logging.info('header 2 ok')
    #         logging.info(f"Waiting for packets (Size: {PACKET_SIZE} bytes)...")
    #         payload = ser.read(PACKET_SIZE)
    #         req_val, x_pos_mm, y_pos_mm, z_pos_mm = uart.get_data(payload, STRUCT_FORMAT, PACKET_SIZE)
    #         req_name = REQUEST_TYPES.get(req_val, "UNKNOWN")
               
    #         print("-" * 30)
    #         logging.info(f"Request Type : {req_name} ({req_val})")
    #         logging.info(f"X Position   : {x_pos_mm} mm")
    #         logging.info(f"Y Position   : {y_pos_mm} mm")
    #         logging.info(f"Z Position   : {z_pos_mm} mm")
    #     else:
    #         print('ERROR: Incomplete packet received.')

    time.sleep(0.01)


    
