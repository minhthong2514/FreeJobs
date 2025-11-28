# from uart_protocol import UART
# import time
# import serial


# ser = serial.Serial(port= "/dev/ttyUSB0", baudrate= 115200, timeout= 1)

# # int value
# HEADER_1st = 0xAA       # 170
# HEADER_2st = 0X55

# uart = UART(HEADER_1st, HEADER_2st)

# data_flow = [0xAA, 0x55, 1, 5, 10, 15,8,9,
#              0xAA, 0x55, 2, 5, 45, 44,3]
# # print(data_flow)


# # for frame in data_flow:
# #     result = uart.get_data(frame)
# #     if result:
# #         print(result)


# while True:
#     # if ser.in_waiting == 0:
#     #     continue                # jetson always read data ignoring none data
#     # StringData = ser.read()
#     # StringData = int.from_bytes(StringData, 'little')
#     # # print(StringData)
#     # # print(type(StringData))
#     # frame_payload = uart.get_data(StringData)
#     # if not frame_payload:
#     #     continue
#     # print(frame_payload)

#     # if StringData == 0x55:
#     #     print('true')
#     # else:
#     #     print('false')

#     # for data in data_flow:
#     #     frame_payload = uart.get_data(data)
#     #     if not frame_payload:
#     #         continue
#     #     print(frame_payload)   
         
#     uart.send_data(1,2,3,4, ser = ser)
#     #Send data
#     # frame = uart.send_data(ser= ser)
#     # print(frame)

#     time.sleep(1)
# ser,close()


import serial
import struct
import time

# --- Configuration ---
# Update this to your correct UART port (e.g., /dev/ttyTHS1 for GPIO UART)
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200

# --- Struct Definition ---
# Corresponds to:
# typedef struct {
#    uint8_t request;   (Enum is 1 byte)
#    uint16_t x_pos_mm; (2 bytes)
#    uint16_t y_pos_mm; (2 bytes)
#    uint16_t z_pos_mm; (2 bytes)
# } newPackage_t;

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

# --- Enum Mapping (For easy reading) ---
REQUEST_TYPES = {
    0: "ASK_CURRENT_POSITION",
    1: "CMD_CURRENT_POSITION",
    2: "MOVE_X",
    3: "MOVE_Y",
    4: "MOVE_Z"
}

def read_uart():
    try:
        print(f"Opening {SERIAL_PORT} at {BAUD_RATE}...")
        # timeout=1 makes the read non-blocking (returns after 1s if no data)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Waiting for packets (Size: {PACKET_SIZE} bytes)...")

        while True:
            # 1. Wait for Start Byte 1 (0xAA)
            # ser.read(1) returns a bytes object, e.g., b'\xAA'
            if ser.read(1) == b'\xAA':
                
                # 2. Check for Start Byte 2 (0x55)
                if ser.read(1) == b'\x55':
                    
                    # 3. We found the header! Now read the struct data.
                    payload = ser.read(PACKET_SIZE)

                    # Ensure we received the full 7 bytes
                    if len(payload) == PACKET_SIZE:
                        
                        # 4. Unpack the binary data
                        # result is a tuple: (request, x, y, z)
                        req_val, x_mm, y_mm, z_mm = struct.unpack(STRUCT_FORMAT, payload)

                        # 5. Process/Display Data
                        req_name = REQUEST_TYPES.get(req_val, "UNKNOWN")
                        
                        print("-" * 30)
                        print(f"Request Type : {req_name} ({req_val})")
                        print(f"X Position   : {x_mm} mm")
                        print(f"Y Position   : {y_mm} mm")
                        print(f"Z Position   : {z_mm} mm")
                    
                    else:
                        print("Error: Incomplete packet received.")

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    read_uart()
