""" 
This file is using for uart protocol between Jetson nano and MCU.
The data frame is in the following format: [request, x_pos_mm, y_pos_mm, z_pos_mm].
"""

import struct

class UART:
    # The first is initialized the params.
    def __init__(self, HEADER_1st, HEADER_2st):
        self.HEADER_1st = HEADER_1st
        self.HEADER_2st = HEADER_2st
        self.buffer = []
        self.request = None
        self.x_pos_mm = None
        self.y_pos_mm = None
        self.z_pos_mm = None
    
    # This function is used for receiving data from MCU.
    def get_data(self, payload, struct_format, packet_size):                                    # payload is data read, struct_format is amount of bytes, packet_size is the quantity of packet
        if len(payload) == packet_size:                                                         # ckeck payload length is equal to packet size, 
            request, x_pos_mm, y_pos_mm, z_pos_mm = struct.unpack(struct_format, payload)       # unpacking the payload.
            return request, x_pos_mm, y_pos_mm, z_pos_mm
    
    
    def send_data(self, request, x_pos_mm, y_pos_mm, z_pos_mm, ser=None):
        STRUCT_FORMAT = "<BHHH"                                 # frame must be little endian and 7 bytes.
        PACKET_SIZE = struct.calcsize(STRUCT_FORMAT)            # return the quantity of current packet's byte.
        buffer = [request, x_pos_mm, y_pos_mm, z_pos_mm]        # this is a temporary data storage for easy updating of new data. 

        if ser is None:                                         
            print("No connected to serial port.")

        # Send HEADER 1 to MCU
        ser.write(bytes([self.HEADER_1st]))

        # Send HEADER 2 to MCU
        ser.write(bytes([self.HEADER_2st]))

        # Send Payload
        payload = struct.pack(STRUCT_FORMAT, *buffer)           # packaging the buffer for sending.
        print(payload)
        if len(payload) == PACKET_SIZE:                        
            ser.write(payload)
