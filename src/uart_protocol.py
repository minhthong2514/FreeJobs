""" 
This file is using for uart protocol between Jetson nano and MCU.
The data frame is in the following format: [request, x_pos_mm, y_pos_mm, z_pos_mm].
"""

import struct
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class UART:
    # The first is initialized the params.
    def __init__(self, HEADER_1st, HEADER_2st, ser):
        self.HEADER_1st = HEADER_1st
        self.HEADER_2st = HEADER_2st
        self.ser = ser
        self.buffer = []
        self.request = None
        self.x_pos_mm = 0
        self.y_pos_mm = 0
        self.z_pos_mm = 0
        self.gripper = 0
        self.current_fols = None
        self.current_bag = None
        self.axes = {"X": 0, "Y": 0, "Z": 0}

    # This function is used for receiving data from MCU.
    def get_data(self, payload, struct_format, packet_size):                                    # payload is data read, struct_format is amount of bytes, packet_size is the quantity of packet
        if len(payload) == packet_size:                                                         # ckeck payload length is equal to packet size, 
            request, x_pos_mm, y_pos_mm, z_pos_mm, gripper, current_fols, current_bag  = struct.unpack(struct_format, payload)       # unpacking the payload.
            return request, x_pos_mm, y_pos_mm, z_pos_mm, gripper, current_fols, current_bag
     
    # This function is used for sending data to MCU.
    def send_data(self, buffer):
        STRUCT_FORMAT = "<BHHHH"                                 # frame must be little endian and 7 bytes.
        PACKET_SIZE = struct.calcsize(STRUCT_FORMAT)            # return the quantity of current packet's byte.
        # buffer = [request, x_pos_mm, y_pos_mm, z_pos_mm]        # this is a temporary data storage for easy updating of new data. 

        if self.ser is None:                                         
            print("No connected to serial port.")

        # Send HEADER 1 to MCU
        self.ser.write(bytes([self.HEADER_1st]))

        # Send HEADER 2 to MCU
        self.ser.write(bytes([self.HEADER_2st]))

        # Send Payload
        payload = struct.pack(STRUCT_FORMAT, *buffer)           # packaging the buffer for sending.
        # logging.info(payload)
        if len(payload) == PACKET_SIZE:                        
            self.ser.write(payload)

    # This function is used for processing to standard formatting.
    def parse_axis_value(self, user_input):
        parts = [p.strip() for p in user_input.split(",")]

        if len(parts) % 2 != 0:
            print("Invalid format!")
            return 0

        updated_axes = set()

        for i in range(0, len(parts), 2):
            axis = parts[i].upper()
            value = parts[i + 1]

            if axis not in self.axes:
                print(f"Invalid axis: {axis}")
                return 0

            try:
                value = int(value)
            except ValueError:
                print(f"Invalid value for {axis}: {value}")
                return 0

            self.axes[axis] = value
            updated_axes.add(axis)
            print(f"{axis} = {value} mm")

        return len(updated_axes)
    
    def request_ask_current_position(self, request):
        frame = [request, self.axes["X"], self.axes["Y"], self.axes["Z"], self.gripper]
        self.send_data(frame)
 
    def request_run_sequence(self, request):
        frame = [request, self.axes["X"], self.axes["Y"], self.axes["Z"], self.gripper]
        self.send_data(frame)
        print(f"\nSend data: {frame}\n")

    # Process move_xyz mode.
    def request_move_xyz(self, request):
        while True:
            user = input("\nFormat is (Axis, Value) \nPress 'n' to send or Press 'q' to exit: ").strip().lower()

            if user == "q":
                print("\nExit MOVE XYZ mode")
                return

            if user == "n":
                frame = [request, self.axes["X"], self.axes["Y"], self.axes["Z"], self.gripper]
                self.send_data(frame)
                print(f"\nSend data : {frame}\n")
                continue
            count = self.parse_axis_value(user)
            if count == 0:
                continue

            if count == 3:
                frame = [request, self.axes["X"], self.axes["Y"], self.axes["Z"], self.gripper]
                self.send_data(frame)
                print(f"\nSend data : {frame}\n")  
    
    # Process controll gripper mode.
    def request_controll_gripper(self, request):
        while True:
            print("\nPress 'q' to exit | ", end ='')
            user = input("Degree values of gripper (0-180): ").strip().lower()
            # if user press q, exit mode
            if user == "q":
                print("\nExit Controll Gripper mode")
                return 
            else:
                self.gripper = int(user)
                # Value must be in range 0-180 degree
                if (self.gripper >= 0 and self.gripper <= 180):
                    frame = [request, self.axes["X"], self.axes["Y"], self.axes["Z"], self.gripper]
                    self.send_data(frame)
                    print(f"\nSend data : {frame}\n")
                    continue
                else:
                    print("\nGripper value must be in range 0-180.\n")
                    continue
