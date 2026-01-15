""" 
This file is using for uart protocol between Jetson nano and MCU.
The data frame is in the following format: [request, x_pos_mm, y_pos_mm, z_pos_mm].
"""

# Format String:
# <  = Little Endian (Standard for ESP32/Jetson)
# B = 1 byte (uint8)
# b = 1 byte (int8)
# H = 2 byte (uint16)
# h = 2 byte (int16)
# I = 4 byte (uint32)
# i = 4 byte (int32)

import struct
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class UART:
    # The first is initialized the params.
    def __init__(self, ser):
        self.HEADER_1st = 0xAA          # 170
        self.HEADER_2st = 0x55          # 85
        self.ser = ser
        self.buffer = []
        self.request = None
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.current_pos_z = 0
        self.gripper = 0
        # self.current_fols = 0
        # self.current_bag = 0
        self.axes = {"X": 0, "Y": 0, "Z": 0}
        self.STRUCT_FORMAT = "<BHHHH"     # frame must be little endian and 9 bytes.
        self.PACKET_SIZE = struct.calcsize(self.STRUCT_FORMAT)  # Should be 9 bytes

    # This function is used for receiving data from MCU.
    def get_data(self, payload, struct_format, packet_size):                                    # payload is data read, struct_format is amount of bytes, packet_size is the quantity of packet
        if len(payload) == packet_size:                                                         # ckeck payload length is equal to packet size, 
            request, current_pos_x, current_pos_y, current_pos_z, gripper = struct.unpack(struct_format, payload)       # unpacking the payload.
            return request, current_pos_x, current_pos_y, current_pos_z, gripper
     
    # This function is used for sending data to MCU.
    def send_data(self, buffer):
        # buffer = [request, x_pos_mm, y_pos_mm, z_pos_mm]        # this is a temporary data storage for easy updating of new data. 

        if self.ser is None:                                         
            print("No connected to serial port.")

        # Send HEADER 1 to MCU
        self.ser.write(bytes([self.HEADER_1st]))

        # Send HEADER 2 to MCU
        self.ser.write(bytes([self.HEADER_2st]))

        # Send Payload
        payload = struct.pack(self.STRUCT_FORMAT, *buffer)           # packaging the buffer for sending.
        # logging.info(payload)
        if len(payload) == self.PACKET_SIZE:                        
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
        frame = [request, self.current_pos_x, self.current_pos_y, self.current_pos_z, self.gripper]
        self.send_data(frame)
        print(f"\nSend data: {frame}\n")
        return
        
 
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

    def request_homing(self, request):
            self.axes = {"X": 0, "Y": 0, "Z": 0}
            frame = [request, self.axes["X"], self.axes["Y"], self.axes["Z"], self.gripper]
            self.send_data(frame)
            print(f"\nSend data: {frame}\n")

    def send_setup_cmd(self, bags, foils, holder):
        data = [4] + bags + foils + holder
        payload = struct.pack("<B" + "B" + "B" +"H"*12, *data)
        self._transmit(payload)

    def _transmit(self, payload):
        self.ser.write(bytes([self.HEADER_1st]))
        self.ser.write(bytes([self.HEADER_2st]))
        self.ser.write(payload)

    def serial_listener(self, REQUEST_TYPES, incoming_mailbox):
        while True:
            try:
                # Check if data exisr
                if self.ser.in_waiting > 0:
                    if self.ser.read(1) == bytes([self.HEADER_1st]):
                        if self.ser.read(1) == bytes([self.HEADER_2st]): 
                            payload = self.ser.read(self.PACKET_SIZE)
                            if len(payload) == self.PACKET_SIZE:
                                req_val, self.current_pos_x, self.current_pos_y, self.current_pos_z, self.gripper = self.get_data(payload, self.STRUCT_FORMAT, self.PACKET_SIZE)
                                
                                # Display current position on monitor
                                print("\n--- RESPONSE FROM MCU ---")
                                print(f"Type : {req_val}")
                                print(f"X    : {self.current_pos_x} mm")
                                print(f"Y    : {self.current_pos_y} mm")
                                print(f"Z    : {self.current_pos_z} mm")
                                print(f"Grip : {self.gripper} deg")
                                print("\n" + str(REQUEST_TYPES), flush=True)
                                print("\nCHOOSE REQUEST OR PRESS 's' TO STOP PROGRAM: ", end="")
                                
                                # Put it Mailbox(Thread-safe)
                                # This saves the data in RAM safely
                                result = {"type": req_val, "Current_X": self.current_pos_x, "Current_Y": self.current_pos_y, "Current_Z": self.current_pos_z, "gripper": self.gripper}
                                incoming_mailbox.put_nowait(result)
                time.sleep(0.01)

            except Exception as e:
                print(f"Serial Error: {e}")
                time.sleep(1)