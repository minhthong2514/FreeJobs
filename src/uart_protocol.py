class UART:
    # The first is initialized the params
    def __init__(self, HEADER_1st, HEADER_2st):
        self.HEADER_1st = HEADER_1st
        self.HEADER_2st = HEADER_2st
        self.state = 0           
        self.buffer = []
        self.value_x = 0
        self.value_y = 0
        self.value_z = 0
        # self.frame_xyz = [self.value_x, self.value_y, self.value_z]

    # The function using for update data of XYZ value.
    def Update_XYZ(self, value_x, value_y, value_z):
        self.value_x = value_x
        self.value_y = value_y
        self.value_z = value_z
    # Reset buffer and state
    def reset_buffer(self):
        self.state = 0
        self.buffer = []
    # The function using for receiving data from MCU.
    def get_data(self, byte):                       # The byte variable can have many different data types (now is single byte, it can be list, tuple or dics...)
        # print(byte)                               
        if self.state == 0:                         # Check index 0 is HEADER1
            if byte == self.HEADER_1st:     
                self.state = 1
                print(byte)
            else:
                self.state = 0
        elif self.state == 1:                       # Check index 1 is HEARDER2
            if byte == self.HEADER_2st:
                self.state = 2
                print(byte)

            else:
                self.state = 0

                
            # frame = [request, x_pos_mm, y_pos_mm, z_pos_mm]
                
        elif self.state == 2:                       # if frame is satisfied, add data to buffer and update data.
            print(byte)
            # self.buffer.append(byte)
            # if (len(self.buffer) == 3):
            #     self.Update_XYZ(self.buffer[0], self.buffer[1], self.buffer[2])
            #     frame_xyz = [self.value_x, self.value_y, self.value_z]
            #     self.reset_buffer()
            #     return frame_xyz
            # else:
            #     return None
            
    def send_data(self, value_x=None, value_y=None, value_z=None, ser=None):
        frame = [
            self.HEADER_1st,
            self.HEADER_2st,
            # [value_x, value_y, value_z]      # This code maybe get error because python cannot byte a list.
        ]
        if ser is not None:
            ser.write(bytes(frame))
        else:
            return frame
