from uart_protocol import UART
import time
import serial

ser = serial.Serial(port= "/dev/ttyUSB0", baudrate= 115200, timeout= 1)

# int value
HEADER_1st = 0xAA       # 170
HEADER_2st = 0X55

uart = UART(HEADER_1st, HEADER_2st)

# data_flow = [0xAA, 0x55, 5, 10, 15,
#           0xAA, 0x55, 20, 25, 30]
# print(data_flow)


# for frame in data_flow:
#     result = uart.get_data(frame)
#     if result:
#         print(result)


while True:
    # if ser.in_waiting == 0:
    #     continue                # jetson always read data ignoring none data
    StringData = ser.read()
    StringData = int.from_bytes(StringData, 'little')
    # print(StringData)
    # print(type(StringData))
    # if StringData == 0x55:
    #     print('true')
    # else:
    #     print('false')
    # for data in StringData:
    #     result = uart.get_data(data)
    #     if result:
    #         print(data)
    uart.get_data(byte=StringData)
    # #Send data
    # frame = uart.send_data(ser= ser)
    # print(frame)

    time.sleep(1)
ser,close()