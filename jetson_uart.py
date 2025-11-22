import serial
import time


HEADER_1st = 0xAA
HEADER_2st = 0X55
data = []

# Jetson will send something if it receive true two header.
def read_header():
    StringData = ser.readline().decode().strip()
    data.append(StringData)

    if (data[0] == HEADER_1st):
        if (data[1] == HEADER_2st):
            data.append("Hello MCU.\n")
            ser.write(data.encode('UTF-8')) 
            print(f"Sent data to MCU: {data}")
                        
def send_header():
    pass
    
#Set your port which jetson detected
ser = serial.Serial(port= "/dev/ttyUSB0", baudrate= 115200, timeout= 1)

while True:
    if ser.in_waiting == 0:
        continue                # jetson always read data ignoring none data
    StringData = ser.readline().decode().strip()
    print(f"Received data from MCU: {StringData}", end = " | ")

    text = "Hello MCU.\n"
    ser.write(text.encode('UTF-8'))  # jetson always send data
    print(f"Sent data to MCU: {text}")
    time.sleep(1)

ser.close()

# This code is not optimized