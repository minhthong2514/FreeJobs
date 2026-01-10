import serial
import time

ser = serial.Serial(
    "/dev/ttyUSB0",
    115200,
    timeout=1,
    rtscts=False,
    dsrdtr=False
)

ser.setDTR(False)
ser.setRTS(False)

while True:
    ser.write(b'\xAA\x55\x01\x00\x00\x00')
    time.sleep(1)
