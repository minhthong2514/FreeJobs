import serial
import time
from uart_protocol import UART
import struct
import logging
import threading
import queue
import camera
import cv2
# Format String:
# <  = Little Endian (Standard for ESP32/Jetson)
# B = 1 byte (uint8)
# b = 1 byte (int8)
# H = 2 byte (uint16)
# h = 2 byte (int16)
# I = 4 byte (uint32)
# i = 4 byte (int32)

STRUCT_FORMAT = "<BHHHHH" #11 BYTE
PACKET_SIZE = struct.calcsize(STRUCT_FORMAT) # Should be 11 bytes
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
    1: "RUN_SEQUENCE",
    2: "MOVE_X",
    3: "MOVE_Y",
    4: "MOVE_Z",
    5: "OPEN_GRIPPER",
    6: "CLOSE_GRIPPER",
    7: "MOTION_COMPLETE"
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
                        req_val, x_pos_mm, y_pos_mm, z_pos_mm, current_foil, current_bag = uart.get_data(payload, STRUCT_FORMAT, PACKET_SIZE)
                        req_name = REQUEST_TYPES.get(req_val, "UNKNOWN")
                        
                        print("-" * 30)
                        print(f"Request Type : {req_name} ({req_val})")
                        print(f"X Position   : {x_pos_mm} mm")
                        print(f"Y Position   : {y_pos_mm} mm")
                        print(f"Z Position   : {z_pos_mm} mm")
                        print(f"Current Foils: {current_foil} pcs")
                        print(f"Current Bags : {current_bag} pcs")
                        # Put it Mailbox(Thread-safe)
                        # This saves the data in RAM safely
                        result = {"type": req_val, "x": x_pos_mm, "y": y_pos_mm, "z": z_pos_mm, "foil": current_foil, "bag": current_bag}
                        incoming_mailbox.put(result)
                    else:
                        print('ERROR: Incomplete packet received.')
        except:
            pass
def test():
    while True:
        dataX = int(input('X value: '))
        dataY = int(input('Y value: '))
        dataZ = int(input('Z value: '))
        uart.send_data(2, dataX, dataY, dataZ, ser) # run sequency automatically
        uart.send_data(3, dataX, dataY, dataZ, ser) # run sequency automatically
        uart.send_data(4, dataX, dataY, dataZ, ser) # run sequency automatically

# Start the listener
t1 = threading.Thread(target=serial_listener, daemon=True)
t1.start()
t2 = threading.Thread(target=test, daemon=True)
t2.start()


cap = cv2.VideoCapture(0)   # change index if needed

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Pixel to World")
cv2.setMouseCallback("Pixel to World", camera.mouse_callback)
clicked_points = []   # list of dict

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(camera.K, camera.dist, (w,h), 1)
    undistorted = cv2.undistort(frame, camera.K, camera.dist, None, newK)

    display_img = undistorted.copy()

    # Draw all clicked points
    camera.draw_points(display_img)
    cv2.imshow("origin frame", frame)

    cv2.imshow("Pixel to World", display_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r'):
        origin_world = None
        clicked_points.clear()
        print("Origin and points reset")

    
cap.release()
cv2.destroyAllWindows()

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


    
