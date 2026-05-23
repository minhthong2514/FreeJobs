import cv2
import os

pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Can't open the camera.")
    exit()

path = r"/home/minhthong/Desktop/code/farmbot/calib-camera/result_imgs"

def get_next_available_index(folder_path):
    index = 1
    while True:
        filename = f"img-{index}.jpg"
        if not os.path.exists(os.path.join(folder_path, filename)):
            return index
        index += 1

print("Press number 1 for shooting / press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot receive frame from camera.")
        break
    current_idx = get_next_available_index(path)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        filename = f"img-{current_idx}.jpg"
        img_path = os.path.join(path, filename)
        cv2.imwrite(img_path, frame)
        print(f"Save img at: {img_path}")
        
    elif key == ord('q'):
        print("Exit.")
        break

cap.release()
cv2.destroyAllWindows()