import cv2
import os


label = "bag"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Can't open the camera.")
    exit()

path = r"/home/minhthong/Desktop/code/farmbot/models/datasets/bags"

def get_next_available_index(folder_path):
    index = 1
    while True:
        filename = f"{label}-{index}.jpg"
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
        filename = f"{label}-{current_idx}.jpg"
        img_path = os.path.join(path, filename)
        cv2.imwrite(img_path, frame)
        print(f"Save img at: {img_path}")
        
    elif key == ord('q'):
        print("Exit.")
        break

cap.release()
cv2.destroyAllWindows()