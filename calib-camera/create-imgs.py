import cv2
import os

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can't open the camera.")
    exit()

path = r"/home/minhthong/Desktop/code/farmbot/calib-camera/images"

img_count = 1
while os.path.exists(os.path.join(path, f"img-{img_count}.jpg")):
    img_count += 1

print("Press number 1 for shooting / press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot receive frame from camera.")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        filename = f"img-{img_count}.jpg"
        img_path = os.path.join(path, filename)
        cv2.imwrite(img_path, frame)
        print(f"Save img at: {img_path}")
        img_count += 1

    elif key == ord('q'):
        print("Exit.")
        break

cap.release()
cv2.destroyAllWindows()