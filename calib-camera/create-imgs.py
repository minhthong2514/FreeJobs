import cv2
import os

# Open camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Can't open the camera.")
    exit()

# Find number at start
img_count = 1
while os.path.exists(f"img{img_count}.jpg"):
    img_count += 1
print("Press number 1 for shooting / press 'q' to exit.")

# The path of images folder.
path = r"F:\FARMBOT\calib-camera\images"

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot receive frame from camera.")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        filename = f"fruit{img_count}.jpg"
        img_path = os.path.join(path,filename)
        cv2.imwrite(filename, frame)
        print(f"Save img at: {filename}")
        img_count += 1
    elif key == ord('q'):
        print("Exit.")
        break

cap.release()
cv2.destroyAllWindows()
