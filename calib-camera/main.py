import numpy as np
from ultralytics import YOLO
import cv2

# Load model YOLOv8 segmentation
model = YOLO(r"F:\University\Nam_ba\Do_an_TGMT\YOLOv8-seg\yolov8s-seg\runs\segment\train\weights\best.pt")

# Màu theo class ID
class_colors = {
    0: (0, 0, 255),      # Đỏ (freshapple)
    1: (0, 255, 255),    # Vàng (freshbanana)
    2: (0, 165, 255),    # Cam (freshorange)
    3: (0, 0, 139),      # Đỏ thẫm (rottenapple)
    4: (0, 140, 140),    # Vàng thẫm (rottenbanana)
    5: (0, 100, 200),    # Cam thẫm (rottenorange)
}

# Camera intrinsic parameters
fx = np.array([1.16980682e+03], dtype=np.float32)
fy = np.array([1.17361892e+03], dtype=np.float32)
cx = np.array([9.80014068e+02], dtype=np.float32)
cy = np.array([5.19590613e+02], dtype=np.float32)
Z = 20.0  # cm

# Hàm tính khoảng cách Euclidean
def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Đọc video
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    centers = []

    # Dự đoán trên từng frame
    results = model.predict(source=frame, imgsz=640, conf=0.6, verbose=False)

    for result in results:
        masks = result.masks
        boxes = result.boxes
        names = result.names

        if masks is not None and boxes is not None:
            for i in range(len(masks.data)):
                mask = masks.data[i].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                cls_id = int(boxes[i].cls.item())
                color = class_colors.get(cls_id, (255, 255, 255))

                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                label = names[cls_id]

                centers.append((label, (u, v)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(frame, (u, v), 5, (255, 255, 255), -1)

    # Sau khi có đủ 2 vật thì tính và hiển thị khoảng cách
    if len(centers) >= 2:
        (_, (u1, v1)) = centers[0]
        (_, (u2, v2)) = centers[1]

        # Tính tọa độ 3D
        X1 = (u1 - cx) * Z / fx
        Y1 = (v1 - cy) * Z / fy
        X2 = (u2 - cx) * Z / fx
        Y2 = (v2 - cy) * Z / fy

        p1 = np.array([X1, Y1])
        p2 = np.array([X2, Y2])

        distance = euclidean_dist(p1, p2)

        # Vẽ line và hiển thị khoảng cách
        cv2.line(frame, (u1, v1), (u2, v2), (0, 255, 0), 2)
        mid_x = (u1 + u2) // 2
        mid_y = (v1 + v2) // 2
        cv2.putText(frame, f"{distance:.2f} cm", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    # Hiển thị frame
    cv2.imshow("Distance between objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
