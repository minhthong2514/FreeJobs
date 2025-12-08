from ultralytics import YOLO
import cv2
import numpy as np

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

# Đường dẫn ảnh cần dự đoán
image_path = r"F:\calib_camera\fruit2.jpg"

# Dự đoán trên ảnh
results = model.predict(source=image_path, imgsz=640, conf=0.6, verbose=False)

# Lặp qua các kết quả (trường hợp predict 1 ảnh, thường chỉ có 1 phần tử)
for result in results:
    frame = result.orig_img.copy()
    masks = result.masks
    boxes = result.boxes
    names = result.names
    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
    cropped = np.zeros_like(frame, dtype=np.uint8)

    if masks is not None and boxes is not None:
        for i in range(len(masks.data)):
            mask = masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            # Resize mask về kích thước của frame
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            cls_id = int(boxes[i].cls.item())
            color = class_colors.get(cls_id, (255, 255, 255))

            # Tạo mask màu để overlay
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask > 0, color[c], 0)

            # Overlay lên khung hình
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            # Vẽ box và label
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
            label = names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Hiển thị kết quả
    cv2.imshow("YOLOv8 Segmentation Result", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
