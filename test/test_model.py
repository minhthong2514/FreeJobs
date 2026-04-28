import cv2
import ctypes
import os
import time
from yolov5_trt import YoLov5TRT

# --- CẤU HÌNH ---
ENGINE_PATH = "/home/minhthong/Desktop/code/farmbot/models/farmbot_seg_model.engine"
PLUGIN_PATH = "libmyplugins.so"
LABELS = ["strawberry", "sweet potato"]
CONF_THRESHOLD = 0.8
IOU_THRESHOLD = 0.45

def main():
    if not os.path.exists(PLUGIN_PATH):
        print(f"Lỗi: Không tìm thấy {PLUGIN_PATH}")
        return
    ctypes.CDLL(PLUGIN_PATH)

    # Khởi tạo với categories và ngưỡng
    model = YoLov5TRT(ENGINE_PATH, LABELS, CONF_THRESHOLD, IOU_THRESHOLD)

    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Suy luận
        batch_results, infer_time = model.infer([frame])
        result_frame = batch_results[0]
        
        # Tính FPS hệ thống
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Hiển thị
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (20, 40), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Inference time: {infer_time*1000:.1f}ms", (20, 80), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Farmbot AI - High Precision", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    model.destroy()

if __name__ == "__main__":
    main()