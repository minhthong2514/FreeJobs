import cv2
import time

def test_camera():
    # Pipeline tối ưu hóa sử dụng phần cứng NVIDIA (NVDEC và NVVIC)
    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
    

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Lỗi: Không thể mở GStreamer với nvv4l2decoder. Thử lại...")
        return

    print("Camera đang chạy với tăng tốc phần cứng...")
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Jetson Hardware Decoded", frame)

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
