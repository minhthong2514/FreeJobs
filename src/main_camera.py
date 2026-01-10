from camera import Mapping
import cv2
import onnxruntime as ort
import numpy as np
import time
import threading

camera = Mapping()
cx_cam, cy_cam = camera.return_cx_cy()

Camera_params = np.load(
    "/home/minhthong/Desktop/code/farmbot/calib-camera/camera_params.npz"
)
H = Camera_params["H"]
K = Camera_params["K"]
newK = Camera_params["newK"]
dist = Camera_params["dist"]

# =========================
#        CONFIG
# =========================

ONNX_MODEL_PATH = "/home/minhthong/Desktop/code/farmbot/src/traffic_sign_model.onnx"
INPUT_SIZE = 640

CONF_THRESH = 0.9
IOU_THRESH = 0.45

classes = ['go-ahead', 'stop', 'turn-around', 'turn-left', 'turn-right']

DISPLAY_SKIP = 2   # ✅ chỉ hiển thị 1/2 frame để tránh nghẽn
fps_smooth = 0

# ========================
#  LOAD ONNX (GPU/CPU)
# ========================

providers = ort.get_available_providers()
print("Available providers:", providers)

if "CUDAExecutionProvider" in providers:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CUDAExecutionProvider"])
    print(">>> Using GPU for inference.")
else:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    print(">>> GPU NOT available → using CPU.")

input_name = session.get_inputs()[0].name

# ========================
#  PREPROCESS
# ========================

def preprocess(img):
    h, w = img.shape[:2]
    
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))

    pad_x = (INPUT_SIZE - nw) // 2
    pad_y = (INPUT_SIZE - nh) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    return img_rgb, scale, pad_x, pad_y

# ========================
#  NMS
# ========================

def nms(boxes, scores, iou_thresh=IOU_THRESH):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]

    return keep


# ========================
#  CAMERA THREAD
# ========================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame = None
running = True
display_img = None

def camera_thread():
    global frame, running, display_img
    while running:
        ret, img = cap.read()
        if ret:
            frame = img
        h, w = frame.shape[:2]
        undistorted = cv2.undistort(frame, K, dist, None, newK)

        display_img = undistorted.copy()

# ========================
#  INFERENCE THREAD
# ========================

det_frame = None

def infer_thread():
    global frame, det_frame, fps_smooth, display_img

    while running:
        if display_img is None:
            continue

        img_input, scale, pad_x, pad_y = preprocess(display_img)

        # ---- INFERENCE ----
        start = time.time()
        outputs = session.run(None, {input_name: img_input})
        infer_time = (time.time() - start) * 1000  # ms

        fps = 1000.0 / infer_time
        fps_smooth = fps_smooth * 0.9 + fps * 0.1

        pred = outputs[0][0]

        boxes, scores, class_ids = [], [], []

        for det in pred:
            conf = det[4]
            if conf < CONF_THRESH:
                continue

            class_prob = det[5:]
            class_id = int(np.argmax(class_prob))
            score = float(conf * class_prob[class_id])
            if score < CONF_THRESH:
                continue

            cx, cy, w, h = det[:4]

            x1 = int((cx - w/2 - pad_x) / scale)
            y1 = int((cy - h/2 - pad_y) / scale)
            x2 = int((cx + w/2 - pad_x) / scale)
            y2 = int((cy + h/2 - pad_y) / scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        idxs = nms(boxes, scores)

        draw = display_img.copy()

        for i in idxs:
            x, y, w, h = boxes[i]
            # print(x,y)
            # print(w,h)
            cx = (x*2 + w) // 2
            cy = (y*2 + h) // 2
            # print(cx)
            # print(cy)
            label = classes[class_ids[i]]
            score = scores[i]
            cam_bag_mm = camera.pixel_to_world(u=cx, v=cy)
            cam_bag_mm_x = cam_bag_mm[0]
            cam_bag_mm_y = cam_bag_mm[1]
            print(cam_bag_mm)
            cv2.rectangle(draw, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(draw, f"{label} {score:.2f}",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)
            cv2.circle(draw, (cx,cy), 5, (0,0,255), -1)
            cv2.circle(draw, (int(cx_cam), int(cy_cam)), 5, (255,0,0), -1)
            cv2.line(draw, (int(cx_cam), int(cy_cam)), (cx,cy), (0,255,255), 2)
            cv2.putText(draw, f"{cam_bag_mm_x:.2f}, {cam_bag_mm_y:.2f}mm",
                    (cx+8,cy+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.putText(draw, f"Infer: {infer_time:.1f} ms | FPS: {fps_smooth:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        det_frame = draw

# ========================
#  START THREADS
# ========================

threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=infer_thread, daemon=True).start()

print("\n===== START REALTIME DETECTION =====")

display_count = 0

while True:
    if det_frame is not None:
        display_count += 1

        if display_count % DISPLAY_SKIP == 0:
            cv2.imshow("YOLO ONNX Threaded", det_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()