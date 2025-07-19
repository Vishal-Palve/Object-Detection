
import os
import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub


open("detections_log.txt", "w").close()

COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball",
    38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}


print("Downloading / loading SSD-MobileNet-v2 … (~25 MB)")
model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
model = hub.load(model_url)
print("Model loaded.")


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("Could not open webcam. Try another index or check permissions.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


MIN_CONF_THRESH = 0.4

def draw_boxes(frame, boxes, class_ids, scores, threshold=MIN_CONF_THRESH):
    height, width, _ = frame.shape
    detections = []

    for i in range(len(scores)):
        score = scores[i]
        if score < threshold:
            continue

        box = boxes[i]
        y1, x1, y2, x2 = box
        left, top, right, bottom = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        label = COCO_LABELS.get(class_ids[i], "Unknown")
        detections.append(f"{label} ({score:.2f})")

        color = (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{label}:{int(score * 100)}%",
                    (left, max(top - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    if detections:
        print("Detected:", ", ".join(detections))


    with open("detections_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {', '.join(detections)}\n")


times = []
print("\nPress 'q' or ESC to quit, 's' to save snapshot.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected.")
        break


    rgb = cv2.cvtColor(cv2.resize(frame, (512, 512)), cv2.COLOR_BGR2RGB)
    input_tensor = tf.expand_dims(tf.convert_to_tensor(rgb, dtype=tf.uint8), 0)


    outputs = model(input_tensor)
    boxes = outputs["detection_boxes"][0].numpy()
    class_ids = outputs["detection_classes"][0].numpy().astype(np.int32)
    scores = outputs["detection_scores"][0].numpy()


    draw_boxes(frame, boxes, class_ids, scores)


    times.append(time.time())
    if len(times) > 30:
        times = times[-30:]
    fps = len(times) / (times[-1] - times[0] + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Real-Time Object Detection (SSD-MobileNet-v2)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    elif key == ord('s'):
        filename = f"detection_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")


cap.release()
cv2.destroyAllWindows()
