import os
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = f'{det["animal"]} {det["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return image


def predict_image(image_path):
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "animal": label,
                "confidence": round(confidence, 3),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    image = cv2.imread(image_path)
    annotated = draw_boxes(image, detections)

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/annotated_image.jpg"
    cv2.imwrite(output_path, annotated)

    return detections, output_path


def predict_video(video_path, frame_step=10):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/annotated_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_detections = []
    last_detections = []
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_step == 0:
            frame_detections = []

            results = model.predict(
                source=frame,
                conf=0.25,
                save=False,
                verbose=False
            )

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    det = {
                        "animal": label,
                        "confidence": round(confidence, 3),
                        "frame": frame_index,
                        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    }

                    frame_detections.append(det)
                    all_detections.append(det)

            last_detections = frame_detections

        frame = draw_boxes(frame, last_detections)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    best_detections = {}

    for det in all_detections:
        animal = det["animal"]

        if animal not in best_detections:
            best_detections[animal] = det
        elif det["confidence"] > best_detections[animal]["confidence"]:
            best_detections[animal] = det

    return list(best_detections.values()), output_path


def run_vision_pipeline(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        return predict_image(file_path)

    elif ext in [".mp4", ".avi", ".mov"]:
        return predict_video(file_path, frame_step=10)

    else:
        return [], None