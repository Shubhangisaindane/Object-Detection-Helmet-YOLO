from ultralytics import YOLO
import os
import cv2

model = YOLO(os.path.join("best_model", "best.pt"))

def predict_image(image_path):
    results = model.predict(image_path, conf=0.1)

    predictions = []

    for result in results:
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            predictions.append({
                "class": int(cls),
                "confidence": float(conf)
            })

    return predictions


def run_webcam_detection():
    # Load trained model
    model = YOLO(os.path.join("best_model", "best.pt"))

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'Q' to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # Run inference
        results = model.predict(
            source=frame,
            conf=0.4,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                # Class information
                cls_id = int(box.cls[0])
                detected = result.names[cls_id]
                confidence = float(box.conf[0])

                # Bounding box coordinates
                xmin, ymin, xmax, ymax = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Colors
                if detected == "With Helmet":
                    color = (0, 255, 0)      # Green
                elif detected == "Without Helmet":
                    color = (0, 0, 255)      # Red
                else:
                    color = (255, 255, 255)

                # Draw box
                cv2.rectangle(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    color,
                    2
                )

                # Draw label
                label = f"{detected} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # Show output
        cv2.imshow("Helmet Detection", frame)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_detection()