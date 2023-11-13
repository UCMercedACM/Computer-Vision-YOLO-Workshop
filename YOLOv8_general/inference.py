import cv2
import numpy
from ultralytics import YOLO

model = YOLO("weights/yolov8m.pt")


def run():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Flip on horizontal
            image = cv2.flip(frame, 1)
            results = model(image, save=False)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO-v8 Pose Estimation", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
