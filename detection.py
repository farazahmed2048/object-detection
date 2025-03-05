# object_detection_tracking.py
# Requirements: pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetectorTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.track_history = {}

    def process_stream(self, source=0, show=True, conf=0.5):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error opening video stream/file")
            return

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = self.model.track(frame, persist=True, conf=conf)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.float().cpu().tolist()

                for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
                    x, y, w, h = box
                    class_name = self.model.names[class_id]
                    
                    # Draw detection
                    color = self.get_color(track_id)
                    cv2.rectangle(frame, 
                                (int(x - w/2), int(y - h/2)),
                                (int(x + w/2), int(y + h/2)),
                                color, 2)
                    
                    # Create label
                    label = f"{track_id} {class_name} {confidence:.2f}"
                    cv2.putText(frame, label,
                                (int(x - w/2), int(y - h/2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Update track history
                    self.update_track_history(track_id, (float(x), float(y)))

            if show:
                cv2.imshow("Object Detection & Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def get_color(self, track_id):
        np.random.seed(track_id)
        return (np.random.randint(0, 255), (np.random.randint(0, 255)), (np.random.randint(0, 255))

    def update_track_history(self, track_id, point):
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(point)
        if len(self.track_history[track_id]) > 50:  # Keep last 50 points
            self.track_history[track_id].pop(0)

if __name__ == "__main__":
    detector = ObjectDetectorTracker()
    
    # For webcam: source=0
    # For video file: source='path/to/video.mp4'
    detector.process_stream(source=0, conf=0.5)