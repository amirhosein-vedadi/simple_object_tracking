import cv2
import numpy as np
from ultralytics import YOLO
from simple_tracker import Tracker
import time

# Initialize YOLO model and Tracker
model = YOLO("../data/weights/yolov8n.pt")
tracker = Tracker()

# Open the video file
video_path = "../data/videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
output_path = "../data/videos/output_traffic.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize previous time for dt calculation
prev_time = time.time()

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Make a copy of the original frame for drawing
        display_frame = frame.copy()

        # Run YOLO detection on the current frame
        results = model.predict(frame, verbose=False)

        # Calculate current time and dt
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                if score > 0.5:  # Only consider detections with confidence > 0.5
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([x1, y1, x2 - x1, y2 - y1])  # Format: [x_min, y_min, width, height]
                    
                    # Draw bounding box for YOLO detections (optional)
                    # cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        tracker.run(detections)
        
        # Update Kalman Filter's dt dynamically (if needed)
        tracker.update_dt(dt)

        # Get current tracks and draw them on the display frame
        current_tracks = tracker.get_tracks()
        
        for track_id, track in current_tracks.items():
            bbox = track.get_state_as_bbox()  # Get the bounding box from the track object
            x_min, y_min, width, height = bbox
            
            # Draw the bounding box and ID on the display frame
            if track.coast_cycles_ < 4:
                cv2.rectangle(display_frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)
                cv2.putText(display_frame, f'ID: {track_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

        out.write(display_frame)

        cv2.imshow("YOLO Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
