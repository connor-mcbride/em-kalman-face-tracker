import cv2
import numpy as np


DETECT_EVERY_N_FRAMES = 10
frame_count = 0

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for the detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # Run facial detector every N frames
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        # Detect faces, return list of [x, y, w, h] bounding boxes
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,    # how much image is scaled down each step
            minNeighbors=5,     # higher = fewer false positives
            minSize=(60, 60)    # ignore tiny detections
        )

        if len(detections) > 0:
            x, y, w, h = detections[0]
            z_t = np.array([x, y, w, h], dtype=float)

            # Draw green bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Measured', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # # After Kalman predict/update step:
            # kf_x, kf_y, kf_w, kf_h = int(x_hat[0]), int(x_hat[1]), int(x_hat[2]), int(x_hat[3])

            # # Kalman filtered box (blue)
            # cv2.rectangle(frame, (kf_x, kf_y), (kf_x+kf_w, kf_y+kf_h), (255, 0, 0), 2)
            # cv2.putText(frame, 'Kalman', (kf_x, kf_y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    cv2.imshow('Face Tracker', frame)

    # Press q to exit window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
