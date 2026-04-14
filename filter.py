import cv2
import numpy as np
from pykalman import KalmanFilter

# State transiiton matrix 
# Follows forward Euler x_{k+1} = x_k + dx
F = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

# First four states are observed at each step
H = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
])

# Estimated noise covariances
Q_est = np.load('parameters/Q.npy')
R_est = np.load('parameters/R.npy')

n = F.shape[0]  # Number of predicted states
m = H.shape[0]  # Number of observed states

# We don't use control for this
u = np.zeros(n)

DETECT_EVERY_N_FRAMES = 1
frame_count = 0

x_hat = None
P_hat = None
detected = False

kf = KalmanFilter(
    transition_matrices=F,
    observation_matrices=H,
    transition_covariance=Q_est,
    observation_covariance=R_est,
    n_dim_state=8,
    n_dim_obs=4,
)

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

            if not detected:
                # Initialize state from first detection
                x_hat = np.concatenate((z_t, np.zeros(m)))
                P_hat = np.diag([100, 100, 100, 100, 10, 10, 10, 10]).astype(float)
                detected = True

            x_hat, P_hat = kf.filter_update(x_hat, P_hat, observation=z_t)

            # Draw green bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Observed', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            if detected:
                x_hat, P_hat = kf.filter_update(x_hat, P_hat, observation=None)

    # Draw Kalman box every frame if initialized
    if detected and x_hat is not None:
        # After Kalman predict/update step:
        kf_x, kf_y, kf_w, kf_h = round(x_hat[0]), round(x_hat[1]), round(x_hat[2]), round(x_hat[3])

        cv2.rectangle(frame, (kf_x, kf_y), (kf_x+kf_w, kf_y+kf_h), (255, 0, 0), 2)
        cv2.putText(frame, 'Kalman', (kf_x, kf_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    cv2.imshow('Face Tracker', frame)
    # Press q to exit window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
