import cv2
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator

# Initialize camera movement estimator
camera_movement_estimator = CameraMovementEstimator()

# Video file path
video_path = "input_videos/A1606b0e6_0 (29).mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Estimate camera movement
    H = camera_movement_estimator.estimate_movement(frame)
    print("Homography Matrix:\n", H)

cap.release()
# cv2.destroyAllWindows()
