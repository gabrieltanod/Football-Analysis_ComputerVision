import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        orb = cv2.ORB_create(500)  # Initialize ORB with 500 keypoints
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Detect ORB keypoints and descriptors in the first frame
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_keypoints, old_descriptors = orb.detectAndCompute(old_gray, None)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Detect ORB keypoints and descriptors in the current frame
            new_keypoints, new_descriptors = orb.detectAndCompute(frame_gray, None)

            if old_descriptors is not None and new_descriptors is not None:
                # Match descriptors between the old and new frame
                matches = bf.match(old_descriptors, new_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance

                # Compute camera movement from matched keypoints
                max_distance = 0
                camera_movement_x, camera_movement_y = 0, 0

                for m in matches[:50]:  # Use the top 50 matches
                    old_point = old_keypoints[m.queryIdx].pt
                    new_point = new_keypoints[m.trainIdx].pt

                    distance = measure_distance(old_point, new_point)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)

                if max_distance > self.minimum_distance:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]

            # Update for the next frame
            old_keypoints, old_descriptors = new_keypoints, new_descriptors
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
