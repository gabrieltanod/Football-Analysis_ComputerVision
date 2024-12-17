# import cv2
# import numpy as np

# class CameraMovementEstimator:
#     def __init__(self):
#         # Parameters for Optical Flow
#         self.lk_params = dict(winSize=(15, 15), maxLevel=2,
#                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#         self.optical_flow_points = None
        
#         # SIFT/ORB Initialization
#         self.sift = cv2.SIFT_create()
#         self.orb = cv2.ORB_create(nfeatures=1000)  # ORB for faster feature extraction
#         self.matcher_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # SIFT Matcher
#         self.matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # ORB Matcher

#         # Previous frame for comparison
#         self.prev_frame = None

#     def estimate_movement(self, current_frame):
#         gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

#         if self.prev_frame is None:
#             # Initialize optical flow points
#             self.optical_flow_points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
#             self.prev_frame = gray_frame
#             return np.eye(3)  # Identity matrix (no movement on first frame)

#         # Optical Flow
#         if self.optical_flow_points is not None:
#             new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_frame, self.optical_flow_points, None, **self.lk_params)
#             good_old = self.optical_flow_points[status == 1]
#             good_new = new_points[status == 1]

#             # If sufficient points remain, estimate movement using optical flow
#             if len(good_old) > 10:
#                 H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
#                 self.optical_flow_points = good_new.reshape(-1, 1, 2)  # Update points for next iteration
#                 self.prev_frame = gray_frame
#                 return H

#         # SIFT Matching
#         keypoints1, descriptors1 = self.sift.detectAndCompute(self.prev_frame, None)
#         keypoints2, descriptors2 = self.sift.detectAndCompute(gray_frame, None)

#         if descriptors1 is not None and descriptors2 is not None:
#             matches = self.matcher_sift.knnMatch(descriptors1, descriptors2, k=2)
#             good_matches = []
#             for m, n in matches:
#                 if m.distance < 0.75 * n.distance:  # Ratio test
#                     good_matches.append(m)

#             if len(good_matches) > 10:
#                 src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#                 H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#                 self.optical_flow_points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)  # Reset optical flow points
#                 self.prev_frame = gray_frame
#                 return H

#         # ORB Matching (Fallback)
#         keypoints1_orb, descriptors1_orb = self.orb.detectAndCompute(self.prev_frame, None)
#         keypoints2_orb, descriptors2_orb = self.orb.detectAndCompute(gray_frame, None)

#         if descriptors1_orb is not None and descriptors2_orb is not None:
#             matches_orb = self.matcher_orb.match(descriptors1_orb, descriptors2_orb)
#             matches_orb = sorted(matches_orb, key=lambda x: x.distance)

#             if len(matches_orb) > 10:
#                 src_pts_orb = np.float32([keypoints1_orb[m.queryIdx].pt for m in matches_orb]).reshape(-1, 1, 2)
#                 dst_pts_orb = np.float32([keypoints2_orb[m.trainIdx].pt for m in matches_orb]).reshape(-1, 1, 2)

#                 H_orb, _ = cv2.findHomography(src_pts_orb, dst_pts_orb, cv2.RANSAC, 5.0)
#                 self.optical_flow_points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)  # Reset optical flow points
#                 self.prev_frame = gray_frame
#                 return H_orb

#         # If all methods fail, return identity matrix
#         self.prev_frame = gray_frame
#         return np.eye(3)

import cv2
import numpy as np

class CameraMovementEstimator:
    def __init__(self):
        # Initialize SIFT and ORB
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()

        # Initialize Brute-Force Matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # For SIFT/ORB

        # Previous frame for comparison
        self.prev_frame = None

    def estimate_movement(self, current_frame):
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            # Set the first frame
            self.prev_frame = gray_frame
            return np.eye(3), "None"  # Identity matrix (no movement on first frame)

        # Use SIFT to compute keypoints and descriptors
        keypoints1, descriptors1 = self.sift.detectAndCompute(self.prev_frame, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(gray_frame, None)

        if descriptors1 is not None and descriptors2 is not None:
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # Ratio test
                    good_matches.append(m)

            if len(good_matches) > 10:
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                self.prev_frame = gray_frame
                return H, "SIFT"

        # If SIFT fails, try ORB
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.prev_frame, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(gray_frame, None)

        if descriptors1 is not None and descriptors2 is not None:
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # Ratio test
                    good_matches.append(m)

            if len(good_matches) > 10:
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                self.prev_frame = gray_frame
                return H, "ORB"

        # If all methods fail, return identity matrix
        self.prev_frame = gray_frame
        return np.eye(3), "None"
