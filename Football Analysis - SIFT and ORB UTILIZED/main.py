from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Read Video
    video_frames = read_video('input_videos/test (10) - Trim.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator with Hybrid Optical Flow + SIFT + ORB
    camera_movement_estimator = CameraMovementEstimator()
    camera_movements = []

    # Iterate through the frames to compute camera movements
    # for i, frame in enumerate(video_frames):
    #     if i == 0:
    #         # Set the first frame as the reference frame
    #         camera_movement_estimator.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         camera_movements.append(np.eye(3))  # Identity matrix for the first frame
    #         continue

    #     # Estimate camera movement for the current frame
    #     H = camera_movement_estimator.estimate_movement(frame)
    #     camera_movements.append(H)
    # UNDO INI NANTI
    # Iterate through the frames to compute camera movements
    for i, frame in enumerate(video_frames):
        if i == 0:
            # Set the first frame as the reference frame
            camera_movement_estimator.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_movements.append(np.eye(3))  # Identity matrix for the first frame
            print(f"Frame {i}: Reference frame set.")
            continue

        # Estimate camera movement using SIFT or ORB
        H, method_used = camera_movement_estimator.estimate_movement(frame)
        camera_movements.append(H)
        print(f"Frame {i}: Estimated camera movement using {method_used}.")

    # Add adjusted positions to tracks
    for i, track in enumerate(tracks['players']):
        for player_id, player_data in track.items():
            pos = np.array([player_data['position'][0], player_data['position'][1], 1])
            
            if i == 0:
                # For the first frame, no adjustment; use original position
                adjusted_pos = pos[:2]  # Use only x and y
            else:
                # Adjust position using the homography matrix
                adjusted_pos = np.dot(camera_movements[i], pos)
                adjusted_pos /= adjusted_pos[2]  # Normalize to handle homogeneous coordinates
                adjusted_pos = adjusted_pos[:2]  # Extract only x and y coordinates

            # Store the adjusted position under the correct key
            tracks['players'][i][player_id]['position_adjusted'] = adjusted_pos.tolist()

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = [0]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                # Handle the case when the list is empty, for example, appending a default value
                team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)

    # Draw Output 
    ## Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera Movement
    for i, H in enumerate(camera_movements):
        if i > 0:
            cv2.putText(output_video_frames[i], f"Homography Matrix: {H}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    print('Video saved')
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
