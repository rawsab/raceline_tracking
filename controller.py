import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# low level controller -> converts desired steering/velocity to control inputs
def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    current_steering = state[2]
    current_velocity = state[3]
    desired_steering, desired_velocity = desired
    
    # proportional control for steering rate
    steering_gain = 4.5
    steering_error = desired_steering - current_steering
    steering_velocity = steering_gain * steering_error
    
    # clamp steering rate
    min_steer_rate = parameters[7]
    max_steer_rate = parameters[9]
    steering_velocity = np.clip(steering_velocity, min_steer_rate, max_steer_rate)
    
    # proportional control for acceleration
    velocity_gain = 1.5
    velocity_error = desired_velocity - current_velocity
    acceleration = velocity_gain * velocity_error
    
    # clamp acceleration
    min_accel = parameters[8]
    max_accel = parameters[10]
    acceleration = np.clip(acceleration, min_accel, max_accel)
    
    # return control inputs
    return np.array([steering_velocity, acceleration])

# high level controller -> determines desired steering/velocity based on track curvature and position
def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    pos_x, pos_y, current_steer, current_vel, heading_angle = state
    L = parameters[0]  # car wheelbase length
    
    track_points = racetrack.centerline
    num_points = track_points.shape[0]
    
    # compute distances from current position to all track points
    position_vec = np.array([pos_x, pos_y])
    point_distances = np.sqrt(np.sum((track_points - position_vec)**2, axis=1))
    nearest_point_index = np.argmin(point_distances)
    
    # lookahead distance for path following
    lookahead_index = nearest_point_index

    lookahead = 16.0
    accumulated_distance = 0.0
    
    # traverse track points until we've covered the lookahead distance
    while accumulated_distance < lookahead:
        current_segment_start = track_points[lookahead_index % num_points]
        current_segment_end = track_points[(lookahead_index + 1) % num_points]
        segment_length = np.sqrt(np.sum((current_segment_end - current_segment_start)**2))
        accumulated_distance += segment_length
        lookahead_index += 1
    
    goal_point = track_points[lookahead_index % num_points]
    
    # calculate direction vector to goal point
    x_diff = goal_point[0] - pos_x
    y_diff = goal_point[1] - pos_y
    
    # desired heading angle towards goal
    target_heading = np.arctan2(y_diff, x_diff)
    
    # angle error between current heading and desired heading
    heading_error = target_heading - heading_angle
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error)) # normalize angle to [-pi, pi] range
    
    # compute steering command using "pure pursuit" geometry
    numerator = 2.0 * L * np.sin(heading_error)
    steering_command = np.arctan2(numerator, lookahead)
    
    # apply steering limits
    max_steer = parameters[4]
    steering_command = np.clip(steering_command, -max_steer, max_steer)
    
    # look ahead at multiple points to anticipate upcoming bends
    # estimate track curvature ahead for speed adaptation
    lookahead_samples = 8
    max_upcoming_curvature = 0.0
    
    for i in range(lookahead_samples):
        sample_idx = (nearest_point_index + i * 2) % num_points
        prev_idx = (sample_idx - 2) % num_points
        curr_idx = sample_idx
        next_idx = (sample_idx + 2) % num_points
        
        prev_pt = track_points[prev_idx]
        curr_pt = track_points[curr_idx]
        next_pt = track_points[next_idx]
        
        # compute vectors along the track
        vec_a = curr_pt - prev_pt
        vec_b = next_pt - curr_pt
        
        # calculate curvature using cross product method
        vec_a_mag = np.sqrt(np.sum(vec_a**2))
        vec_b_mag = np.sqrt(np.sum(vec_b**2))
        denominator = vec_a_mag * vec_b_mag + 1e-9
        sample_curvature = abs(np.cross(vec_a, vec_b) / denominator)
        
        # track maximum curvature in the lookahead region
        max_upcoming_curvature = max(max_upcoming_curvature, sample_curvature)
    
    # velocity planning -> aggressive speed reduction for extreme bends
    # use non-linear relationship (curvature squared) so extreme bends cause much more slowdown
    max_velocity = 54.0
    curvature_factor = 34.0

    # square the curvature to make extreme bends cause exponentially more speed reduction
    curvature_penalty = curvature_factor * (max_upcoming_curvature ** 3.2)
    target_velocity = max_velocity / (1.0 + curvature_penalty)
    target_velocity = np.clip(target_velocity, 10.0, max_velocity)
    
    # return desired steering/velocity
    return np.array([steering_command, target_velocity])