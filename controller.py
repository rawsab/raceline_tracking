import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# Control gains - tuned for maximum stability and staying on track
K_LAT = 0.5          # Lateral error gain (very conservative)
K_HEAD = 1.0         # Heading error gain (very conservative)
K_DELTA = 4.0        # Steering rate gain
K_V = 1.5            # Speed control gain
TARGET_SPEED = 8.0   # Desired speed (m/s) - VERY LOW for stability
MAX_LATERAL_ERROR_FOR_FULL_SPEED = 1.0  # Reduce speed aggressively if lateral error exceeds this
LOOKAHEAD_BASE = 10.0  # Base lookahead distance (longer for stability)
LOOKAHEAD_KV = 0.15  # Speed-dependent lookahead coefficient
STEERING_SMOOTHING = 0.6  # Smoothing factor for steering (higher = smoother, less wobbling)

def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-π, π] range.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def get_closest_index(state: ArrayLike, centerline: ArrayLike) -> int:
    """
    Find the index of the closest point on the centerline to the car's position.
    
    Args:
        state: Car state [x, y, δ, v, φ]
        centerline: Nx2 array of centerline points
        
    Returns:
        Index of closest centerline point
    """
    pos = state[0:2]
    diffs = centerline - pos
    distances = np.linalg.norm(diffs, axis=1)
    return np.argmin(distances)

def compute_track_curvature(centerline: ArrayLike, idx: int, lookahead_points: int = 15) -> float:
    """
    Compute the curvature of the track ahead of a given point.
    Uses multiple points ahead to estimate curvature.
    
    Args:
        centerline: Nx2 array of centerline points
        idx: Current index on centerline
        lookahead_points: Number of points ahead to use for curvature estimation
        
    Returns:
        Curvature estimate (1/radius, higher = sharper turn)
    """
    # Get points ahead
    points_ahead = []
    for i in range(lookahead_points):
        check_idx = (idx + i) % len(centerline)
        points_ahead.append(centerline[check_idx])
    
    if len(points_ahead) < 3:
        return 0.0
    
    # Compute curvature using three consecutive points
    # Curvature = 2 * area of triangle / (side lengths product)
    max_curvature = 0.0
    for i in range(len(points_ahead) - 2):
        p1 = points_ahead[i]
        p2 = points_ahead[i + 1]
        p3 = points_ahead[i + 2]
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Lengths
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 1e-6 and len2 > 1e-6:
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Curvature estimate (turn angle per unit distance)
            avg_len = (len1 + len2) / 2.0
            if avg_len > 1e-6:
                curvature = angle / avg_len
                max_curvature = max(max_curvature, curvature)
    
    return max_curvature

def get_lookahead_index(state: ArrayLike, centerline: ArrayLike, lookahead_dist: float) -> int:
    """
    Find the lookahead point on the centerline.
    ALWAYS finds a point ahead of the car (in direction of travel).
    
    Args:
        state: Car state [x, y, δ, v, φ]
        centerline: Nx2 array of centerline points
        lookahead_dist: Lookahead distance in meters
        
    Returns:
        Index of lookahead point (guaranteed to be ahead of car)
    """
    pos = state[0:2]
    heading = state[4]
    car_dir = np.array([np.cos(heading), np.sin(heading)])
    
    # Find the point on centerline that is most ahead of the car
    # (not necessarily closest, but ahead in direction of travel)
    best_idx = 0
    best_dot = -1.0
    
    # Search all points to find the one most aligned with car heading
    for i in range(len(centerline)):
        to_point = centerline[i] - pos
        dist_to_point = np.linalg.norm(to_point)
        
        if dist_to_point > 1e-3 and dist_to_point < 100.0:  # Reasonable distance
            to_point_norm = to_point / dist_to_point
            dot = np.dot(car_dir, to_point_norm)
            
            # Only consider points ahead (positive dot product)
            if dot > best_dot and dot > 0.0:  # Must be ahead
                best_dot = dot
                best_idx = i
    
    # If no point is ahead, use closest point as fallback
    if best_dot < 0:
        best_idx = get_closest_index(state, centerline)
    
    # Determine forward direction along track from best point
    next_idx = (best_idx + 1) % len(centerline)
    prev_idx = (best_idx - 1) % len(centerline)
    
    to_next = centerline[next_idx] - centerline[best_idx]
    to_prev = centerline[best_idx] - centerline[prev_idx]
    
    # Normalize
    if np.linalg.norm(to_next) > 1e-6:
        to_next = to_next / np.linalg.norm(to_next)
    if np.linalg.norm(to_prev) > 1e-6:
        to_prev = to_prev / np.linalg.norm(to_prev)
    
    # Check which direction is more aligned with car heading
    dot_next = np.dot(car_dir, to_next)
    dot_prev = np.dot(car_dir, to_prev)
    
    # Start from best point and go in forward direction
    if dot_next > dot_prev:
        start_idx = best_idx
        direction = 1
    else:
        start_idx = best_idx
        direction = -1
    
    # Search forward along centerline for lookahead point
    accumulated_dist = 0.0
    current_idx = start_idx
    
    for _ in range(len(centerline)):
        next_idx = (current_idx + direction) % len(centerline)
        segment_dist = np.linalg.norm(centerline[next_idx] - centerline[current_idx])
        accumulated_dist += segment_dist
        
        if accumulated_dist >= lookahead_dist:
            # Verify this point is actually ahead of the car
            to_lookahead = centerline[next_idx] - pos
            dist_to_lookahead = np.linalg.norm(to_lookahead)
            if dist_to_lookahead > 1e-3:
                to_lookahead_norm = to_lookahead / dist_to_lookahead
                dot_lookahead = np.dot(car_dir, to_lookahead_norm)
                if dot_lookahead > 0:  # Point is ahead
                    return next_idx
            
        current_idx = next_idx
    
    # Fallback: return point in forward direction
    return (start_idx + direction) % len(centerline)

def compute_track_errors(state: ArrayLike, centerline: ArrayLike, idx: int) -> tuple[float, float]:
    """
    Compute lateral and heading errors relative to the track.
    
    Args:
        state: Car state [x, y, δ, v, φ]
        centerline: Nx2 array of centerline points
        idx: Index of closest centerline point
        
    Returns:
        (lateral_error, heading_error)
    """
    pos = state[0:2]
    heading = state[4]
    
    # Get closest point and next point for tangent
    closest_point = centerline[idx]
    
    # Compute tangent vector (direction of track)
    # Use next point, wrapping around if at end
    next_idx = (idx + 1) % len(centerline)
    tangent = centerline[next_idx] - closest_point
    
    # Normalize tangent
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6:
        # Fallback: use previous point if tangent is too small
        prev_idx = (idx - 1) % len(centerline)
        tangent = closest_point - centerline[prev_idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            # Still too small, use unit vector in current heading
            tangent = np.array([np.cos(heading), np.sin(heading)])
            tangent_norm = 1.0
    
    tangent = tangent / tangent_norm
    
    # Compute normal vector (perpendicular to tangent, pointing left)
    normal = np.array([-tangent[1], tangent[0]])
    
    # Compute lateral error (signed distance from centerline)
    to_car = pos - closest_point
    lateral_error = np.dot(to_car, normal)
    
    # Compute heading error (difference between track heading and car heading)
    track_heading = np.arctan2(tangent[1], tangent[0])
    heading_error = wrap_angle(track_heading - heading)
    
    return lateral_error, heading_error

# Global variables for smoothing and recovery detection
_prev_delta_des = 0.0
_prev_lateral_error = 0.0
_recovery_mode = False  # Track if we're in recovery (reducing lateral error)

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    """
    High-level controller that computes desired steering angle and velocity.
    
    Args:
        state: Car state [x, y, δ, v, φ]
        parameters: Car parameters array
        racetrack: RaceTrack object with centerline
        
    Returns:
        [δ_des, v_des] - desired steering angle and velocity
    """
    global _prev_delta_des
    
    assert state.shape == (5,)
    assert parameters.shape == (11,)
    
    # Get raceline if available, otherwise use centerline
    # Use raceline for tracking if it's available (instructor recommendation)
    # NOTE: Raceline may be closer to track boundaries, so we blend with centerline for safety
    using_raceline = False
    if hasattr(racetrack, 'raceline') and racetrack.raceline is not None:
        # Blend raceline with centerline to stay away from boundaries
        # Use 70% raceline + 30% centerline to create a safer path
        raceline = racetrack.raceline
        centerline_ref = racetrack.centerline
        
        # Interpolate between raceline and centerline
        # Find corresponding points (assume similar indexing)
        blended_centerline = np.zeros_like(raceline)
        for i in range(len(raceline)):
            # Find closest centerline point to this raceline point
            dists = np.linalg.norm(centerline_ref - raceline[i], axis=1)
            closest_center_idx = np.argmin(dists)
            # Blend: 70% raceline, 30% centerline (moves away from boundaries)
            blended_centerline[i] = 0.7 * raceline[i] + 0.3 * centerline_ref[closest_center_idx]
        
        centerline = blended_centerline
        using_raceline = True
    else:
        centerline = racetrack.centerline
    
    # When using raceline, be more aggressive about staying on it (raceline is closer to boundaries)
    # Reduce thresholds for lateral correction to prevent drift
    if using_raceline:
        # More aggressive correction thresholds when using raceline
        LAT_ERROR_THRESHOLD_SMALL = 0.5  # Reduced from 1.0
        LAT_ERROR_THRESHOLD_MEDIUM = 1.0  # Reduced from 2.0
        LAT_ERROR_THRESHOLD_LARGE = 2.0  # Reduced from 3.0
        LAT_ERROR_THRESHOLD_RECOVERY = 1.5  # Reduced from 3.0
    else:
        # Normal thresholds for centerline
        LAT_ERROR_THRESHOLD_SMALL = 1.0
        LAT_ERROR_THRESHOLD_MEDIUM = 2.0
        LAT_ERROR_THRESHOLD_LARGE = 3.0
        LAT_ERROR_THRESHOLD_RECOVERY = 3.0
    
    # Use Pure Pursuit for more stable control
    v = state[3]
    wheelbase = parameters[0]
    
    # Get car position and heading
    pos = state[0:2]
    heading = state[4]
    
    # Find closest point and compute errors
    closest_idx = get_closest_index(state, centerline)
    closest_point = centerline[closest_idx]
    lateral_error, heading_error_track = compute_track_errors(state, centerline, closest_idx)
    abs_lateral_error = abs(lateral_error)
    
    # Detect upcoming turn curvature (look ahead on track)
    # Use more points to detect turns earlier
    track_curvature = compute_track_curvature(centerline, closest_idx, lookahead_points=10)
    # Very low threshold to detect even moderate turns as "sharp" for better recovery
    # This ensures we handle turns aggressively even if they're not extremely sharp
    is_sharp_turn = track_curvature > 0.03  # Very low threshold to catch most turns
    
    # Detect recovery mode: if lateral error is decreasing, we're recovering
    global _prev_lateral_error, _recovery_mode
    is_recovering = False
    if abs(_prev_lateral_error) > abs_lateral_error and abs_lateral_error > 1.0:
        # Lateral error is decreasing and we're still off track - we're recovering
        is_recovering = True
        _recovery_mode = True
    elif abs_lateral_error < (1.0 if using_raceline else 1.5):
        # Close to track, exit recovery mode quickly to prevent overshoot
        # When using raceline, exit recovery mode sooner (raceline is closer to boundaries)
        _recovery_mode = False
        is_recovering = False
    else:
        # Check if we just started recovering (error changed sign)
        if _prev_lateral_error * lateral_error < 0 and abs_lateral_error > 1.0:
            # Error changed sign - we're crossing the track, be very careful
            is_recovering = True
            _recovery_mode = True
    
    _prev_lateral_error = lateral_error
    
    # Adaptive lookahead: adjust based on track conditions and recovery needs
    # CRITICAL: When off track, use VERY long lookahead to consider track shape ahead
    # This allows recovery to follow the track's direction rather than steering perpendicular
    if abs_lateral_error > LAT_ERROR_THRESHOLD_RECOVERY:
        # Off track: use very long lookahead to consider track shape ahead
        # Scale with error and curvature to ensure we look far enough ahead
        error_factor = min(1.0, abs_lateral_error / 8.0)
        curvature_factor = min(1.0, track_curvature / 0.15) if is_sharp_turn else 0.0
        # Use very long lookahead: 30-50m to see track shape ahead (increased)
        lookahead_multiplier = 3.0 + 2.0 * error_factor + 1.5 * curvature_factor  # 3.0 to 6.5
        lookahead_dist = max(30.0, LOOKAHEAD_BASE * lookahead_multiplier)
    elif abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM and is_sharp_turn:
        # Moderately off track + sharp turn: use long lookahead
        curvature_factor = min(1.0, track_curvature / 0.15)
        lookahead_multiplier = 2.0 + 0.8 * curvature_factor  # 2.0 to 2.8
        lookahead_dist = max(20.0, LOOKAHEAD_BASE * lookahead_multiplier)
    elif is_sharp_turn:
        # Sharp turn ahead (on track): use SHORTER lookahead to react faster
        curvature_factor = min(1.0, track_curvature / 0.25)
        lookahead_reduction = 0.3 + 0.2 * (1.0 - curvature_factor)  # 0.3 to 0.5 reduction
        lookahead_dist = max(5.0, LOOKAHEAD_BASE * (1.0 - lookahead_reduction))
    else:
        # Normal operation: adaptive lookahead based on speed
        lookahead_dist = LOOKAHEAD_BASE + LOOKAHEAD_KV * v
    
    # Find lookahead point (always forward-looking)
    lookahead_idx = get_lookahead_index(state, centerline, lookahead_dist)
    lookahead_point = centerline[lookahead_idx]
    
    # CRITICAL: When recovering, compute track direction at recovery point (not closest point)
    # This allows the car to follow the track's direction rather than steering perpendicular
    recovery_tangent = None
    if abs_lateral_error > LAT_ERROR_THRESHOLD_RECOVERY:
        # Off track: use track direction at lookahead point for recovery
        # Get track tangent at lookahead point (where we want to be)
        recovery_next_idx = (lookahead_idx + 1) % len(centerline)
        recovery_prev_idx = (lookahead_idx - 1) % len(centerline)
        
        # Compute track direction at recovery point
        to_next = centerline[recovery_next_idx] - lookahead_point
        to_prev = lookahead_point - centerline[recovery_prev_idx]
        
        # Use the direction that's more aligned with car's forward direction
        to_next_norm = np.linalg.norm(to_next)
        to_prev_norm = np.linalg.norm(to_prev)
        
        if to_next_norm > 1e-6 and to_prev_norm > 1e-6:
            to_next_unit = to_next / to_next_norm
            to_prev_unit = to_prev / to_prev_norm
            
            # Check which direction is more aligned with car heading
            car_dir = np.array([np.cos(heading), np.sin(heading)])
            dot_next = np.dot(car_dir, to_next_unit)
            dot_prev = np.dot(car_dir, to_prev_unit)
            
            # Also check if direction points toward track (not away)
            to_track = lookahead_point - pos
            to_track_norm = np.linalg.norm(to_track)
            if to_track_norm > 1e-6:
                to_track_unit = to_track / to_track_norm
                dot_next_track = np.dot(to_track_unit, to_next_unit)
                dot_prev_track = np.dot(to_track_unit, to_prev_unit)
                
                # Prefer direction that's both aligned with car AND points toward track
                if dot_next > 0.3 and dot_next_track > 0.3:
                    recovery_tangent = to_next_unit
                elif dot_prev > 0.3 and dot_prev_track > 0.3:
                    recovery_tangent = to_prev_unit
                elif dot_next > dot_prev:
                    recovery_tangent = to_next_unit
                else:
                    recovery_tangent = to_prev_unit
            else:
                if dot_next > dot_prev:
                    recovery_tangent = to_next_unit
                else:
                    recovery_tangent = to_prev_unit
    
    # Vector from car to lookahead point
    to_lookahead = lookahead_point - pos
    dist_to_lookahead = np.linalg.norm(to_lookahead)
    
    if dist_to_lookahead < 1e-3:
        # Fallback: use track heading error
        delta_des = 0.5 * K_HEAD * heading_error_track
    else:
        # CRITICAL RECOVERY MODE: When off track, use Pure Pursuit with very long lookahead
        # Pure Pursuit naturally follows track shape, preventing perpendicular steering
        # The long lookahead ensures we consider track shape ahead, not just one point
        if abs_lateral_error > LAT_ERROR_THRESHOLD_RECOVERY:
            # Recovery mode: use Pure Pursuit with aggressive steering
            # Pure Pursuit naturally follows the track's direction, preventing overshoot
            angle_to_lookahead = np.arctan2(to_lookahead[1], to_lookahead[0])
            heading_error_lookahead = wrap_angle(angle_to_lookahead - heading)
            sin_alpha = np.sin(heading_error_lookahead)
            curvature = 2.0 * sin_alpha / dist_to_lookahead
            
            # Use aggressive steering gain for recovery
            if abs_lateral_error > 8.0:
                steering_gain = 1.0  # Full gain
            elif abs_lateral_error > 5.0:
                steering_gain = 0.95
            else:
                steering_gain = 0.9
            
            delta_des = steering_gain * np.arctan(wheelbase * curvature)
        else:
            # Normal Pure Pursuit: compute steering based on curvature to lookahead point
            angle_to_lookahead = np.arctan2(to_lookahead[1], to_lookahead[0])
            heading_error = wrap_angle(angle_to_lookahead - heading)
            sin_alpha = np.sin(heading_error)
            curvature = 2.0 * sin_alpha / dist_to_lookahead
            
            # Adjust steering gain based on turn sharpness and recovery state
            if abs_lateral_error > 6.0:
                # Very far off track (near violation): use maximum steering gain
                steering_gain = 1.0  # Full gain for recovery
            elif abs_lateral_error > 5.0:
                # Far off track: use high steering gain
                steering_gain = 0.95
            elif is_sharp_turn:
                # Sharp turn: increase steering gain to turn faster
                curvature_factor = min(1.0, track_curvature / 0.2)
                
                if abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM:
                    # Off track + sharp turn: very aggressive steering
                    steering_gain = 0.85 + 0.15 * curvature_factor  # Range: 0.85 to 1.0
                else:
                    # On track + sharp turn: normal aggressive steering
                    steering_gain = 0.6 + 0.35 * curvature_factor  # Range: 0.6 to 0.95
            else:
                # Normal turn: conservative gain
                steering_gain = 0.5
            
            delta_des = steering_gain * np.arctan(wheelbase * curvature)
        
        # Add lateral correction - CRITICAL: More aggressive when using raceline
        # Pure Pursuit with long lookahead handles recovery by following track shape
        # Adding perpendicular correction causes overshoot and violations on opposite side
        if abs_lateral_error > LAT_ERROR_THRESHOLD_RECOVERY and not (is_recovering or _recovery_mode):
            # Off track but not yet recovering: use small correction to initiate recovery
            # Once recovery starts, Pure Pursuit takes over
            lateral_correction = -0.1 * K_LAT * lateral_error
            lateral_correction = np.clip(lateral_correction, -0.15, 0.15)
            delta_des = delta_des + lateral_correction
        elif abs_lateral_error > LAT_ERROR_THRESHOLD_RECOVERY:
            # Recovery mode: NO perpendicular lateral correction
            # Pure Pursuit with long lookahead naturally follows track direction
            # This prevents the car from cutting across and exiting on the other side
            pass  # No lateral correction - Pure Pursuit handles it
        elif is_recovering or _recovery_mode:
            # In recovery (moderate error): minimal correction to prevent overshoot
            if abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM:
                # Small correction only
                lateral_correction = -0.05 * K_LAT * lateral_error
                lateral_correction = np.clip(lateral_correction, -0.08, 0.08)
                delta_des = delta_des + lateral_correction
        elif abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM:
            # Normal operation: moderate correction when off track
            # When using raceline, be more aggressive to prevent drift
            correction_factor = 0.3 if using_raceline else 0.25
            lateral_correction = -correction_factor * K_LAT * lateral_error
            lateral_correction = np.clip(lateral_correction, -0.35 if using_raceline else -0.3, 0.35 if using_raceline else 0.3)
            delta_des = delta_des + lateral_correction
        elif abs_lateral_error > LAT_ERROR_THRESHOLD_SMALL:
            # Small error: use moderate correction to prevent drift
            # When using raceline, be more aggressive
            correction_factor = 0.25 if using_raceline else 0.2
            lateral_correction = -correction_factor * K_LAT * lateral_error
            lateral_correction = np.clip(lateral_correction, -0.3 if using_raceline else -0.25, 0.3 if using_raceline else 0.25)
            delta_des = delta_des + lateral_correction
        else:
            # Very close to track: small fine-tuning
            # When using raceline, be slightly more aggressive
            correction_factor = 0.12 if using_raceline else 0.1
            delta_des = delta_des - correction_factor * K_LAT * lateral_error
    
    # Apply steering smoothing to prevent wobbling
    # BUT: Reduce smoothing when off track before sharp turn for faster response
    global _prev_delta_des
    if abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM and is_sharp_turn:
        # Off track + sharp turn: minimal smoothing for fastest response
        smoothing_factor = STEERING_SMOOTHING * 0.3  # Only 30% of normal smoothing
        delta_des = smoothing_factor * _prev_delta_des + (1.0 - smoothing_factor) * delta_des
    else:
        # Normal smoothing
        delta_des = STEERING_SMOOTHING * _prev_delta_des + (1.0 - STEERING_SMOOTHING) * delta_des
    _prev_delta_des = delta_des
    
    # Prevent excessive steering to avoid spinning
    # BUT: Allow more steering when far off track (critical recovery)
    current_delta = state[2]
    abs_current_delta = abs(current_delta)
    if abs_lateral_error > 8.0:
        # Very far off track: allow maximum steering (up to 0.9)
        if abs_current_delta > 0.9:
            saturation_factor = 1.0 - (abs_current_delta - 0.9) / 0.05
            saturation_factor = max(0.7, saturation_factor)
            delta_des = delta_des * saturation_factor
    elif abs_lateral_error > 5.0:
        # Far off track: allow high steering (up to 0.85)
        if abs_current_delta > 0.85:
            saturation_factor = 1.0 - (abs_current_delta - 0.85) / 0.1
            saturation_factor = max(0.6, saturation_factor)
            delta_des = delta_des * saturation_factor
    elif abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM and is_sharp_turn:
        # Off track + sharp turn: allow high steering (up to 0.85)
        if abs_current_delta > 0.85:
            saturation_factor = 1.0 - (abs_current_delta - 0.85) / 0.05
            saturation_factor = max(0.6, saturation_factor)
            delta_des = delta_des * saturation_factor
    elif abs_current_delta > 0.6:  # Normal threshold
        # Reduce new steering command to prevent over-steering
        saturation_factor = 1.0 - (abs_current_delta - 0.6) / 0.3
        saturation_factor = max(0.4, saturation_factor)
        delta_des = delta_des * saturation_factor
    
    # Clip δ_des to steering angle limits
    # parameters[1] = -max_steering_angle, parameters[4] = max_steering_angle
    delta_des = np.clip(delta_des, parameters[1], parameters[4])
    
    # Set desired velocity (reduce speed for sharp turns and when off track)
    abs_lateral_error = abs(lateral_error)
    
    # CRITICAL: Reduce speed proactively to prevent drift and violations
    # Reduce speed when lateral error is increasing (even if small)
    # Note: _prev_lateral_error is already declared as global earlier in the function
    error_rate = abs_lateral_error - abs(_prev_lateral_error) if '_prev_lateral_error' in globals() else 0.0
    
    if is_sharp_turn:
        # Sharp turn detected: reduce speed proactively
        # More curvature = more speed reduction
        curvature_factor = min(1.0, track_curvature / 0.25)
        turn_speed_reduction = 0.6 * curvature_factor  # Up to 60% reduction
        
        # If also off track, reduce even more
        if abs_lateral_error > LAT_ERROR_THRESHOLD_MEDIUM:
            error_reduction = min(0.3, abs_lateral_error / 5.0)  # Additional 30% from error
            total_reduction = min(0.95, turn_speed_reduction + error_reduction)  # Up to 95% total
            v_des = TARGET_SPEED * (1.0 - total_reduction)
            v_des = max(1.5, v_des)  # Can go very slow for critical recovery
        else:
            # On track but sharp turn: reduce speed for better turning
            v_des = TARGET_SPEED * (1.0 - turn_speed_reduction)
            v_des = max(3.5, v_des)  # Lower minimum for turns
    elif abs_lateral_error > MAX_LATERAL_ERROR_FOR_FULL_SPEED:
        # Off track (no sharp turn): reduce speed
        speed_reduction = min(0.8, abs_lateral_error / 4.0)  # Reduce by up to 80%
        v_des = TARGET_SPEED * (1.0 - speed_reduction)
        v_des = max(3.0, v_des)  # Don't go below 3 m/s
    elif abs_lateral_error > 1.0 and error_rate > 0.1:
        # Lateral error is increasing: reduce speed to prevent drift
        speed_reduction = min(0.4, abs_lateral_error / 3.0)  # Up to 40% reduction
        v_des = TARGET_SPEED * (1.0 - speed_reduction)
        v_des = max(5.0, v_des)  # Moderate reduction
    else:
        v_des = TARGET_SPEED
    
    # Clip v_des to velocity limits
    v_des = np.clip(v_des, parameters[2], parameters[5])
    
    return np.array([delta_des, v_des])

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    """
    Low-level controller that tracks desired steering angle and velocity.
    
    Args:
        state: Car state [x, y, δ, v, φ]
        desired: [δ_des, v_des] from high-level controller
        parameters: Car parameters array
        
    Returns:
        [v_δ, a] - steering rate and acceleration
    """
    assert state.shape == (5,)
    assert desired.shape == (2,)
    assert parameters.shape == (11,)
    
    # Extract current state
    delta = state[2]  # Current steering angle
    v = state[3]      # Current velocity
    
    # Extract desired values
    delta_des = desired[0]
    v_des = desired[1]
    
    # Compute steering error (wrapped)
    steering_error = wrap_angle(delta_des - delta)
    
    # Compute steering rate command
    v_delta = K_DELTA * steering_error
    
    # Compute acceleration command
    a = K_V * (v_des - v)
    
    # Note: Clipping is handled automatically in RaceCar.normalize_system
    return np.array([v_delta, a])