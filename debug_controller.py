#!/usr/bin/env python3
"""
Debug script to log car state and controller outputs at each time step.
Outputs to CSV file for analysis.
"""

import numpy as np
import sys
from racetrack import RaceTrack
from racecar import RaceCar
from controller import controller, lower_controller, get_closest_index, compute_track_errors, compute_track_curvature
import csv

def simulate_and_log(track_path, output_csv, raceline_path=None, max_time=60.0, dt=0.1):
    """
    Run simulation and log all states to CSV.
    
    Args:
        track_path: Path to track CSV file
        output_csv: Path to output CSV file
        raceline_path: Optional path to raceline CSV file
        max_time: Maximum simulation time (seconds)
        dt: Time step (should match RaceCar.time_step)
    """
    # Load track
    racetrack = RaceTrack(track_path, raceline_path=raceline_path)
    car = RaceCar(racetrack.initial_state)
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'time', 'x', 'y', 'delta', 'v', 'phi',
            'closest_idx', 'closest_x', 'closest_y',
            'lateral_error', 'heading_error',
            'delta_des', 'v_des',
            'v_delta', 'a',
            'track_violation', 'track_curvature', 'is_sharp_turn', 'lookahead_dist'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        time = 0.0
        step = 0
        
        # Spinning detection
        position_history = []  # Store last N positions
        SPIN_CHECK_WINDOW = 30  # Check last 3 seconds (30 steps at 0.1s)
        SPIN_RADIUS_THRESHOLD = 5.0  # If car moves in circle < 5m radius, it's spinning
        SPIN_TIME_THRESHOLD = 3.0  # If spinning for 3+ seconds, stop
        
        print(f"Starting simulation...")
        print(f"Track: {track_path}")
        print(f"Output: {output_csv}")
        print(f"Max time: {max_time}s, dt: {dt}s")
        print()
        
        spinning_start_time = None
        
        while time < max_time:
            # Get current state
            state = car.state.copy()
            pos = state[0:2]
            
            # Find closest centerline point
            closest_idx = get_closest_index(state, racetrack.centerline)
            closest_point = racetrack.centerline[closest_idx]
            
            # Compute errors
            lateral_error, heading_error = compute_track_errors(
                state, racetrack.centerline, closest_idx
            )
            
            # Compute track curvature for analysis
            track_curvature = compute_track_curvature(racetrack.centerline, closest_idx, lookahead_points=10)
            is_sharp_turn = track_curvature > 0.03  # Match controller threshold
            
            # Get controller outputs
            desired = controller(state, car.parameters, racetrack)
            control = lower_controller(state, desired, car.parameters)
            
            # Check for track violation
            track_violation = check_track_violation(pos, racetrack, closest_idx)
            
            # Estimate lookahead distance (approximate based on speed and conditions)
            v = state[3]
            abs_lat_err = abs(lateral_error)
            if abs_lat_err > 5.0:
                lookahead_dist = max(12.0, 10.0 * 1.2) if not is_sharp_turn else max(18.0, 10.0 * 1.8)
            elif is_sharp_turn:
                lookahead_dist = max(5.0, 10.0 * 0.7)
            else:
                lookahead_dist = 10.0 + 0.15 * v
            
            # Write to CSV
            writer.writerow({
                'time': f"{time:.3f}",
                'x': f"{state[0]:.6f}",
                'y': f"{state[1]:.6f}",
                'delta': f"{state[2]:.6f}",
                'v': f"{state[3]:.6f}",
                'phi': f"{state[4]:.6f}",
                'closest_idx': closest_idx,
                'closest_x': f"{closest_point[0]:.6f}",
                'closest_y': f"{closest_point[1]:.6f}",
                'lateral_error': f"{lateral_error:.6f}",
                'heading_error': f"{heading_error:.6f}",
                'delta_des': f"{desired[0]:.6f}",
                'v_des': f"{desired[1]:.6f}",
                'v_delta': f"{control[0]:.6f}",
                'a': f"{control[1]:.6f}",
                'track_violation': 1 if track_violation else 0,
                'track_curvature': f"{track_curvature:.6f}",
                'is_sharp_turn': 1 if is_sharp_turn else 0,
                'lookahead_dist': f"{lookahead_dist:.3f}"
            })
            
            # Print status every second
            if step % 10 == 0:
                violation_str = "VIOLATION!" if track_violation else "OK"
                print(f"t={time:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), "
                      f"v={state[3]:.2f}, δ={state[2]:.3f}, "
                      f"lat_err={lateral_error:.3f}, {violation_str}")
            
            # Update car
            car.update(control)
            
            # Track position for spinning detection
            position_history.append(pos.copy())
            if len(position_history) > SPIN_CHECK_WINDOW:
                position_history.pop(0)
            
            # Check for spinning (circular motion)
            is_spinning = False
            if len(position_history) >= SPIN_CHECK_WINDOW:
                # Calculate center of mass of recent positions
                positions_array = np.array(position_history)
                center = np.mean(positions_array, axis=0)
                
                # Calculate distances from center
                distances = np.linalg.norm(positions_array - center, axis=1)
                avg_radius = np.mean(distances)
                std_radius = np.std(distances)
                
                # If positions form a circle (small radius, low variation)
                if avg_radius < SPIN_RADIUS_THRESHOLD and std_radius < avg_radius * 0.5:
                    is_spinning = True
                    if spinning_start_time is None:
                        spinning_start_time = time
                        print(f"\nWARNING: Spinning detected at t={time:.1f}s (radius={avg_radius:.1f}m)")
                else:
                    spinning_start_time = None
            
            # Stop if spinning for too long
            if is_spinning and spinning_start_time is not None:
                spin_duration = time - spinning_start_time
                if spin_duration >= SPIN_TIME_THRESHOLD:
                    print(f"\nStopping: Car has been spinning for {spin_duration:.1f}s (threshold: {SPIN_TIME_THRESHOLD}s)")
                    break
            
            time += dt
            step += 1
            
            # Stop if car is clearly out of control (very far from track)
            if np.linalg.norm(pos - closest_point) > 100:
                print(f"\nStopping: Car too far from track at t={time:.2f}s")
                break
    
    print(f"\nSimulation complete. Logged {step} steps to {output_csv}")

def check_track_violation(car_position, racetrack, closest_idx):
    """
    Check if car is outside track boundaries.
    Uses the SAME logic as simulator.py to ensure accurate detection.
    Returns True if violating, False otherwise.
    """
    # Match simulator logic exactly
    min_dist_right = float('inf')
    min_dist_left = float('inf')
    
    # Find minimum distances to boundaries (like simulator does)
    for i in range(len(racetrack.right_boundary)):
        dist_right = np.linalg.norm(car_position - racetrack.right_boundary[i])
        dist_left = np.linalg.norm(car_position - racetrack.left_boundary[i])
        
        if dist_right < min_dist_right:
            min_dist_right = dist_right
        if dist_left < min_dist_left:
            min_dist_left = dist_left
    
    # Find closest centerline point
    centerline_distances = np.linalg.norm(racetrack.centerline - car_position, axis=1)
    closest_idx_check = np.argmin(centerline_distances)
    
    # Get vectors (same as simulator)
    to_right = racetrack.right_boundary[closest_idx_check] - racetrack.centerline[closest_idx_check]
    to_left = racetrack.left_boundary[closest_idx_check] - racetrack.centerline[closest_idx_check]
    to_car = car_position - racetrack.centerline[closest_idx_check]
    
    right_dist = np.linalg.norm(to_right)
    left_dist = np.linalg.norm(to_left)
    
    # Project car position onto boundary vectors (same as simulator)
    proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
    proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
    
    # Check violation (same as simulator)
    is_violating = proj_right > right_dist or proj_left > left_dist
    return is_violating

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 debug_controller.py <track.csv> <output.csv>")
        print("Example: python3 debug_controller.py ./racetracks/Montreal.csv debug_output.csv")
        sys.exit(1)
    
    track_path = sys.argv[1]
    output_csv = sys.argv[2]
    
    simulate_and_log(track_path, output_csv)

if __name__ == "__main__":
    main()

