#!/usr/bin/env python3
"""
Quick test script to verify controller implementation.
"""

import numpy as np
from racetrack import RaceTrack
from controller import controller, lower_controller
from racecar import RaceCar

def test_controller():
    """Test controller with both tracks."""
    
    tracks = [
        ("Montreal", "./racetracks/Montreal.csv"),
        ("IMS", "./racetracks/IMS.csv")
    ]
    
    for track_name, track_path in tracks:
        print(f"\n{'='*50}")
        print(f"Testing {track_name} track")
        print(f"{'='*50}")
        
        try:
            # Load track
            racetrack = RaceTrack(track_path)
            print(f"✓ Loaded {track_name} track")
            print(f"  Centerline points: {len(racetrack.centerline)}")
            
            # Test with initial state
            initial_state = racetrack.initial_state
            print(f"  Initial state: x={initial_state[0]:.2f}, y={initial_state[1]:.2f}, "
                  f"δ={initial_state[2]:.2f}, v={initial_state[3]:.2f}, φ={initial_state[4]:.2f}")
            
            # Get car parameters
            dummy_car = RaceCar(initial_state)
            parameters = dummy_car.parameters
            
            # Test high-level controller
            desired = controller(initial_state, parameters, racetrack)
            print(f"✓ High-level controller output: δ_des={desired[0]:.4f}, v_des={desired[1]:.2f}")
            assert desired.shape == (2,), f"Expected shape (2,), got {desired.shape}"
            
            # Test low-level controller
            control = lower_controller(initial_state, desired, parameters)
            print(f"✓ Low-level controller output: v_δ={control[0]:.4f}, a={control[1]:.2f}")
            assert control.shape == (2,), f"Expected shape (2,), got {control.shape}"
            
            # Test a few steps forward
            print(f"\n  Testing forward simulation (5 steps):")
            state = initial_state.copy()
            for step in range(5):
                desired = controller(state, parameters, racetrack)
                control = lower_controller(state, desired, parameters)
                
                # Simple Euler step (not RK4, just for testing)
                dt = 0.1
                state[0] += state[3] * np.cos(state[4]) * dt
                state[1] += state[3] * np.sin(state[4]) * dt
                state[2] += control[0] * dt
                state[3] += control[1] * dt
                state[4] += (state[3] / parameters[0]) * np.tan(state[2]) * dt
                
                # Normalize heading
                state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))
                
                print(f"    Step {step+1}: pos=({state[0]:.2f}, {state[1]:.2f}), "
                      f"v={state[3]:.2f}, φ={state[4]:.4f}")
            
            print(f"✓ {track_name} track test passed!")
            
        except Exception as e:
            print(f"✗ Error testing {track_name} track: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*50}")
    print("✓ All controller tests passed!")
    print(f"{'='*50}")
    return True

if __name__ == "__main__":
    success = test_controller()
    exit(0 if success else 1)


