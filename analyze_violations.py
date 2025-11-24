#!/usr/bin/env python3
"""
Analyze violations relative to track geometry.
Helps identify where violations occur (straights vs turns) and recovery patterns.
"""

import csv
import numpy as np
import sys

def analyze_violations(csv_path):
    """Analyze violations and their relationship to track geometry."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    violations = [int(row['track_violation']) for row in data]
    curvatures = [float(row['track_curvature']) for row in data]
    is_sharp_turns = [int(row['is_sharp_turn']) for row in data]
    lateral_errors = [abs(float(row['lateral_error'])) for row in data]
    times = [float(row['time']) for row in data]
    lookahead_dists = [float(row['lookahead_dist']) for row in data]
    
    print(f'=== Violation Analysis: {csv_path} ===')
    print(f'Total steps: {len(data)}')
    print(f'Total violations: {sum(violations)} ({100*sum(violations)/len(violations):.1f}%)')
    print()
    
    if sum(violations) == 0:
        print('✓ No violations detected!')
        print()
        print('Track geometry statistics:')
        print(f'  - Avg curvature: {np.mean(curvatures):.4f}')
        print(f'  - Max curvature: {max(curvatures):.4f}')
        print(f'  - Sharp turn sections: {sum(is_sharp_turns)} ({100*sum(is_sharp_turns)/len(is_sharp_turns):.1f}%)')
        return
    
    # Find violation periods
    violation_indices = [i for i, v in enumerate(violations) if v == 1]
    
    print('Violation Statistics:')
    violation_curvatures = [curvatures[i] for i in violation_indices]
    violation_sharp_turns = [is_sharp_turns[i] for i in violation_indices]
    violation_lat_errors = [lateral_errors[i] for i in violation_indices]
    violation_times = [times[i] for i in violation_indices]
    violation_lookaheads = [lookahead_dists[i] for i in violation_indices]
    
    print(f'  - Avg curvature at violation: {np.mean(violation_curvatures):.4f}')
    print(f'  - Max curvature at violation: {max(violation_curvatures):.4f}')
    print(f'  - Sharp turn at violation: {sum(violation_sharp_turns)}/{len(violation_indices)} ({100*sum(violation_sharp_turns)/len(violation_indices):.1f}%)')
    print(f'  - Avg lateral error at violation: {np.mean(violation_lat_errors):.2f} m')
    print(f'  - Max lateral error at violation: {max(violation_lat_errors):.2f} m')
    print(f'  - Avg lookahead at violation: {np.mean(violation_lookaheads):.2f} m')
    print()
    
    # Compare to overall track
    print('Comparison to Overall Track:')
    print(f'  - Overall avg curvature: {np.mean(curvatures):.4f}')
    print(f'  - Overall max curvature: {max(curvatures):.4f}')
    print(f'  - Overall sharp turn sections: {100*sum(is_sharp_turns)/len(is_sharp_turns):.1f}%')
    print()
    
    # Find violation periods (consecutive violations)
    violation_periods = []
    in_violation = False
    period_start = None
    
    for i, v in enumerate(violations):
        if v == 1 and not in_violation:
            period_start = i
            in_violation = True
        elif v == 0 and in_violation:
            violation_periods.append((period_start, i-1, times[period_start], times[i-1]))
            in_violation = False
    
    if in_violation:
        violation_periods.append((period_start, len(violations)-1, times[period_start], times[-1]))
    
    print(f'Violation Periods: {len(violation_periods)}')
    for i, (start_idx, end_idx, start_time, end_time) in enumerate(violation_periods[:5]):
        duration = end_time - start_time
        start_curv = curvatures[start_idx]
        end_curv = curvatures[end_idx]
        start_sharp = is_sharp_turns[start_idx]
        end_sharp = is_sharp_turns[end_idx]
        
        print(f'  Period {i+1}: {start_time:.1f}s - {end_time:.1f}s (duration: {duration:.1f}s)')
        print(f'    Start: curvature={start_curv:.4f}, sharp_turn={start_sharp}, lat_err={lateral_errors[start_idx]:.2f}m')
        print(f'    End: curvature={end_curv:.4f}, sharp_turn={end_sharp}, lat_err={lateral_errors[end_idx]:.2f}m')
        print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_violations.py <debug_csv_file>")
        print("Example: python3 analyze_violations.py debug_montreal_track_aware.csv")
        sys.exit(1)
    
    analyze_violations(sys.argv[1])


