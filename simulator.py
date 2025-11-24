import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from time import time
from datetime import datetime
import os

from racetrack import RaceTrack
from racecar import RaceCar
from controller import lower_controller, controller

class Simulator:

    def __init__(self, rt : RaceTrack):
        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["font.size"] = 8

        self.rt = rt
        self.figure, self.axis = plt.subplots(1, 1)

        self.axis.set_xlabel("X"); self.axis.set_ylabel("Y")

        self.car = RaceCar(self.rt.initial_state.T)

        self.lap_time_elapsed = 0
        self.lap_start_time = None
        self.lap_finished = False
        self.lap_started = False
        self.track_limit_violations = 0
        self.currently_violating = False
        
        # Setup logging
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
        self.last_log_time = 0.0
        self.log_interval = 0.5  # Log every 500ms
        self.log_handle = open(self.log_file, 'w')
        self.log_handle.write("time,x,y,delta,v,phi,delta_des,v_des,v_delta,a,track_violations,currently_violating\n")
        print(f"Logging to: {self.log_file}")

    def check_track_limits(self):
        car_position = self.car.state[0:2]
        
        min_dist_right = float('inf')
        min_dist_left = float('inf')
        
        for i in range(len(self.rt.right_boundary)):
            dist_right = np.linalg.norm(car_position - self.rt.right_boundary[i])
            dist_left = np.linalg.norm(car_position - self.rt.left_boundary[i])
            
            if dist_right < min_dist_right:
                min_dist_right = dist_right
            if dist_left < min_dist_left:
                min_dist_left = dist_left
        
        centerline_distances = np.linalg.norm(self.rt.centerline - car_position, axis=1)
        closest_idx = np.argmin(centerline_distances)
        
        to_right = self.rt.right_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_left = self.rt.left_boundary[closest_idx] - self.rt.centerline[closest_idx]
        to_car = car_position - self.rt.centerline[closest_idx]
        
        right_dist = np.linalg.norm(to_right)
        left_dist = np.linalg.norm(to_left)
        
        proj_right = np.dot(to_car, to_right) / right_dist if right_dist > 0 else 0
        proj_left = np.dot(to_car, to_left) / left_dist if left_dist > 0 else 0
        
        is_violating = proj_right > right_dist or proj_left > left_dist
        
        if is_violating and not self.currently_violating:
            self.track_limit_violations += 1
            self.currently_violating = True
            # Log violation details
            print(f"VIOLATION #{self.track_limit_violations} at t={self.lap_time_elapsed:.2f}s: "
                  f"pos=({car_position[0]:.2f}, {car_position[1]:.2f}), "
                  f"proj_right={proj_right:.2f}/{right_dist:.2f}, proj_left={proj_left:.2f}/{left_dist:.2f}")
        elif not is_violating:
            self.currently_violating = False

    def run(self):
        try:
            if self.lap_finished:
                self.close_log()
                exit()

            self.figure.canvas.flush_events()
            self.axis.cla()

            self.rt.plot_track(self.axis)

            self.axis.set_xlim(self.car.state[0] - 200, self.car.state[0] + 200)
            self.axis.set_ylim(self.car.state[1] - 200, self.car.state[1] + 200)

            desired = controller(self.car.state, self.car.parameters, self.rt)
            cont = lower_controller(self.car.state, desired, self.car.parameters)
            self.car.update(cont)
            self.update_status()
            self.check_track_limits()
            
            # Log every 500ms
            if self.lap_start_time is not None:
                current_time = time() - self.lap_start_time
                if current_time - self.last_log_time >= self.log_interval:
                    self.log_state(current_time, desired, cont)
                    self.last_log_time = current_time

            self.axis.arrow(
                self.car.state[0], self.car.state[1], \
                self.car.wheelbase*np.cos(self.car.state[4]), \
                self.car.wheelbase*np.sin(self.car.state[4])
            )

            self.axis.text(
                self.car.state[0] + 195, self.car.state[1] + 195, "Lap completed: " + str(self.lap_finished),
                horizontalalignment="right", verticalalignment="top",
                fontsize=8, color="Red"
            )

            self.axis.text(
                self.car.state[0] + 195, self.car.state[1] + 170, "Lap time: " + f"{self.lap_time_elapsed:.2f}",
                horizontalalignment="right", verticalalignment="top",
                fontsize=8, color="Red"
            )

            self.axis.text(
                self.car.state[0] + 195, self.car.state[1] + 155, "Track violations: " + str(self.track_limit_violations),
                horizontalalignment="right", verticalalignment="top",
                fontsize=8, color="Red"
            )

            self.figure.canvas.draw()
            return True

        except KeyboardInterrupt:
            self.close_log()
            exit()
    
    def close_log(self):
        """Close log file and print summary."""
        if hasattr(self, 'log_handle') and self.log_handle:
            self.log_handle.close()
            print(f"\nLog file closed: {self.log_file}")
            print(f"Total violations: {self.track_limit_violations}")

    def update_status(self):
        progress = np.linalg.norm(self.car.state[0:2] - self.rt.centerline[0, 0:2], 2)

        if progress > 10.0 and not self.lap_started:
            self.lap_started = True
    
        if progress <= 1.0 and self.lap_started and not self.lap_finished:
            self.lap_finished = True
            self.lap_time_elapsed = time() - self.lap_start_time
            # Log final statistics
            print(f"\n=== LAP COMPLETE ===")
            print(f"Lap time: {self.lap_time_elapsed:.2f}s")
            print(f"Total violations: {self.track_limit_violations}")

        if not self.lap_finished and self.lap_start_time is not None:
            self.lap_time_elapsed = time() - self.lap_start_time
    
    def log_state(self, current_time, desired, control):
        """Log car state, controller outputs, and track status to file."""
        state = self.car.state
        self.log_handle.write(
            f"{current_time:.3f},"
            f"{state[0]:.6f},{state[1]:.6f},"
            f"{state[2]:.6f},{state[3]:.6f},{state[4]:.6f},"
            f"{desired[0]:.6f},{desired[1]:.6f},"
            f"{control[0]:.6f},{control[1]:.6f},"
            f"{self.track_limit_violations},{int(self.currently_violating)}\n"
        )
        self.log_handle.flush()  # Ensure data is written immediately

    def start(self):
        # Run the simulation loop every 1 second.
        self.timer = self.figure.canvas.new_timer(interval=1)
        self.timer.add_callback(self.run)
        self.lap_start_time = time()
        self.timer.start()