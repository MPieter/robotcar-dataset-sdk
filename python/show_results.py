import argparse
import os
from transform import build_se3_transform, se3_to_components
import numpy.linalg
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2

parser = argparse.ArgumentParser(description='Play back radar data from a given directory')

parser.add_argument('dir', type=str, help='Directory containing radar data.')

args = parser.parse_args()

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, 'radar.timestamps'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find timestamps file")

radarodometry_path = os.path.join(os.path.join(args.dir, os.pardir, 'gt', 'radar_odometry.csv'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find radar odometry file")


radar_odometry = pd.read_csv(radarodometry_path, sep=',')
radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
# initial odometry
idx = radar_odometry.source_radar_timestamp[radar_odometry.source_radar_timestamp == radar_timestamps[0]].index.tolist()[0]

se3_abs = matlib.identity(4)
xyzrpy_abs = se3_to_components(se3_abs)

odometry = np.loadtxt("odometry_results.txt")
odometry_orb_cart = np.loadtxt("odometry_results_with_cart_orb.txt")
odometry_orb_polar = np.loadtxt("odometry_results_with_polar_orb.txt")

car_pos = []
car_pos_odom_estimates = []
car_pos_odom_orb_cart_estimates = []
car_pos_odom_orb_polar_estimates = []

# Init initial robot pose and initial landmarks of initial frame
# For this state there is no uncertainty as we are certain of this position
car_pos_odom_estimates.append([
    odometry[0][0],
    odometry[0][1],
    odometry[0][2],
    odometry[0][3] + np.pi / 2
])
car_pos_odom_orb_cart_estimates.append([
    odometry_orb_cart[0][0],
    odometry_orb_cart[0][1],
    odometry_orb_cart[0][2],
    odometry_orb_cart[0][3] + np.pi / 2
])
car_pos_odom_orb_polar_estimates.append([
    odometry_orb_polar[0][0],
    odometry_orb_polar[0][1],
    odometry_orb_polar[0][2],
    odometry_orb_polar[0][3] + np.pi / 2
])

for odomIdx, odom in enumerate(odometry):
    if odomIdx != 0:
        # Odometry via minimization
        radar_timestamp = odom[0]
        dx = odom[1] - odometry[odomIdx - 1][1]
        dy = odom[2] - odometry[odomIdx - 1][2]
        dthetha = (odom[3] - odometry[odomIdx - 1][3])
        dr = np.sqrt(dx ** 2 + dy ** 2)
        dt = (radar_timestamp - odometry[odomIdx - 1][0]) / 1e6  # Timestamps are in microseconds

        ut = np.array([dr / dt, dthetha / dt])
        vt = ut[0]
        wt = ut[1]

        # Ground truth and analysis

        idx = radar_odometry.source_radar_timestamp[radar_odometry.source_radar_timestamp == radar_timestamp].index.tolist()[0]
        curr_radar_odometry = radar_odometry.iloc[idx]
        xyzrpy = np.array([curr_radar_odometry.x, curr_radar_odometry.y, curr_radar_odometry.z,
                           curr_radar_odometry.roll, curr_radar_odometry.pitch, curr_radar_odometry.yaw * (-1)])
        se3_rel = build_se3_transform(xyzrpy)
        xyzrpy_abs_before = se3_to_components(se3_abs)
        se3_abs = se3_abs * se3_rel
        xyzrpy_abs = se3_to_components(se3_abs)

        car_x = xyzrpy_abs[0]  # meters (not 100% sure)
        car_y = xyzrpy_abs[1]  # meters (idem)
        car_yaw = xyzrpy_abs[5]

        car_pos.append([
            radar_timestamp,
            -car_y,
            car_x,
            car_yaw
        ])
        car_pos_odom_estimates.append([
            radar_timestamp,
            car_pos_odom_estimates[-1][1] + np.cos(car_pos_odom_estimates[-1][3] + wt * dt) * vt * dt,
            car_pos_odom_estimates[-1][2] + np.sin(car_pos_odom_estimates[-1][3] + wt * dt) * vt * dt,
            car_pos_odom_estimates[-1][3] + dthetha
        ])

for odomIdx, odom in enumerate(odometry_orb_cart):
    radar_timestamp = odom[0]
    dx = odom[1] - odometry_orb_cart[odomIdx - 1][1]
    dy = odom[2] - odometry_orb_cart[odomIdx - 1][2]
    dthetha = (odom[3] - odometry_orb_cart[odomIdx - 1][3])
    dr = np.sqrt(dx ** 2 + dy ** 2)
    dt = (radar_timestamp - odometry_orb_cart[odomIdx - 1][0]) / 1e6

    ut = np.array([dr / dt, dthetha / dt])
    vt = ut[0]
    wt = ut[1]

    car_pos_odom_orb_cart_estimates.append([
        radar_timestamp,
        car_pos_odom_orb_cart_estimates[-1][1] + np.cos(car_pos_odom_orb_cart_estimates[-1][3] + wt * dt) * vt * dt,
        car_pos_odom_orb_cart_estimates[-1][2] + np.sin(car_pos_odom_orb_cart_estimates[-1][3] + wt * dt) * vt * dt,
        car_pos_odom_orb_cart_estimates[-1][3] + dthetha
    ])

for odomIdx, odom in enumerate(odometry_orb_polar):
    radar_timestamp = odom[0]
    dx = odom[1] - odometry_orb_polar[odomIdx - 1][1]
    dy = odom[2] - odometry_orb_polar[odomIdx - 1][2]
    dthetha = (odom[3] - odometry_orb_polar[odomIdx - 1][3])
    dr = np.sqrt(dx ** 2 + dy ** 2)
    dt = (radar_timestamp - odometry_orb_polar[odomIdx - 1][0]) / 1e6

    ut = np.array([dr / dt, dthetha / dt])
    vt = ut[0]
    wt = ut[1]

    car_pos_odom_orb_polar_estimates.append([
        radar_timestamp,
        car_pos_odom_orb_polar_estimates[-1][1] + np.cos(car_pos_odom_orb_polar_estimates[-1][3] + wt * dt) * vt * dt,
        car_pos_odom_orb_polar_estimates[-1][2] + np.sin(car_pos_odom_orb_polar_estimates[-1][3] + wt * dt) * vt * dt,
        car_pos_odom_orb_polar_estimates[-1][3] + dthetha
    ])

_, xs, ys, rads = zip(*car_pos)
_, xs_est, ys_est, rads_est = zip(*car_pos_odom_estimates)
_, xs_est_orb_cart, ys_est_orb_cart, rads_est_orb_cart = zip(*car_pos_odom_orb_cart_estimates)
_, xs_est_orb_polar, ys_est_orb_polar, rads_est_orb_polar = zip(*car_pos_odom_orb_polar_estimates)
plt.figure(1)
plt.clf()
plt.plot(xs, ys, label="Ground truth")
plt.plot(xs_est, ys_est, label="Odom estimation")
plt.plot(xs_est_orb_cart, ys_est_orb_cart, label="Odom orb cart")
plt.plot(xs_est_orb_polar, ys_est_orb_polar, label="Odom orb polar")
plt.title("Odometry")
plt.legend()

plt.figure(2)
plt.clf()
plt.plot(np.diff(xs), 'r', label="Ground truth")
plt.plot(np.diff(xs_est), 'bo', label="Estimation")
plt.plot(np.diff(xs_est_orb_cart), 'go', label='Cart Estimation')
plt.plot(np.diff(xs_est_orb_polar), 'ro', label='Polar Estimation')
plt.title("X")
plt.legend()

plt.figure(3)
plt.clf()
plt.plot(np.diff(ys), 'r', label="Ground truth")
plt.plot(np.diff(ys_est), 'bo', label="Estimation")
plt.plot(np.diff(ys_est_orb_cart), 'go', label='Cart Estimation')
plt.plot(np.diff(ys_est_orb_polar), 'ro', label='Polar Estimation')
plt.title("Y")
plt.legend()

plt.figure(4)
plt.clf()
plt.plot(np.diff(rads), 'r', label="Ground truth")
plt.plot(np.diff(rads_est), 'bo', label="Estimation")
plt.plot(np.diff(rads_est_orb_cart), 'go', label='Cart Estimation')
plt.plot(np.diff(rads_est_orb_polar), 'ro', label='Polar Estimation')
plt.title("Yaw")
plt.legend()

plt.figure(5)
plt.clf()
plt.plot(rads, 'r', label="Ground truth")
plt.plot(rads_est, 'bo', label="Estimation")
plt.title("Absolute Yaw")
plt.legend()
plt.show()


cv2.imshow('test', np.zeros((500, 500)))
cv2.waitKey(0)
