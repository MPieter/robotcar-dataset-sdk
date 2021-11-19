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


def update_state(state, ut, dt, dx_gt, dy_gt, dthetha_gt):
    vt = ut[0]
    wt = ut[1]

    # Only need to update the first 3 elements: x, y, thetha
    updated_state = np.copy(state)

    dx = -vt / wt * np.sin(state[2]) + vt / wt * np.sin(state[2] + wt * dt)
    dy = vt / wt * np.cos(state[2]) - vt / wt * np.cos(state[2] + wt * dt)
    dthetha = wt * dt

    updated_state[0] = state[0] + dy * (-1)
    updated_state[1] = state[1] + dx
    updated_state[2] = state[2] + dthetha

    return updated_state


def update_cov(state, cov, ut, dt, Rtx):
    Gt = np.diag(np.full(cov.shape[0], 1))
    vt = ut[0]
    wt = ut[1]
    Gt[0, 2] = - vt / wt * np.cos(state[2]) + vt / wt * np.cos(state[2] + wt * dt)
    Gt[1, 2] = - vt / wt * np.sin(state[2]) + vt / wt * np.sin(state[2] + wt * dt)

    Fx = np.zeros((3, cov.shape[1]))
    Fx[0, 0] = 1
    Fx[1, 1] = 1
    Fx[2, 2] = 1

    # TODO could be sped-up by only updating the relevant parts of the multiplication
    return Gt @ cov @ np.transpose(Gt) + np.transpose(Fx) @ Rtx @ Fx


def kalman_gain(updated_cov, Q, landmarkIdx, delta, q):
    lowHi = np.array([
        [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
        [delta[1], -delta[0], -q, -delta[1], delta[0]]])
    lowHi = np.divide(lowHi, q)
    Fxj = np.zeros((5, updated_cov.shape[0]))
    Fxj[0, 0] = 1
    Fxj[1, 1] = 1
    Fxj[2, 2] = 1
    Fxj[3, 3 + landmarkIdx * 2] = 1
    Fxj[4, 3 + landmarkIdx * 2 + 1] = 1
    Hi = lowHi @ Fxj
    return updated_cov @ np.transpose(Hi) @ numpy.linalg.inv(Hi @ updated_cov @ np.transpose(Hi) + Q), Hi


def ekf(state, cov, ut, zt, dt, Rtx, Q, dx, dy, dthetha):
    # prediction
    updated_state = update_state(state, ut, dt, dx, dy, dthetha)
    updated_cov = update_cov(state, cov, ut, dt, Rtx)

    # dbg
    predicted_state = np.copy(updated_state)
    predicted_cov = np.copy(updated_cov)

    # correction
    for obs_t in zt:
        landmarkNew = obs_t[0]
        landmarkIdx = obs_t[1]
        landmark_r = obs_t[2]
        landmark_thetha = obs_t[3]

        if landmarkNew:
            updated_state[3 + landmarkIdx * 2] = updated_state[0] + landmark_r * np.cos(landmark_thetha + updated_state[2])
            updated_state[3 + landmarkIdx * 2 + 1] = updated_state[1] + landmark_r * np.sin(landmark_thetha + updated_state[2])

        # Calculate expected observation
        delta = np.array([updated_state[3 + landmarkIdx * 2] - updated_state[0],
                          updated_state[3 + landmarkIdx * 2 + 1] - updated_state[1]]).transpose()
        q = np.transpose(delta) @ delta
        obs = np.array([landmark_r, landmark_thetha]).transpose()
        expected_obs = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - updated_state[2]]).transpose()

        # Get kalman gain for this observation
        Ki, Hi = kalman_gain(updated_cov, Q, landmarkIdx, delta, q)
        updated_state = updated_state + Ki @ (obs - expected_obs)
        updated_cov = (np.identity(updated_cov.shape[0]) - Ki @ Hi) @ updated_cov
    return predicted_state, predicted_state, predicted_cov


# EKF Slam Parameters
N_LANDMARKS = 1000

varR = 0.0438 * 2  # Variance of range measurements of landmarks
varThetha = 2 / 180  # Variance of thetha measurements of landmarks
Q = np.diag([varR, varThetha])
# End Parameters

odometry = np.loadtxt("odometry_results.txt")
landmarks = np.loadtxt("landmarks.txt")

# Filter landmarks to only include those with a certain track length
filtered_landmarks = list(filter(lambda landmark: landmark[7] > 6, landmarks))
landmarks = filtered_landmarks

car_pos = []
car_pos_odom_estimates = []
car_pos_ekf_predicted_estimates = []
car_pos_ekf_estimates = []
state = np.zeros((2 * N_LANDMARKS + 3))
cov = np.zeros((2 * N_LANDMARKS + 3, 2 * N_LANDMARKS + 3))

# Init initial robot pose and initial landmarks of initial frame
# For this state there is no uncertainty as we are certain of this position
state[0] = odometry[0][1]
state[1] = odometry[0][2]
state[2] = odometry[0][3] - np.pi / 2
car_pos_odom_estimates.append([
    odometry[0][0],
    state[0],
    state[1],
    state[2]
])
car_pos_ekf_predicted_estimates.append([
    odometry[0][0],
    state[0],
    state[1],
    state[2]
])
car_pos_ekf_estimates.append([
    odometry[0][0],
    state[0],
    state[1],
    state[2]
])

# TODO init initial landmarks
landmarksInitialFrame = filter(lambda x: x[1] == 0, landmarks)
if len(list(landmarksInitialFrame)) > 0:
    maxLandmarkId = max(landmarksInitialFrame, key=lambda x: x[0])[0]
else:
    maxLandmarkId = 0

for landmark in landmarksInitialFrame:
    landmarkId = landmark[0]
    landmark_r = landmark[5]
    landmark_thetha = landmark[6]
    state[3 + landmarkId * 2] = state[0] + landmark_r * np.cos(landmark_thetha + state[2])
    state[3 + landmarkId * 2 + 1] = state[1] + landmark_r * np.sin(landmark_thetha + state[2])

    cov[3 + landmarkId * 2, 3 + landmarkId * 2] = landmark_r * np.cos(landmark_thetha + state[2]) * 0.1  # TODO check if this is a good initial value
    cov[3 + landmarkId * 2 + 1, 3 + landmarkId * 2 + 1] = landmark_r * np.sin(landmark_thetha + state[2]) * 0.1  # TODO idem
# for i in range(3, 2 * N_LANDMARKS + 3):
#     cov[i, i] = 3.4028235e+38


def getLandMarksCurrentFrame(landmarks, radar_timestamp, highestLandmarkId):
    landmarksCurrentFrame = filter(lambda x: x[2] == radar_timestamp, landmarks)

    result = []
    for landmark in landmarksCurrentFrame:
        landmarkId = int(landmark[0])
        landmark_r = landmark[5]
        landmark_thetha = landmark[6]

        landmarkNew = landmarkId > highestLandmarkId
        result.append((landmarkNew, landmarkId, landmark_r, landmark_thetha))
    return result


for odomIdx, odom in enumerate(odometry):
    if odomIdx != 0:
        radar_timestamp = odom[0]
        dx = odom[1] - odometry[odomIdx - 1][1]
        dy = odom[2] - odometry[odomIdx - 1][2]
        dthetha = odom[3] - odometry[odomIdx - 1][3]
        dr = np.sqrt(dx ** 2 + dy ** 2)
        dt = (radar_timestamp - odometry[odomIdx - 1][0]) / 1e6  # Timestamps are in microseconds

        ut = np.array([dr / dt, dthetha / dt])

        zt = getLandMarksCurrentFrame(landmarks, radar_timestamp, maxLandmarkId)
        if len(zt) > 0:
            maxLandmarkId = max(maxLandmarkId, max(zt, key=lambda x: x[0])[0])

        # TODO noise on velocity measurements
        Rtx = np.zeros((3, 3))  # TODO is this matrix diagonal?
        Rtx[0, 0] = dx * 0.05
        Rtx[1, 1] = dy * 0.05
        Rtx[2, 2] = dthetha * 0.05

        predicted_state, state, cov = ekf(state, cov, ut, zt, dt, Rtx, Q, dx, dy, dthetha)

        # Ground truth and analysis

        idx = radar_odometry.source_radar_timestamp[radar_odometry.source_radar_timestamp == radar_timestamp].index.tolist()[0]
        curr_radar_odometry = radar_odometry.iloc[idx]
        xyzrpy = np.array([curr_radar_odometry.x, curr_radar_odometry.y, curr_radar_odometry.z,
                           curr_radar_odometry.roll, curr_radar_odometry.pitch, curr_radar_odometry.yaw])
        se3_rel = build_se3_transform(xyzrpy)
        xyzrpy_abs_before = se3_to_components(se3_abs)
        se3_abs = se3_abs * se3_rel
        xyzrpy_abs = se3_to_components(se3_abs)

        car_x = xyzrpy_abs[0]  # meters (not 100% sure)
        car_y = xyzrpy_abs[1]  # meters (idem)
        car_yaw = xyzrpy_abs[5]

        car_pos.append([
            radar_timestamp,
            car_x,
            car_y,
            car_yaw - np.pi / 2
        ])
        car_pos_odom_estimates.append([
            radar_timestamp,
            car_pos_odom_estimates[-1][1] + dx,
            car_pos_odom_estimates[-1][2] + dy,
            car_pos_odom_estimates[-1][3] + dthetha
        ])
        car_pos_ekf_predicted_estimates.append([
            radar_timestamp,
            predicted_state[0],
            predicted_state[1],
            predicted_state[2]
        ])
        car_pos_ekf_estimates.append([
            radar_timestamp,
            state[0],
            state[1],
            state[2]
        ])

        _, xs, ys, rads = zip(*car_pos)
        _, xs_odom_est, ys_odom_est, rad_odom_est = zip(*car_pos_odom_estimates)
        _, xs_ekf_predicted_est, ys_ekf_predicted_est, rad_ekf_predicted_est = zip(*car_pos_ekf_predicted_estimates)
        _, xs_ekf_est, ys_ekf_est, rad_ekf_est = zip(*car_pos_ekf_estimates)
        plt.figure(1)
        plt.clf()
        plt.plot(xs, ys, label="Ground truth")
        plt.plot(xs_odom_est, ys_odom_est, label="Odom estimation")
        plt.plot(xs_ekf_predicted_est, ys_ekf_predicted_est, label="EKF Prediction step")
        plt.plot(xs_ekf_est, ys_ekf_est, label="EKF Estimation")
        plt.title("Odometry")
        plt.legend()
        plt.ion()
        plt.show()

        cv2.imshow('test', np.zeros((500, 500)))
        cv2.waitKey(1)