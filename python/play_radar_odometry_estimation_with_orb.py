################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
from transform import build_se3_transform, se3_to_components
from radar import load_radar, radar_polar_to_cartesian
import numpy as np
import numpy.matlib as matlib
import cv2
import pandas as pd
from amplitude_gridmap import AmplitudeGridmap
import scipy.signal
import skimage.draw
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import optimize


parser = argparse.ArgumentParser(description='Play back radar data from a given directory')

parser.add_argument('dir', type=str, help='Directory containing radar data.')

args = parser.parse_args()

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, 'radar.timestamps'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find timestamps file")

radarodometry_path = os.path.join(os.path.join(args.dir, os.pardir, 'gt', 'radar_odometry.csv'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find radar odometry file")

# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 501  # pixels
interpolate_crossover = True

# Occupancy grid map
gridmap = np.full((300, 300), 0.5)
amplitudeGridMapWithMask = AmplitudeGridmap()
amplitudeGridMapWithoutMask = AmplitudeGridmap()

radar_odometry = pd.read_csv(radarodometry_path, sep=',')

title = "Radar Visualisation Example"

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

# initial odometry
idx = radar_odometry.source_radar_timestamp[radar_odometry.source_radar_timestamp == radar_timestamps[0]].index.tolist()[0]

se3_abs = matlib.identity(4)
xyzrpy_abs = se3_to_components(se3_abs)


def get_inverse_sensor_model_mask(cart_img, fft_data, azimuths, radar_resolution):
    keypoints = []
    for azimuth in range(fft_data.shape[0]):
        azimuth_data = fft_data[azimuth, :, 0]
        peaks = scipy.signal.find_peaks(azimuth_data,
                                        distance=100,
                                        height=np.max(azimuth_data) * 0.70)[0]

        xs = peaks * radar_resolution * np.sin(azimuths[azimuth])
        ys = peaks * radar_resolution * np.cos(azimuths[azimuth]) * (-1)

        xidxs = np.multiply(xs, 4).astype(int) + 250
        yidxs = np.multiply(ys, 4).astype(int) + 250

        keypoints.extend(list(zip(yidxs, xidxs)))

    mask = skimage.draw.polygon2mask(cart_img.shape, keypoints)
    return mask


def determine_best_transform(cart_img1, cart_img2, homMatrix, current_yaw):
    try:
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        img1 = ((cart_img1 / np.max(cart_img1)) * 255).astype(np.uint8)
        img2 = ((cart_img2 / np.max(cart_img2)) * 255).astype(np.uint8)

        kpt1, des1 = orb.detectAndCompute(img1, None)
        kpt2, des2 = orb.detectAndCompute(img2, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        vis_matches = cv2.drawMatches(img1, kpt1, img2, kpt2, matches[:30], None, flags=2)

        cv2.imshow('Matches', vis_matches)
        cv2.waitKey(1)

        matchesSrcPoints = []
        matchesDstPoints = []

        CART_RESOLUTION = 0.25
        X_CENTER = int(cart_img1.shape[0] / 2)
        Y_CENTER = int(cart_img1.shape[1] / 2)

        for match in matches:
            x_src = (kpt1[match.queryIdx].pt[0] - X_CENTER) * CART_RESOLUTION
            y_src = (kpt1[match.queryIdx].pt[1] - Y_CENTER) * CART_RESOLUTION
            x_dst = (kpt2[match.trainIdx].pt[0] - X_CENTER) * CART_RESOLUTION
            y_dst = (kpt2[match.trainIdx].pt[1] - Y_CENTER) * CART_RESOLUTION
            matchesSrcPoints.append([x_src, y_src])
            matchesDstPoints.append([x_dst, y_dst])

        matchesSrcPoints = np.array(matchesSrcPoints)
        matchesDstPoints = np.array(matchesDstPoints)
        inliers = np.array([])
        retval, inliers = cv2.estimateAffinePartial2D(matchesSrcPoints, matchesDstPoints, inliers, cv2.RANSAC, 1, 3000)
        if retval is not None:
            if homMatrix is None:
                homMatrix = np.vstack([retval, [0, 0, 1]])
            else:
                homMatrix = np.matmul(homMatrix, np.vstack([retval, [0, 0, 1]]))

        return homMatrix
    except Exception as e:
        print(e)
        return 0, 0, 0


car_pos = []
car_pos_estimates = []

previous_cart_img = np.array([])
homMatrix = None

for radar_timestamp in radar_timestamps:
    filename = os.path.join(args.dir, str(radar_timestamp) + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)
    # cart_img = cart_img / np.max(cart_img)
    # # plot_peaks(cart_img, fft_data, azimuths, radar_resolution)
    # # inv_sensor_model_mask = get_inverse_sensor_model_mask(cart_img, fft_data, azimuths, radar_resolution)

    # Combine polar and cartesian for visualisation
    # The raw polar data is resized to the height of the cartesian representation
    downsample_rate = 4
    fft_data_vis = fft_data[:, ::downsample_rate]
    resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
    fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
    vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))

    cv2.imshow(title, vis * 2.)  # The data is doubled to improve visualisation
    cv2.waitKey(1)

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

    if previous_cart_img.size > 0:
        homMatrix = determine_best_transform(previous_cart_img, cart_img, homMatrix, car_pos_estimates[-1][2])
        newPos = np.matmul(homMatrix, np.array([0, 0, 1]).transpose())
        car_pos_estimates.append([
            newPos[0],
            newPos[1],
            0  # TODO : get angle from homMatrix
        ])
    else:
        # init car_pos_estimates
        car_pos_estimates.append([car_x, car_y, car_yaw])
    previous_cart_img = cart_img

    car_pos.append([car_x, car_y, car_yaw])

    xs, ys, rads = zip(*car_pos)
    xs_est, ys_est, rads_est = zip(*car_pos_estimates)
    plt.figure(1)
    plt.clf()
    plt.plot(xs, ys, label="Ground truth")
    plt.plot(ys_est, xs_est, label="Estimation")
    plt.title("Odometry")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.plot(np.diff(xs), 'r', label="Ground truth")
    plt.plot(np.diff(ys_est), 'bo', label="Estimation")
    plt.title("X")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(3)
    plt.clf()
    plt.plot(np.diff(ys), 'r', label="Ground truth")
    plt.plot(np.diff(xs_est), 'bo', label="Estimation")
    plt.title("Y")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(4)
    plt.clf()
    plt.plot(np.diff(rads), 'r', label="Ground truth")
    plt.plot(np.diff(rads_est), 'bo', label="Estimation")
    plt.title("Yaw")
    plt.legend()
    plt.ion()
    plt.show()


xs, ys, rads = zip(*car_pos)
xs_est, ys_est, rads_est = zip(*car_pos_estimates)
plt.figure(1)
plt.clf()
plt.plot(xs, ys, label="Ground truth")
plt.plot(xs_est, ys_est, label="Estimation")
plt.title("Odometry")
plt.legend()

plt.figure(2)
plt.clf()
plt.plot(np.diff(xs), 'r', label="Ground truth")
plt.plot(np.diff(xs_est), 'bo', label="Estimation")
plt.title("X")
plt.legend()

plt.figure(3)
plt.clf()
plt.plot(np.diff(ys), 'r', label="Ground truth")
plt.plot(np.diff(ys_est), 'bo', label="Estimation")
plt.title("Y")
plt.legend()

plt.figure(4)
plt.clf()
plt.plot(np.diff(rads), 'r', label="Ground truth")
plt.plot(np.diff(rads_est), 'bo', label="Estimation")
plt.title("Yaw")
plt.legend()

plt.show()
cv2.waitKey(0)
