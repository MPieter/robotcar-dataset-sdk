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


cart_img1_rotated = np.array([])
cart_img2_rotated = np.array([])


def calc_transform_mean_error(xy_trans):
    cart_img_transformed = xy_transf(cart_img2_rotated, xy_trans)

    # only select subregion that is guaranteed to overlap

    cart_img1_subset = cart_img1_rotated[100:-100, 100:-100]
    cart_img2_subset = cart_img_transformed[100:-100, 100:-100]
    return np.mean(np.abs(cart_img1_subset - cart_img2_subset))


def calc_corr(xy_trans):
    cart_img_transformed = xy_transf(cart_img2_rotated, xy_trans)
    cart_img1_subset = cart_img1_rotated[100:-100, 100:-100]
    cart_img2_subset = cart_img_transformed[100:-100, 100:-100]
    corrcoef = np.corrcoef(cart_img1_subset.ravel(), cart_img2_subset.ravel())
    return -corrcoef[0, 1]


def determine_best_transform(cart_img1, cart_img2, gt_dy, gt_img1_yaw, gt_img2_yaw):
    gt_dy_idx = gt_dy * 4  # 0.25 m per cell in cart_img
    # TODO remove gt_img1_yaw, gt_img2_yaw, only focus on dx and dy for now
    global cart_img1_rotated
    global cart_img2_rotated
    cart_img1_rotated = ndimage.rotate(cart_img1, np.rad2deg(gt_img1_yaw), reshape=False)
    cart_img2_rotated = ndimage.rotate(cart_img2, np.rad2deg(gt_img2_yaw), reshape=False)

    best_params = optimize.fmin_powell(calc_corr, [0, 0])

    x_optimal = best_params[0]
    y_optimal = best_params[1]

    return x_optimal / 4, y_optimal / 4


def xy_transf(img, xy_trans):
    return ndimage.affine_transform(np.reshape(img, (501, 501)), [1, 1], np.array(xy_trans), order=1)


def x_trans(img, x_trans):
    # Make a 0-filled array of same shape as `img_slice`
    trans_img = np.zeros(img.shape)
    # Use slicing to select voxels out of the image and move them
    # up or down on the first (x) axis
    if x_trans < 0:
        trans_img[:x_trans, :] = img[-x_trans:, :]
    elif x_trans == 0:
        trans_img[:, :] = img
    else:
        trans_img[x_trans:, :] = img[:-x_trans, :]
    return trans_img


def y_trans(img, y_trans):
    # Make a 0-filled array of same shape as `img_slice`
    trans_img = np.zeros(img.shape)
    # Use slicing to select voxels out of the image and move them
    # up or down on the first (x) axis
    if y_trans < 0:
        trans_img[:, :y_trans] = img[:, -y_trans:]
    elif y_trans == 0:
        trans_img[:, :] = img
    else:
        trans_img[:, y_trans:] = img[:, :-y_trans]
    return trans_img


car_pos = []
car_pos_estimates = []

previous_cart_img = np.array([])

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
        x_optimal, y_optimal = determine_best_transform(previous_cart_img, cart_img, car_y - xyzrpy_abs_before[1], xyzrpy_abs_before[5], car_yaw)
        car_pos_estimates.append([car_pos_estimates[-1][0] + x_optimal, car_pos_estimates[-1][1] + y_optimal])
    else:
        # init car_pos_estimates
        car_pos_estimates.append([car_x, car_y])
    previous_cart_img = cart_img

    car_pos.append([car_x, car_y])

    xs, ys = zip(*car_pos)
    xs_est, ys_est = zip(*car_pos_estimates)
    plt.figure(1)
    plt.clf()
    plt.plot(xs, ys, label="Ground truth")
    plt.plot(xs_est, ys_est, label="Estimation")
    plt.title("Odometry")
    plt.legend()
    plt.ion()
    plt.show()


xs, ys = zip(*car_pos)
xs_est, ys_est = zip(*car_pos_estimates)
plt.figure()
plt.plot(xs, ys, label="Ground truth")
plt.plot(xs_est, ys_est, label="Estimation")
plt.title("Odometry")
plt.legend()
plt.show()

cv2.waitKey(0)
