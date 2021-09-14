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
from gridmap import updateGridMap, convertToProbabilities
from amplitude_gridmap import AmplitudeGridmap
import scipy.signal
import skimage.draw

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


def plot_peaks(cart_img, fft_data, azimuths, radar_resolution):
    cart_img_with_peaks = np.array(cart_img)
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

        for keypoint in zip(xidxs, yidxs):
            cv2.drawMarker(cart_img_with_peaks,
                           keypoint,
                           (255, 0, 0),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=10,
                           thickness=2,
                           line_type=cv2.LINE_AA)

    cv2.imshow("Cart img", cart_img_with_peaks)

    mask = skimage.draw.polygon2mask(cart_img_with_peaks.shape, keypoints)
    mask = mask.astype(np.uint8)
    mask *= 255
    cv2.imshow("Polygon mask", mask)

    cv2.waitKey(1)


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


for radar_timestamp in radar_timestamps:
    filename = os.path.join(args.dir, str(radar_timestamp) + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)
    cart_img = cart_img / np.max(cart_img)
    # plot_peaks(cart_img, fft_data, azimuths, radar_resolution)
    inv_sensor_model_mask = get_inverse_sensor_model_mask(cart_img, fft_data, azimuths, radar_resolution)

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
    se3_abs = se3_abs * se3_rel
    xyzrpy_abs = se3_to_components(se3_abs)

    car_x = xyzrpy_abs[0]  # meters (not 100% sure)
    car_y = xyzrpy_abs[1]  # meters (idem)
    car_yaw = xyzrpy_abs[5]

    # Occupancy gridmap
    updateGridMap(gridmap, car_x, car_y, car_yaw, np.flipud(cart_img), np.flipud(inv_sensor_model_mask))
    pgridmap = convertToProbabilities(gridmap)
    pgridmap = np.flipud(pgridmap)
    resized = cv2.resize(pgridmap, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow("Occupancy gridmap", resized)
    cv2.waitKey(1)

    tgridmap = pgridmap > 0.8
    tgridmap = tgridmap.astype(float)
    tgridmap = cv2.resize(tgridmap, (500, 500))
    cv2.imshow("Thresholded occupancy gridmap", tgridmap)
    cv2.waitKey(1)

    # Amplitude gridmap
    amplitudeGridMapWithMask.updateGridMap(car_x, car_y, car_yaw, np.flipud(cart_img), np.flipud(inv_sensor_model_mask))
    agridmap = amplitudeGridMapWithMask.get_amplitude_gridmap()
    agridmap = agridmap / np.max(agridmap)
    agridmap = np.flipud(agridmap)
    agridmap_resized = cv2.resize(agridmap, (1000, 1000), interpolation=cv2.INTER_AREA)
    cv2.imshow("Amplitude gridmap with inv sensor model mask", agridmap_resized)
    cv2.waitKey(1)

    amplitudeGridMapWithoutMask.updateGridMap(car_x, car_y, car_yaw, np.flipud(cart_img), np.flipud(inv_sensor_model_mask), False)
    agridmap_withoutmask = amplitudeGridMapWithoutMask.get_amplitude_gridmap()
    agridmap_withoutmask = agridmap_withoutmask / np.max(agridmap_withoutmask)
    agridmap_withoutmask = np.flipud(agridmap_withoutmask)
    agridmap_withoutmask_resized = cv2.resize(agridmap_withoutmask, (1000, 1000), interpolation=cv2.INTER_AREA)
    cv2.imshow("Amplitude gridmap without inv sensor model mask", agridmap_withoutmask_resized)
    cv2.waitKey(1)

cv2.waitKey(0)
