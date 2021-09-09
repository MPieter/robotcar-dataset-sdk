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
from radar import load_radar, radar_polar_to_cartesian
import numpy as np
import cv2
import pandas as pd
from gridmap import updateGridMap, convertToProbabilities

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
cell_size = 3  # one cell has a width / height of 4 meters
k = 0.6  # Degredation factor

radar_odometry = pd.read_csv(radarodometry_path, sep=',')
radar_odometry['x_abs'] = radar_odometry['x'].cumsum()
radar_odometry['y_abs'] = radar_odometry['y'].cumsum()
radar_odometry['yaw_abs'] = radar_odometry['yaw'].cumsum()

title = "Radar Visualisation Example"

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

# initial odometry
idx = radar_odometry.source_radar_timestamp[radar_odometry.source_radar_timestamp == radar_timestamps[0]].index.tolist()[0]
car_x_init = radar_odometry.iloc[idx].x_abs  # meters (not 100% sure)
car_y_init = radar_odometry.iloc[idx].y_abs  # meters (idem)
car_yaw_init = radar_odometry.iloc[idx].yaw_abs

for radar_timestamp in radar_timestamps:
    filename = os.path.join(args.dir, str(radar_timestamp) + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)
    cart_img = cart_img / np.max(cart_img)

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
    car_x = (radar_odometry.iloc[idx].x_abs - car_x_init)  # meters (not 100% sure)
    car_y = (radar_odometry.iloc[idx].y_abs - car_y_init)  # meters (idem)
    car_yaw = radar_odometry.iloc[idx].yaw_abs - car_yaw_init

    updateGridMap(gridmap, cell_size, k, car_x, car_y, car_yaw, np.flipud(cart_img))
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

cv2.waitKey(0)
