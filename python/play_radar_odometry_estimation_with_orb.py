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


def writeLandmarks(landmarks):
    filename = "landmarks.txt"
    with open(filename, "w+") as cfarFile:
        goodLandmarks = filter(lambda x: len(x) > 4, landmarks)
        for trackId, track in enumerate(goodLandmarks):
            for item in track:
                cfarFile.writelines([str(trackId) + " " + str(item[0]) + " " + str(item[1]) + " " + str(item[2]) + " " +
                                     str(item[3]) + " " + str(item[4]) + " " + str(item[5]) + " " + str(len(track)) + "\n"])


def determine_best_transform(radar_timestamp_idx, radar_timestamp, cart_img1, cart_img2, homMatrix, current_yaw, landmarks):
    try:
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        img1 = ((cart_img1 / np.max(cart_img1)) * 255).astype(np.uint8)
        img2 = ((cart_img2 / np.max(cart_img2)) * 255).astype(np.uint8)

        kpt1, des1 = orb.detectAndCompute(img1, None)
        kpt2, des2 = orb.detectAndCompute(img2, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        vis_matches = cv2.drawMatches(img1, kpt1, img2, kpt2, matches[:100], None, flags=2)
        cv2.imshow('Matches CART', vis_matches)
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

            # check tracks
            for matchIdx, matchSrcPoint in enumerate(matchesSrcPoints):
                if inliers[matchIdx][0] == 1:
                    x_src = matchSrcPoint[0]
                    y_src = matchSrcPoint[1]
                    x_dst = matchesDstPoints[matchIdx][0]
                    y_dst = matchesDstPoints[matchIdx][1]
                    r_dst = np.sqrt(x_dst ** 2 + y_dst ** 2)
                    thetha_dst = np.arctan2(y_dst, x_dst)
                    found_track = False
                    for track in landmarks:
                        if track[-1][0] == radar_timestamp_idx - 1 and track[-1][2] == x_src and track[-1][3] == y_src:
                            track.append((radar_timestamp_idx, radar_timestamp, x_dst, y_dst, r_dst, thetha_dst))
                            found_track = True
                    if found_track == False:
                        landmarks.append([(radar_timestamp_idx, radar_timestamp, x_dst, y_dst, r_dst, thetha_dst)])

        return homMatrix
    except Exception as e:
        print(e)
        return homMatrix


def determine_best_transform_polar(polar1, polar2, homMatrix, azimuths, radar_resolution):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    polar1 = (polar1 / np.max(polar1) * 255).astype(np.uint8)
    polar2 = (polar2 / np.max(polar2) * 255).astype(np.uint8)

    kpt1, des1 = orb.detectAndCompute(polar1, None)
    kpt2, des2 = orb.detectAndCompute(polar2, None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    vis_matches = cv2.drawMatches(polar1, kpt1, polar2, kpt2, matches[:30], None, flags=2)
    cv2.imshow('Matches POLAR', vis_matches)
    cv2.waitKey(1)

    matchesSrcPoints = []
    matchesDstPoints = []

    for match in matches:
        a_src = azimuths[int(np.round(kpt1[match.queryIdx].pt[1]))][0]
        r_src = kpt1[match.queryIdx].pt[0] * radar_resolution[0]
        a_dst = azimuths[int(np.round(kpt2[match.trainIdx].pt[1]))][0]
        r_dst = kpt2[match.trainIdx].pt[0] * radar_resolution[0]

        x_src = r_src * np.sin(a_src)
        y_src = -r_src * np.cos(a_src)
        x_dst = r_dst * np.sin(a_dst)
        y_dst = -r_dst * np.cos(a_dst)
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


car_pos = []
car_pos_estimates_cart_orb = []
car_pos_estimates_polar_orb = []

previous_cart_img = np.array([])
previous_fft_data = np.array([])
homMatrix_cart_orb = None
homMatrix_polar_orb = None

landmarks = []

for radar_timestamp_idx, radar_timestamp in enumerate(radar_timestamps):
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

    cv2.imshow("FFT data", fft_data)
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
        homMatrix_cart_orb = determine_best_transform(radar_timestamp_idx, radar_timestamp, previous_cart_img, cart_img, homMatrix_cart_orb, car_pos_estimates_cart_orb[-1][2], landmarks)
        newPos = np.matmul(homMatrix_cart_orb, np.array([0, 0, 1]).transpose())
        car_pos_estimates_cart_orb.append([
            newPos[0],
            newPos[1],
            0  # TODO : get angle from homMatrix_cart_orb
        ])

        homMatrix_polar_orb = determine_best_transform_polar(previous_fft_data, fft_data, homMatrix_polar_orb, azimuths, radar_resolution)
        newPos = np.matmul(homMatrix_polar_orb, np.array([0, 0, 1]).transpose())
        car_pos_estimates_polar_orb.append([
            newPos[0],
            newPos[1],
            0  # TODO : get angle from hom matrix
        ])
    else:
        # init car_pos_estimates_cart_orb
        car_pos_estimates_cart_orb.append([car_x, car_y, car_yaw])
        car_pos_estimates_polar_orb.append([car_x, car_y, car_yaw])
    previous_cart_img = cart_img
    previous_fft_data = fft_data

    car_pos.append([car_x, car_y, car_yaw])

    xs, ys, rads = zip(*car_pos)
    xs_est, ys_est, rads_est = zip(*car_pos_estimates_cart_orb)
    xs_est_polar, ys_est_polar, rads_est_polar = zip(*car_pos_estimates_polar_orb)
    plt.figure(1)
    plt.clf()
    plt.plot(xs, ys, label="Ground truth")
    plt.plot(ys_est, xs_est, label="Cart Estimation")
    plt.plot(ys_est_polar, xs_est_polar, label="Polar Estimation")
    plt.title("Odometry")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.plot(np.diff(xs), 'r', label="Ground truth")
    plt.plot(np.diff(ys_est), 'bo', label="Cart Estimation")
    plt.plot(np.diff(ys_est_polar), 'go', label="Polar Estimation")
    plt.title("X")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(3)
    plt.clf()
    plt.plot(np.diff(ys), 'r', label="Ground truth")
    plt.plot(np.diff(xs_est), 'bo', label="Cart Estimation")
    plt.plot(np.diff(xs_est_polar), 'go', label="Polar Estimation")
    plt.title("Y")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(4)
    plt.clf()
    plt.plot(np.diff(rads), 'r', label="Ground truth")
    plt.plot(np.diff(rads_est), 'bo', label="Cart Estimation")
    plt.title("Yaw")
    plt.legend()
    plt.ion()
    plt.show()

    plt.figure(5)
    plt.clf()
    plt.plot(rads, 'r', label="Ground truth")
    plt.plot(rads_est, 'bo', label="Estimation")
    plt.title("Absolute Yaw")
    plt.legend()
    plt.show()

    writeLandmarks(landmarks)


xs, ys, rads = zip(*car_pos)
xs_est, ys_est, rads_est = zip(*car_pos_estimates_cart_orb)
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

plt.figure(5)
plt.clf()
plt.plot(rads, 'r', label="Ground truth")
plt.plot(rads_est, 'bo', label="Estimation")
plt.title("Absolute Yaw")
plt.legend()


plt.show()
cv2.waitKey(0)
