from math import factorial
import numpy as np
from scipy import ndimage
import cv2

cell_size = 3  # one cell has a width / height of 4 meters
k = 0.8  # Degredation factor


def get_envelope_high(r):  # range should already be scaled between 0 and 1
    r_scaled = r * 2  # to have a good exponential effect envelope is between 0 and 2

    return (np.exp(-r_scaled) - np.exp(-2)) / (np.exp(0) - np.exp(-2)) * 0.5 + 0.5


def get_envelope_low(r):  # range should already be scaled between 0 and 1
    r_scaled = r * 2

    return (-np.exp(-r_scaled) + np.exp(-0)) / (-np.exp(-2) + np.exp(0)) * 0.5


# Implements a correct probalistic model
# -> in order to punish high r values, create a function that converges to 0.5 so that
#    log likelihoods are not impacted (log(0.5/(1 - 0.5)) = 0). P = 0.5 equals no information
def plausibility_range(rcs, tmin, tmax):
    # Scales a RCS value between the desired values
    # Desired values depend on the range and the envelope functions, for larger ranges the envelopes converge to 0.5 meaning that the measurement
    # will not have any impact on the probablity calculation

    rmin = 0  # rcs values are between 0 and 1
    rmax = 1
    return (rcs - rmin) / (rmax - rmin) * (tmax - tmin) + tmin


def cart_img_point_to_world_idx_point(x_idx, y_idx, gridmap, cell_size, car_x, car_y, car_yaw):
    rot = car_yaw
    rotMat = np.array([[np.cos(rot), - np.sin(rot)],
                       [np.sin(rot), np.cos(rot)]])

    x_local = (x_idx - 250) / 4  # position in meters in local frame
    y_local = (y_idx - 250) / 4

    pos_world = np.matmul(rotMat, np.array([x_local, y_local]).transpose()) + np.array([car_x, car_y]).transpose()

    pos_world_pixels = np.divide(pos_world, cell_size) + np.array([gridmap.shape[0] / 2, gridmap.shape[1] / 2]).transpose()

    return pos_world_pixels[0], pos_world_pixels[1]


def rotated_cart_img_point_to_world_idx_point(x_idx, y_idx, gridmap, cell_size, car_x, car_y, car_yaw):
    x_world = (x_idx - 250) * 0.25 + car_x
    y_world = (y_idx - 250) * 0.25 * (-1) + car_y

    pos_world_pixels = np.divide(np.array([x_world, y_world]).transpose(), cell_size) + np.array([gridmap.shape[0] / 2, gridmap.shape[1] / 2]).transpose()

    return pos_world_pixels[0], pos_world_pixels[1]


def updateGridMap(gridmap, car_x, car_y, car_yaw, cart_img, inv_sensor_model_mask):
    cart_img_masked = cart_img * inv_sensor_model_mask
    cart_img_rotated = ndimage.rotate(cart_img_masked, np.rad2deg(car_yaw), reshape=False)
    print(car_yaw)
    cv2.imshow("Cart img rotated", cart_img_rotated)
    cv2.waitKey(1)

    # rot = np.deg2rad(car_yaw)
    rot = car_yaw

    idx_x_00, idx_y_00 = cart_img_point_to_world_idx_point(0, 0, gridmap, cell_size, car_x, car_y, car_yaw)
    idx_x_11, idx_y_11 = cart_img_point_to_world_idx_point(cart_img.shape[0], cart_img.shape[1], gridmap, cell_size, car_x, car_y, car_yaw)
    idx_x_01, idx_y_01 = cart_img_point_to_world_idx_point(0, cart_img.shape[1], gridmap, cell_size, car_x, car_y, car_yaw)
    idx_x_10, idx_y_10 = cart_img_point_to_world_idx_point(cart_img.shape[0], 0, gridmap, cell_size, car_x, car_y, car_yaw)

    idx_x_low = np.floor(np.min([idx_x_00, idx_x_11, idx_x_10, idx_x_01])).astype(int)
    idx_y_low = np.floor(np.min([idx_y_00, idx_y_11, idx_y_10, idx_y_01])).astype(int)
    idx_x_high = np.ceil(np.max([idx_x_00, idx_x_11, idx_x_10, idx_x_01])).astype(int)
    idx_y_high = np.ceil(np.max([idx_y_00, idx_y_11, idx_y_10, idx_y_01])).astype(int)

    for i in range(idx_x_low, np.min([idx_x_high, gridmap.shape[0]])):
        for j in range(idx_y_low, np.min([idx_y_high, gridmap.shape[1]])):
            # Find RCS value for this point
            x_pos_world_cell_center = (i - gridmap.shape[0] / 2) * cell_size
            y_pos_world_cell_center = (j - gridmap.shape[1] / 2) * cell_size
            x_pos_local = x_pos_world_cell_center - car_x  # meters
            y_pos_local = y_pos_world_cell_center - car_y

            x_pos_local_boundaries = [x_pos_local - cell_size / 2, x_pos_local + cell_size / 2]
            x_pos_local_boundaries_idx = np.round(np.multiply(x_pos_local_boundaries, 4)).astype(int)
            y_pos_local_boundaries = [y_pos_local - cell_size / 2, y_pos_local + cell_size / 2]
            y_pos_local_boundaries_idx = np.round(np.multiply(y_pos_local_boundaries, 4)).astype(int)
            # compensate that origin is in the middle of the carthesian frame
            x_pos_local_boundaries_idx += 250
            y_pos_local_boundaries_idx += 250

            if x_pos_local_boundaries_idx[0] < 0 or y_pos_local_boundaries_idx[0] < 0:
                continue

            # Select subset from carth_img within the boundaries
            subset = cart_img_rotated[x_pos_local_boundaries_idx[0]:x_pos_local_boundaries_idx[1],
                                      y_pos_local_boundaries_idx[0]: y_pos_local_boundaries_idx[1]]

            if subset.size == 0:
                continue  # This part of the occupancy map is not in the current radar frame
            # if subset.size > 4:
                # print("found subset in cart_img bigger then 4, actual size is " + str(subset.size))
            percentile = np.percentile(subset, 20)
            subset = subset[subset > percentile]

            del_subset = np.delete(subset, np.where(subset == 0))
            if del_subset.size == 0:
                continue  # Th

            rcs = np.average(del_subset)

            # ### inverse sensor model ###
            x_idx = np.average(x_pos_local_boundaries_idx)
            y_idx = np.average(y_pos_local_boundaries_idx)
            r = np.sqrt((x_idx - 250) ** 2 + (y_idx - 250) ** 2) / 4  # Length in pixels
            # plausibility_range = np.exp(-0.0001 * (r ** 2))
            # # plausibility_range = 1

            # TODO test with envelope function (envelope function still has to modified such that it starts within range 0.4-0.6)
            # otherwise probabilities are influenced too fast
            # r_scaled = r / 62.5
            # envelope_min = get_envelope_low(r_scaled)
            # envelope_max = get_envelope_high(r_scaled)
            # p = plausibility_range(rcs, envelope_min, envelope_max)

            p = (rcs - 0) / (1 - 0) * (0.9 - 0.45) + 0.45

            gridmap[i, j] = gridmap[i, j] + np.log(p / (1 - p))


def convertToProbabilities(gridmap):
    pgridmap = 1 - np.divide(np.full(gridmap.shape, 1), 1 + np.exp(gridmap))
    return pgridmap
