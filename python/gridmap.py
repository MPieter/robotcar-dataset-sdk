import numpy as np
from scipy import ndimage
import cv2


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


def updateGridMap(gridmap, cell_size, k, car_x, car_y, car_yaw, cart_img):
    cart_img_rotated = ndimage.rotate(cart_img, np.rad2deg(car_yaw), reshape=False)
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
            percentile = np.percentile(subset, 80)
            subset = subset[subset > percentile]

            del_subset = np.delete(subset, np.where(subset == 0))
            if del_subset.size == 0:
                continue  # Th

            rcs = np.average(del_subset)

            # Scale the detection probability between 0.5 and 1
            p = 0.5 + 0.5 * rcs

            gridmap[i, j] = gridmap[i, j] * k + np.log(p / (1 - p))


def convertToProbabilities(gridmap):
    pgridmap = 1 - np.divide(np.full(gridmap.shape, 1), 1 + np.exp(gridmap))
    return pgridmap
