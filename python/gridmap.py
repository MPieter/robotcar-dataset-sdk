import numpy as np


def updateGridMap(gridmap, cell_size, k, car_x, car_y, car_yaw, cart_img):
    rot = np.deg2rad(car_yaw)
    for i in range(gridmap.shape[0]):
        for j in range(gridmap.shape[1]):
            # Find RCS value for this point
            x_pos_world_cell_center = (i - gridmap.shape[0] / 2) * cell_size
            y_pos_world_cell_center = (j - gridmap.shape[1] / 2) * cell_size
            # x_pos_local = x_pos_world_cell_center - car_x  # TODO take into account rotation of vehicle
            # y_pos_local = y_pos_world_cell_center - car_y

            rotMat = np.array([[np.cos(rot), - np.sin(rot)],
                               [np.sin(rot), np.cos(rot)]])

            # pos_local_rotated = np.matmul(rotMat, np.array([x_pos_local, y_pos_local]).transpose())
            pos_local_rotated = np.matmul(rotMat, np.array([x_pos_world_cell_center, y_pos_world_cell_center]).transpose()) - np.array([car_x, car_y]).transpose()
            x_pos_local = pos_local_rotated[0]
            y_pos_local = pos_local_rotated[1]

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
            subset = cart_img[x_pos_local_boundaries_idx[0]:x_pos_local_boundaries_idx[1],
                              y_pos_local_boundaries_idx[0]: y_pos_local_boundaries_idx[1]]

            if subset.size == 0:
                continue  # This part of the occupancy map is not in the current radar frame
            # if subset.size > 4:
                # print("found subset in cart_img bigger then 4, actual size is " + str(subset.size))
            percentile = np.percentile(subset, 80)
            subset = subset[subset > percentile]

            rcs = np.average(subset)

            # Scale the detection probability between 0.5 and 1
            p = 0.5 + 0.5 * rcs

            gridmap[i, j] = gridmap[i, j] * k + np.log(p / (1 - p))


def convertToProbabilities(gridmap):
    pgridmap = 1 - np.divide(np.full(gridmap.shape, 1), 1 + np.exp(gridmap))
    return pgridmap
