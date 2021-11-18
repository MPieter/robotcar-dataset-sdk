import numpy.linalg
import numpy as np


def update_state(state, ut, dt):
    vt = ut[0]
    wt = ut[1]

    # Only need to update the first 3 elements: x, y, thetha
    updated_state = np.copy(state)

    dx = -vt / wt * np.sin(state[2]) + vt / wt * np.sin(state[2] + wt * dt)
    dy = vt / wt * np.cos(state[2]) - vt / wt * np.cos(state[2] + wt * dt)
    dthetha = wt * dt

    updated_state[0] = state[0] + dx
    updated_state[1] = state[1] + dy
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


def ekf(state, cov, ut, zt, dt, Rtx, Q):
    # prediction
    updated_state = update_state(state, ut, dt)
    updated_cov = update_cov(state, cov, ut, dt, Rtx)

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
        updated_state = update_state + Ki @ (obs - expected_obs)
        updated_cov = (np.identity(updated_cov.shape[0]) - Ki @ Hi) @ updated_cov
    return updated_state, updated_cov


# EKF Slam Parameters
N_LANDMARKS = 100

# TODO noise on velocity measurements
Rtx = np.zeros((3, 3))
Rtx[0, 0] = 1
Rtx[1, 1] = 1
Rtx[2, 2] = 1  # TODO is this matrix diagonal?

varR = 1  # Variance of range measurements of landmarks
varThetha = 1  # Variance of thetha measurements of landmarks
Q = np.diag([varR, varThetha])
# End Parameters

u0 = np.zeros((1, 2 * N_LANDMARKS + 3))
sigma0 = np.zeros((2 * N_LANDMARKS + 3, 2 * N_LANDMARKS + 3))

for i in range(3, 2 * N_LANDMARKS + 3):
    sigma0[i, i] = 3.4028235e+38
