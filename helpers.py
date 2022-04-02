from matplotlib.pyplot import axis
import numpy as np
import cv2
import scipy.optimize


def cross_op(p):
    if p.shape != (3, 1):
        p = p.reshape(3, 1)
    x, y, z = p[0][0], p[1][0], p[2][0]
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def checkerboard_points(n, m):
    ps = np.zeros((3, n, m))
    for i in range(n):
        for j in range(m):
            p = [i - (n-1)/2, j - (m-1)/2, 0]
            ps[:, i, j] = p
    return ps


def normalize2d(x):
    return (x - x.mean()) / x.std()


def estimateHomographies(Q_omega, qs):
    if all([Q_omega.shape[0] != x.shape[0] for x in qs]):
        raise RuntimeError("Shape of Q_omega and must match: (n,3)")
    Hs = []
    for q in qs:
        h, status = cv2.findHomography(Q_omega[:, :2], shom(q)[:, :2])
        Hs.append(h)
    return Hs


def hest(q1, q2):
    # TODO: doens't work for now! :)
    # i have to store there normalization
    # params for later use of homography!
    q1[:, :2] = normalize2d(q1[:, :2])
    q2[:, :2] = normalize2d(q2[:, :2])
    Bs = []
    for p, q in zip(q1, q2):
        B = np.kron(p, np.cross(q, np.identity(q.shape[0]) * -1))
        Bs.append(B)
    U, S, V = np.linalg.svd(np.concatenate(Bs, axis=0))
    He = V[-1].reshape(3, 3).round(3).T
    return He


def shom(m):
    if m.shape == (3,):
        return m/m[2]

    if m.shape == (4,):
        return m/m[3]

    return m/np.stack((m[:, 2], m[:, 2], m[:, 2])).T


def make_camera(f, deltax, deltay, R, t):
    K = np.array([
        [f, 0, deltax],
        [0, f, deltay],
        [0, 0,      1]
    ])
    T = np.concatenate((R, t), axis=1)
    P = K@T
    return (K, T, P)


def triangulate(Q, P):
    B = []
    for i in range(len(P)):
        B.append(P[i][2]*Q[i][0] - P[i][0])
        B.append(P[i][2]*Q[i][1] - P[i][1])
    u, s, v = np.linalg.svd(B)
    return v[-1]/v[-1][-1]


def triangulate_nonlin(Q, P):
    x0 = triangulate(Q, P)

    def compute_residuals(Qw):
        R = []
        for i in range(len(P)):
            R.append((shom(P[i] @ Qw) - Q[i])[:2])
        R = np.array(R)
        R = R.flatten()
        return R

    return scipy.optimize.least_squares(compute_residuals, x0).x


def compute_essential(t2, R2):
    return cross_op(t2) @ R2


def compute_fundamental(t2, R2, K2, K1):
    E = compute_essential(t2, R2)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    F / np.linalg.norm(F)
    return F


def Fest_8point(q1, q2):
    Bs = []
    for i, (p, q) in enumerate(zip(q1, q2)):
        B = np.array([
            p[0]*q[0], p[0]*q[1], p[0], p[1] *
            q[0], p[1]*q[1], p[1], q[0], q[1], 1
        ])
        Bs.append(B)
    U, S, V = np.linalg.svd(np.array(Bs))
    Fe = V[-1].reshape(3, 3).T * -1  # why -1?
    return Fe
