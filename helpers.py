from matplotlib.pyplot import axis
import numpy as np
import cv2
import scipy.optimize

def projectpoints(K, R, t, Q):
    Q.reshape((Q.shape[1],Q.shape[0]))
    T = np.concatenate((R,t),axis=1)
    P = K @ T
    ppsx = []
    ppsy = []
    for i in range(Q.shape[0]):
        p = Q[i,:]
        projected = P @ p.reshape(4,1)
        if (p == np.array([-0.5, -0.5, -0.5, 1.0])).all():
            print(P)
            print(projected/projected[2])
        ppsx.append(float(projected[0]/projected[2]))
        ppsy.append(float(projected[1]/projected[2]))
    return ppsx,ppsy

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


def estHomographyRANSAC(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher_create(crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    good = matches[:200] # just a filter, not really necessary but should speed up things
    P1 = np.array([[kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1],1] for m in good])
    P2 = np.array([[kp1[m.queryIdx].pt[0],kp1[m.queryIdx].pt[1],1] for m in good])
    threshold = (3.84 * 3**2)**2
    max_inliers = 0
    Hest = None
    best_inlier_points = []
    inlier_indices = []
    niter = 10
    for i in range(niter):
        choice  = np.random.choice(good, 4, replace=False)
        points1 = np.array([[kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1],1] for m in choice])
        points2 = np.array([[kp1[m.queryIdx].pt[0],kp1[m.queryIdx].pt[1],1] for m in choice])
        H, status = cv2.findHomography(points1, points2)
        inliers = 0
        inlier_points = []
        for i,(p1,p2) in enumerate(zip(P1,P2)):
            p1proj = (H @ p1)
            p1proj = p1proj / p1proj[-1]
            p2proj = np.linalg.inv(H) @ p2
            p2proj = p2proj / p2proj[-1]
            d = np.linalg.norm(p2[:2] - p1proj[:2],2)**2 + np.linalg.norm(p1[:2] - p2proj[:2],2)**2
            if d < threshold:
                inliers += 1
                inlier_points.append((p1,p2))
            if inliers > max_inliers:
                max_inliers = inliers
                Hest = H
                best_inlier_points = inlier_points
                inlier_indices.append(i)
    best_inlier_points = np.array(best_inlier_points)
    Hest, status = cv2.findHomography(best_inlier_points[:,1,:], best_inlier_points[:,0,:])
    return Hest