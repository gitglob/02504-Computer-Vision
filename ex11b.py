# %%
import numpy as np
from helpers import *
import cv2
import matplotlib.pyplot as plt
import glob

imgs = []
for imf in glob.glob('data/sequence/*.png'):
    im = cv2.imread(imf)
    imgs.append(im)
K = np.loadtxt('data/K.txt')

# %%

tt = np.zeros((3, 1))
tsss = []
init = False
cur_t = np.zeros((3, 1))
tsss.append(cur_t.copy())
cur_R = np.eye(3)
bf = cv2.BFMatcher()

for i in range(2, len(imgs)):
    im1 = imgs[i-2]
    im2 = imgs[i-1]
    im3 = imgs[i]

    orb = cv2.ORB_create(nfeatures=5000)
    kp0, des0 = orb.detectAndCompute(im1, None)
    kp0 = np.array([k.pt for k in kp0])
    kp1, des1 = orb.detectAndCompute(im2, None)
    kp1 = np.array([k.pt for k in kp1])
    # kp2, des2 = orb.detectAndCompute(im3, None)
    # kp2 = np.array([k.pt for k in kp2])

    matches01 = bf.match(des0, des1)
    matches01 = sorted(matches01, key=lambda x: x.distance)[:2000]
    matches01 = np.array([(m.queryIdx, m.trainIdx) for m in matches01])
    # matches12 = bf.match(des1, des2)
    # matches12 = sorted(matches12, key=lambda x: x.distance)[:2000]
    # matches12 = np.array([(m.queryIdx, m.trainIdx) for m in matches12])

    points1 = kp0[matches01[:, 0]]
    points2 = kp1[matches01[:, 1]]
    E, mask = cv2.findEssentialMat(
        points2, points1, K, method=cv2.RANSAC, prob=0.999)
    retval, R, t, mask = cv2.recoverPose(E, points2, points1, K)

    cur_t = cur_t - R.T.dot(t)  # + cur_R.dot(t)
    # cur_R = cur_R.dot(R)
    tsss.append(cur_t.copy())
    print(retval, cur_t)
    
    continue

    # remove the matches that are not in both images
    matches01 = matches01[(mask == 255).reshape(-1)]
    P1 = K @ np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    P2 = K @ np.concatenate((R, t), axis=1)

    _, idx01, idx12 = np.intersect1d(
        matches01[:, 1], matches12[:, 0], return_indices=True)
    fmatches01 = matches01[idx01]
    fmatches12 = matches12[idx12]

    fpoints1 = kp0[fmatches01[:, 0]]
    fpoints2 = kp1[fmatches01[:, 1]]
    fpoints3 = kp2[fmatches12[:, 1]]

    triang = cv2.triangulatePoints(P1, P2, fpoints1.T, fpoints2.T)
    fig = plt.figure(i)
    ax = fig.add_subplot(projection='3d')
    x = triang[2,:]
    y = triang[0,:]
    z = triang[1,:]
    ax.scatter(x, y, z)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        triang[:3, :].T, fpoints3, K, None)
    rot, _ = cv2.Rodrigues(rvec)
    cur_t = cur_t -rot.T.dot(tvec)
    print(cur_t)

    continue

    # img3 = cv2.drawMatches(im1,kp0_raw,im2,kp1_raw,matches01_raw[:2000],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.figure(i, figsize = (20,20))
    # plt.imshow(img3)

    init = True

    triang = cv2.triangulatePoints(P1, P2, fpoints1.T, fpoints2.T)
    # triang = (triang / triang[3, :])
    # print(triang.shape)

    # fig = plt.figure(i)
    # ax = fig.add_subplot(projection='3d')
    # x = triang[2,:]
    # y = triang[0,:]
    # z = triang[1,:]
    # ax.scatter(x, y, z)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        triang[:3, :].T, fpoints3, K, None)
    print(retval)
    if not retval or i > 15:
        print("NOOOO")
        break

    rot, _ = cv2.Rodrigues(rvec)
    tvec = -rot.T.dot(tvec)
    tt += tvec
    tsss.append(tt.copy())
    print(tt, i)
    break


# fig = plt.figure(figsize=(20, 20))
# ax = fig.add_subplot(projection='3d')
# ax.axes.set_xlim3d(left=-5, right=5)
# ax.axes.set_ylim3d(bottom=-5, top=5)
# ax.axes.set_zlim3d(bottom=-5, top=5)
x = np.array(tsss).reshape(-1, 3)[:, 0]
y = np.array(tsss).reshape(-1, 3)[:, 1]
z = np.array(tsss).reshape(-1, 3)[:, 2]
# ax.plot(x, y, z, c='g', alpha=1)
# tsss

#%%

plt.figure(1, figsize=(20, 20))
plt.ylim(-3, 2)
plt.xlim(-1, 13)
plt.plot(x, z, linewidth=5, c='r')
