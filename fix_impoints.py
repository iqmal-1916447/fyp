import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import scipy.io as sio

observer_name = "sunhl-1th-"

# ------default setting
folder = "finished/"
label_folder = "APL_Labels/"
crop_folder = "APL_Crops/"
files = folder + "*.jpg"

file_names = glob(files)

orig_num = 17 * 2

for k in range(len(file_names)):
    fname = file_names[k]

    # load the coordinates of labeled points
    mat_name = label_folder + file_names[k]
    p = sio.loadmat(mat_name + ".mat")["p"]

    # get points
    point_num = len(p) // 4
    maxnum = max(point_num, orig_num)
    minnum = min(point_num, orig_num)

    # read image
    crop_name = crop_folder + file_names[k]
    crop = cv2.imread(crop_name)

    # center cropped
    plt.imshow(cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), cmap="gray")
    plt.title("mark" + fname)
    plt.show()

    # mark image
    if point_num > orig_num:
        lines = [point_num]
    else:
        lines = [orig_num]

    for i in range(orig_num):
        # h = impoint;
        if i < minnum:
            x1 = i * 2
            y1 = point_num * 2 + i * 2
            x2 = i * 2 + 1
            y2 = point_num * 2 + i * 2 + 1
            position = [p[x1], p[y1]], [p[x2], p[y2]]
            h = plt.plot(*zip(*position), marker="o", color="r", ls="--")
        else:
            h = plt.plot(marker="o", color="r", ls="--")
        lines.append(h)
        # p(i,:) = wait(h);

    plt.show()

    # wait for enter to continue
    while True:
        w = plt.waitforbuttonpress()
        if w:  # (keyboard press)
            key = plt.gcf().canvas.get_tk_widget().master.focus_get().char
            if key == "\x1b":  # 27 is the escape key
                print("No changes made")
                break  # break out of the while loop
            elif key == "\r":  # 13 is the return key
                print("User pressed the return key. Fixed markers.")
                # read points

                left = np.zeros((orig_num, 2))
                right = np.zeros((orig_num, 2))
                mid = np.zeros((orig_num, 2))
                for i in range(orig_num):
                    position = lines[i].getPosition()
                    left[i, :] = position[0, :]
                    right[i, :] = position[1, :]
                    mid[i, :] = (position[1, :] + position[0, :]) / 2
                    # p(i,:) = wait(h);

                # sort points
                left = left[np.argsort(left[:, 1])]
                right = right[np.argsort(right[:, 1])]

                # combine left and right markers
                p1 = np.hstack((left[:, 0], right[:, 0]))
                p2 = np.hstack((left[:, 1], right[:, 1]))
               
            p1 = np.ravel(p1)
            p2 = np.ravel(p2)
            p = np.column_stack((p1, p2))

            # save points
            sio.savemat(mat_name + ".mat", {"p": p})
            np.savetxt(mat_name + ".csv", p, delimiter=",")

            break
        else:
            # Wait for a different command.
            pass
