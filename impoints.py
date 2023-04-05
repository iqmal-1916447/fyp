import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

observer_name = "sunhl-1th-"

# ------default setting
point_num = 0
folder = "images/"
label_folder = "SupLabels/"
crop_folder = "SupCrops/"
files = folder + "*.jpg"

file_names = glob(files)

for k in range(len(file_names)):

    l = folder + file_names[k]
    im = cv2.imread(l)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.equalizeHist(image), cmap="gray")
    plt.title(file_names[k])
    plt.show()

    # get points
    p = np.zeros((point_num, 2))
    for i in range(point_num):
        h = plt.ginput(1)
        p[i, :] = h[0]

    # crop images
    rect_pos = plt.ginput(2)
    pos = [
        int(rect_pos[0][0]),
        int(rect_pos[0][1]),
        int(rect_pos[1][0] - rect_pos[0][0]),
        int(rect_pos[1][1] - rect_pos[0][1]),
    ]

    # save the cropped images
    crop_name = crop_folder + observer_name + str(k) + "-" + file_names[k]
    crop = im[pos[1] : pos[1] + pos[3], pos[0] : pos[0] + pos[2]]
    cv2.imwrite(crop_name, crop)

    os.rename(l, "finished/" + l[8:])
