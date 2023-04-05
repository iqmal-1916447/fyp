import os
import cv2

observer_name = 'sunhl-1th-'
point_num = 0
folder = 'images/'
labelFolder = 'SupLabels/'
cropFolder = 'SupCrops/'
files = os.path.join(folder, '*.jpg')

fileNames = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])

for k, fileName in enumerate(fileNames):
    l = os.path.join(folder, fileName)
    im = cv2.imread(l)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(fileName)
    cv2.imshow(fileName, cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    p = None
    if point_num > 0:
        p = []
        for i in range(point_num):
            p.append(cv2.ginput(1, timeout=0)[0])

    # crop images
    r = cv2.selectROI(fileName, im, False)
    crop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    crop_name = os.path.join(cropFolder, observer_name + fileName)
    cv2.imwrite(crop_name, crop)
    os.rename(l, os.path.join('finished', fileName))
