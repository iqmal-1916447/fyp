import numpy as np
import cv2
import os
from math import acos
from decimal import Decimal

ap_num = 32
lat_num = 32

# get labels
folder_l = 'APL_Labels/'
# folder_l = 'C:\Users\zhenxt\Documents\Python Scripts\Spine\labels\training\';
files_l = os.path.join(folder_l, '*.mat')

fileNames_l = []
for file_l in os.listdir(folder_l):
    if file_l.endswith('.mat'):
        fileNames_l.append(file_l)

N = len(fileNames_l)

# get image
folder_im = 'APL_Crops/'
# folder_im = 'C:\Users\zhenxt\Documents\Python Scripts\Spine\data\training\';

fileNames_im = []
for file_name in fileNames_l:
    fileNames_im.append(file_name[:-4])

CobbAn_ap = []
CobbAn_lat = []
landmarks_ap = []
landmarks_lat = []

for k in range(N):
    # get images
    l = folder_im + fileNames_im[k] + ".jpg"
    im = cv2.imread(l)
    H, W, _ = im.shape

    # get labels
    l_n = folder_l + fileNames_l[k]
    p = loadmat(l_n)
    coord = p['p2']

    if 'lateral' not in l_n.lower():
        p2 = np.concatenate((coord[:ap_num], coord[ap_num:ap_num*2]), axis=0)
        vnum = int(ap_num / 4)
        landmarks_ap = np.concatenate((landmarks_ap, coord[:ap_num]/W, coord[ap_num:ap_num*2]/H), axis=0) # scale landmark coordinates
    else:
        p2 = np.concatenate((coord[:lat_num], coord[lat_num:lat_num*2]), axis=0)
        vnum = int(lat_num / 4)
        landmarks_lat = np.concatenate((landmarks_lat, coord[:lat_num]/W, coord[lat_num:lat_num*2]/H), axis=0) # scale landmark coordinates
    
    cob_angles = np.zeros(3)
    
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('GroundTruth')
    plt.hold(True)
    
    mid_p_v = np.zeros((int(p2.shape[0]/2), 2))
    for n in range(int(p2.shape[0]/2)):
        mid_p_v[n,:] = (p2[n*2,:] + p2[(n-1)*2+1,:])/2
    
    # calculate the middle vectors & plot the labeling lines
    mid_p = np.zeros((int(p2.shape[0]/2), 2))
    for n in range(int(p2.shape[0]/4)):
        mid_p[(n-1)*2+1,:] = (p2[n*4-1,:] + p2[(n-1)*4+1,:])/2
        mid_p[n*2,:] = (p2[n*4,:] + p2[(n-1)*4+2,:])/2
    
    # plot the midpoints
    plt.plot(mid_p[:,0], mid_p[:,1], 'y.', markersize=20)
    
    vec_m = np.zeros((int(mid_p.shape[0]/2),2))
    for n in range(int(mid_p.shape[0]/2)):
        vec_m[n,:] = mid_p[n*2,:] - mid_p[(n-1)*2+1,:]
        #plot the midlines
        ax.plot([mid_p[n*2,0],mid_p[(n-1)*2+1,0]],
                [mid_p[n*2,1],mid_p[(n-1)*2+1,1]],color='r',linewidth=2)
    
    mod_v = np.power(np.sum(vec_m * vec_m, axis=1), 0.5)
    dot_v = vec_m.dot(vec_m.T)
    
    #calculate the Cobb angle
    angles = np.arccos(np.round(dot_v/(mod_v.reshape(-1,1) * mod_v),8))
    maxt, pos1 = np.max(angles, axis=1), np.argmax(angles, axis=1)
    pt, pos2 = np.max(maxt), np.argmax(maxt)
    pt = pt/np.pi*180
    cob_angles[0] = pt
    
    #plot the selected lines
    ax.plot([mid_p[pos2*2,0],mid_p[(pos2-1)*2+1,0]],
            [mid_p[pos2*2,1],mid_p[(pos2-1)*2+1,1]],color='g',linewidth=2)
    ax.plot([mid_p[pos1[pos2]*2,0],mid_p[(pos1[pos2]-1)*2+1,0]],
            [mid_p[pos1[pos2]*2,1],mid_p[(pos1[pos2]-1)*2+1,1]],color='g',linewidth=2)
    
    if not isS(mid_p_v): # 'S'
        mod_v1 = np.power(np.sum(vec_m[0,:] * vec_m[0,:]), 0.5)
        mod_vs1 = np.power(np.sum(vec_m[pos2,:] * vec_m[pos2,:]), 0.5)
        mod_v2 = np.power(np.sum(vec_m[vnum,:] * vec_m[vnum,:]), 0.5)
        mod_vs2 = np.power(np.sum(vec_m[pos1[pos2],:] * vec_m[pos1[pos2],:]), 0.5)
        
        dot_v1 = vec_m[0,:] * vec_m[pos2,:].T
        dot_v2 = vec_m[vnum,:] * vec_m[pos1[pos2],:].T
        
        mt = np.arccos(np.round(dot_v1/(mod_v1 * mod_vs1), 8))
        tl = np.arccos(np.round(dot_v2/(mod_v2 * mod_vs2), 8))
        
        mt = mt/np.pi*180
        cob_angles[1] = mt
        tl = tl/np.pi*180
        cob_angles[2] = tl
        
    else:
        # max angle in the upper part
        if (mid_p_v[pos2*2,1] + mid_p_v[pos1[pos2]*2,1]) < im.shape[0]:
            #calculate the Cobb angle (upside)
            mod_v_p = np.power(np.sum(vec_m[pos2,:] * vec_m[pos2,:]), 0.5)
            mod_v1 = np.power(np.sum(vec_m[0:pos2,:] * vec_m[0:pos2,:]), 0.5)
            dot_v1 = vec_m[pos2,:] * vec_m[0:pos2,:] * vec_m[0:pos2,:].T

		angles1 = np.arccos(np.round(dot_v1./(mod_v_p * mod_v), -8))
		CobbAn1, pos1_1 = np.max(angles1), np.argmax(angles1)
		mt = CobbAn1/np.pi*180
		cob_angles[2] = mt

		plt.plot([mid_p[pos1_1*2, 0], mid_p[(pos1_1-1)*2+1, 0]], [mid_p[pos1_1*2, 1], mid_p[(pos1_1-1)*2+1, 1]], 'g', linewidth=2)


    mod_v_p2 = np.power(np.sum(vec_m[pos1_1-1,:] * vec_m[pos1_1-1,:]), 0.5)
    mod_v2 = np.power(np.sum(vec_m[0:pos1_1-1,:] * vec_m[0:pos1_1-1,:]), 0.5)
    dot_v2 = vec_m[pos1_1-1,:] * vec_m[0:pos1_1-1,:].T
    
    angles2 = np.arccos(np.round(dot_v2/(mod_v_p2 * mod_v2), 8))
    CobbAn2, pos1_2 = np.max(angles2), np.argmax(angles2)
    tl = CobbAn2/np.pi*180
    cob_angles[2] = mt
    
    # pos1_2 = pos1_2 + pos1(pos2) - 1
    plt.plot([mid_p[pos1_2*2, 0], mid_p[(pos1_2-1)*2+1, 0]],
             [mid_p[pos1_2*2, 1], mid_p[(pos1_2-1)*2+1, 1]], 'g', linewidth=2)
# pop up a text window
# pause(1)
output = f"{k}: the Cobb Angles(PT, MT, TL/L) are {pt}, {mt} and {tl}, and the two most tilted vertebrae are {pos2} and {pos1[pos2]}.\n"
# h = msgbox(output);
print(output)
#         fprintf('No. %d :The Cobb Angles(PT, MT, TL/L) are %3.1f, and the two most tilted vertebrae are %d and %d. ',...
#             k,CobbAn,pos2,pos1(pos2));

# pause(2)
plt.close()
if "lateral" not in fileNames[k].lower():
    CobbAn_ap.append(cob_angles)  # cobb angles
else:
    CobbAn_lat.append(cob_angles)  # cobb angles

#write to csv file
np.savetxt("angles_ap.csv", CobbAn_ap, delimiter=",")
np.savetxt("angles_lat.csv", CobbAn_lat, delimiter=",")
np.savetxt("landmarks_ap.csv", landmarks_ap, delimiter=",")
np.savetxt("landmarks_lat.csv", landmarks_lat, delimiter=",")
with open("filenames_aplat.csv", "w") as f:
for fn in fileNames_im:
f.write(f"{fn}\n")

