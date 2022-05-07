import numpy as np
import cv2
import os

row = 67
col = 67
path = (r'G:\dataset_5_02\shenzhen\GF1D_PMS_E114.3_N22.4_20210508_L1A1256908821\output')

L = []
for root, dirs, files in os.walk(path):  
    for file in files:  
        if os.path.splitext(file)[1] == '.tif':
            L.append(os.path.join(root, file)) 
L.sort() 
print(L)
imgnum = len(L)

imglist = L

imgarray = []
for i in range(imgnum):
    imgname = imglist[i]
    img = cv2.imread(imgname)
    imgarray.append(img)

joint_img = np.zeros([512*(row+1),512*(col+1),3],dtype=np.uint8)
for ii in range(row):
    for jj in range(col):
        joint_img[0+512*(ii-0):0+512*(ii-0)+512,0+512*(jj-0):0+512*(jj-0)+512,:] = imgarray[jj  + (col) * (ii - 0)]
        #print(jj  + (col) * (ii - 0))
cv2.imwrite(path + '\\' + 'joint.tif', joint_img)

print('prod by gz')
