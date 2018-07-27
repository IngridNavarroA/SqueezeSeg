from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import pdb

from datetime import datetime
import sys
import time
import glob
import pdb    

import numpy as np
from six.moves import xrange
import tensorflow as tf
from PIL import Image

from config import *
from imdb import kitti
from utils.util import *
from nets import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Image_res_y = 63
Image_res_x = 511

v_fov = (-22, 2) #vertical field of view -15to15 for vlp-16
v_fov_total = -v_fov[0] + v_fov[1]

v_res = 0.42 # in degrees for vlp=64
h_res = 0.35 # in degrees for vlp-16 (0.1-0.4)

v_res_rad = v_res*(np.pi/180)
h_res_rad = h_res*(np.pi/180)

x_min = -360.0 / h_res /2
x_max = 360.0 / h_res

y_min = v_fov[0] / v_res
y_max = v_fov_total / v_res

max_coord_x = []
max_coord_y = []
min_coord_x = []
min_coord_y = []
x_img = []
y_img = []

def _normalize(x):
    return (x - x.min())/(x.max() - x.min())

for root, dirs, filenames in os.walk("/home/inavarro/Desktop/workspace/SqueezeSeg/src/kit/"):
    for f in filenames:
		file_n = f.split('.bin')[0]
		print(file_n)
		coords = np.fromfile(os.path.join('/home/inavarro/Desktop/workspace/SqueezeSeg/src/kit/',f), dtype=np.float32).reshape(-1, 4)
	
		x = coords[:, 0]
		y = coords[:, 1]
		z = coords[:, 2]
		r = coords[:, 3]
		
		x_norm = _normalize(x)
		y_norm = _normalize(y)
		z_norm = _normalize(z)
		
		za = np.zeros((x.shape[0], 2))
		
		counter = 0
		for i in range(x.shape[0]):
			i -= 1 
			distance = np.sqrt((x_norm[i+1]-x_norm[i])**2 + (y_norm[i+1]-y_norm[i])**2 + (z_norm[i+1]-z_norm[i])**2)
			print(distance)			
			if distance < 1:
				counter += 1
#				print (counter)
			za[i, 0] = np.arcsin((z_norm[i+1]-z_norm[i]) / distance) #zenith 
			za[i, 1] = 180 + np.arctan2((x_norm[i+1]-x_norm[i]), (y_norm[i+1]-y_norm[i])) * 180 / np.pi #azimuth 
		print(counter)
		#zenith = np.arcsin(z_norm / np.sqrt(x_norm**2+ y_norm**2+ z_norm**2)) * 180 / np.pi
		
#		pairs = np.column_stack((zenith, azimuth))
#		pairs = pairs[np.where(pairs[:, 1] <= 90)]
#		print(pairs.shape)

#		azimuth = np.arcsin(y_norm / np.sqrt(x_norm**2 + y_norm**2)) * 180 / np.pi
#		count = 0		
#		for i in range(x.shape[0]):
#			if x[i] >= 0 and y[i] >= 0:
#				count += 1
#		print(count)

#		print(max(azimuth), min(azimuth))

		# coords = np.fromfile(os.path.join('/home/inavarro/Desktop/workspace/SqueezeSeg/src/kit/',f), dtype=np.float32).reshape(-1, 4)
	     

#		 coord = np.vstack(coords[i] for i in range(coords.shape[0]))
#        #np.save('/home/pratibha/dl_semseg/SQSG_ROOT/src/kitti_3d_bin/1.pcd', coord)
#        #pdb.set_trace()
#        coord_x =  (np.arctan(-coord[:,1]/coord[:,0]))/ h_res_rad
#        coord_y = (np.arctan(coord[:,2]/(np.sqrt(coord[:,0]**2+coord[:,1]**2))))/ v_res_rad
#        
#        #coord_x =  (np.arctan2(coord[:,0],coord[:,2]))
#        #coord_y = (np.arcsin(coord[:,1]/(np.sqrt(coord[:,0]**2+coord[:,1]**2+coord[:,2]**2))))
#        
#        
#        coord_x -= x_min/2
#        coord_y -= y_min
#        
#        max_coord_x.append(max(coord_x))
#        max_coord_y.append(max(coord_y))
#        min_coord_x.append(min(coord_x))
#        min_coord_y.append(min(coord_y))
#        
#        diff_x = max(coord_x)-min(coord_x)
#        diff_y = max(coord_y)-min(coord_y)
#        
#        coord_range = np.sqrt(coord[:,0]**2+coord[:,1]**2+coord[:,2]**2)
#        coord = np.hstack((coord,coord_range[:,np.newaxis]))
#        
#        coord_new = np.zeros([Image_res_y+1, Image_res_x+1, 5])
#        pdb.set_trace()
#        
#        for i in range(coord.shape[0]):
#            y_img.append(int((coord_y[i]/(y_max))*Image_res_y))
#            x_img.append(int(math.floor((coord_x[i]/(0.5*x_max))*Image_res_x)))
#            #print(int(math.floor((coord_x[i]/(x_max/Image_res_x)))), int(math.floor((coord_x[i]/(x_max))*Image_res_x)))
#            #coord_new[int((coord_zenith[i]/del_zenith)*range_azmth), int((coord_azmth[i]/del_azmth)*range_zenith), 0] = coord[i][0]
#            coord_new[int((coord_y[i]/(diff_y))*Image_res_y), int(math.floor((coord_x[i]/(diff_x))*Image_res_x)), 0] = coord[i][0]

#            #coord_new[int((coord_zenith[i]/del_zenith)*range_azmth), int((coord_azmth[i]/del_azmth)*range_zenith), 1] = coord[i][1]
#            coord_new[int((coord_y[i]/(diff_y))*Image_res_y), int(math.floor((coord_x[i]/(diff_x))*Image_res_x)), 1] = coord[i][2]

#            #coord_new[int((coord_zenith[i]/del_zenith)*range_azmth), int((coord_azmth[i]/del_azmth)*range_zenith), 2] = coord[i][2]
#            coord_new[int((coord_y[i]/(diff_y))*Image_res_y), int(math.floor((coord_x[i]/(diff_x))*Image_res_x)), 2] = coord[i][1]

#            #coord_new[int((coord_zenith[i]/del_zenith)*range_azmth), int((coord_azmth[i]/del_azmth)*range_zenith), 3] = coord[i][3]
#            coord_new[int((coord_y[i]/(diff_y))*Image_res_y), int(math.floor((coord_x[i]/(diff_x))*Image_res_x)), 3] = coord[i][3]

#            #coord_new[int((coord_zenith[i]/del_zenith)*range_azmth), int((coord_azmth[i]/del_azmth)*range_zenith), 4] = coord[i][4]
#            coord_new[int((coord_y[i]/(diff_y))*Image_res_y), int(math.floor((coord_x[i]/(diff_x))*Image_res_x)), 4] = coord[i][4]

#        np.save(os.path.join('/home/inavarro/Desktop/workspace/SqueezeSeg/src/kit/', (file_n + ".npy")), np.flipud(coord_new))




#lidar = np.load('/home/inavarro/Desktop/workspace/SqueezeSeg/src/kit/').astype(np.float32, copy=False)
##img = Image.fromarray(lidar[:,:,:3]) # taking all x,y,z
#depth_img = Image.fromarray((255 * _normalize(lidar[:, :, 3])).astype(np.uint8)) #taking just the z
#plt.imshow(depth_img)
#depth_img.save('/home/inavarro/Desktop/workspace/SqueezeSeg/src/kit/dep_img1025range2.png')
