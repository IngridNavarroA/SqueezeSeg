#!/usr/bin/python2
"""
author: Ingrid Navarro 
date: July 11th, 2018

	SqueezeSeg Point cloud Visualization

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import argparse
import os
import numpy as np
from PIL import Image


from datetime import datetime
import os.path
import sys
import time
import glob    
from six.moves import xrange
import tensorflow as tf

from config import *
from imdb import kitti
from utils.util import *
from nets import *


def norm(x):
    return (x - x.min()) / (x.max() - x.min())

class ImageConverter(object):
    """
    Convert images/compressedimages to and from ROS
    From: https://github.com/CURG-archive/ros_rsvp
    """
    ENCODINGMAP_PY_TO_ROS = {'L': 'mono8', 'RGB': 'rgb8','RGBA': 'rgba8', 'YCbCr': 'yuv422'}
    PIL_MODE_CHANNELS = {'L': 1, 'RGB': 3, 'RGBA': 4, 'YCbCr': 3}

    @staticmethod
    def toROS(img):
        """
        Convert a PIL/pygame image to a ROS compatible message (sensor_msgs.Image).
        """
        if img.mode == 'P': # P -> 8-bit pixels, mapped to any other mode using a color palette
            img = img.convert('RGB') # RGB -> 3x8-bit pixels, true color

        rosImage = ImageMsg()
        rosImage.encoding = ImageConverter.ENCODINGMAP_PY_TO_ROS[img.mode]
        (rosImage.width, rosImage.height) = img.size
        rosImage.step = (ImageConverter.PIL_MODE_CHANNELS[img.mode] * rosImage.width)
        rosImage.data = img.tobytes()

        return rosImage

class VisualizeNode(object):
    """
    A ros node to publish training set 2D spherical surface image and point clouds
    """
    HEADER = Header(stamp=rospy.Time(), frame_id='velodyne')
    FIELDS = [
    	PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
		PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
		PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
		PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
		PointField(name='range', offset=20, datatype=PointField.FLOAT32, count=1),
		PointField(name='label', offset=24, datatype=PointField.FLOAT32, count=1)
	]

    def __init__(self, FLAGS):
       	
       	# ROS
       	self.publisherPointCloud   = rospy.Publisher(FLAGS.topic_pcl, PointCloud2, queue_size=1)
       	self.publisherIntensityMap = rospy.Publisher(FLAGS.topic_intensity, ImageMsg, queue_size=1)
       	self.publisherRangeMap     = rospy.Publisher(FLAGS.topic_range, ImageMsg, queue_size=1)
       	self.publisherLabelMap     = rospy.Publisher(FLAGS.topic_label, ImageMsg, queue_size=1)

        # Initialize ROS node
        rospy.init_node('npy_node', anonymous=True)
        rospy.loginfo("npy_node started.")
        rospy.loginfo("Publishing pointclouds from %s in '%s'+'%s'+'%s' topics at %d(hz)...", 
        	FLAGS.input_path, FLAGS.topic_intensity, FLAGS.topic_label, FLAGS.topic_pcl, FLAGS.rate)

        rate     = rospy.Rate(FLAGS.rate)
        counter  = 0
        npyFiles = []

        # Tensorflow
      	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
      	with tf.Graph().as_default():
      		mc = kitti_squeezeSeg_config()
      		mc.LOAD_PRETRAINED_MODEL = False
      		mc.BATCH_SIZE = 1 
      		model = SqueezeSeg(mc)

      		saver = tf.train.Saver(model.model_params)
    		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    			saver.restore(sess, FLAGS.checkpoint)
    			for file in glob.iglob(FLAGS.input_path):
    				npyFiles.append(file)

    			npyFiles.sort()

    			for file in npyFiles:
    				lidarPrediction = np.load(file).astype(np.float32, copy=False)
        			lidar = lidarPrediction[:, :, :5]
         			lidar_mask = np.reshape((lidar[:, :, 4] > 0), [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])
         			lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD

         			pred_cls = sess.run(
	             		model.pred_cls,
	             		feed_dict={
	                 		model.lidar_input:[lidar],
	                 		model.keep_prob: 1.0,
	                 		model.lidar_mask:[lidar_mask]
	             		}
        			)
        			lidarPrediction[:, :, 5] = pred_cls[0]

        			if rospy.is_shutdown():
        				break
        			
        			self.npyPublish(lidarPrediction, file)
        			counter =+ 1
        			rate.sleep()

        		rospy.logwarn("%d frames published.", counter)

    def npyPublish(self, npyFile, file):

        # Point cloud shape (X, 512, 6), where X = 64, 32, 16
        # npyFile = np.load(file).astype(np.float32, copy=False)
        lidar   = npyFile[:, :, :5] # x, y, z, i, r
        label   = npyFile[:, :,  5] # label
       
        label3D = np.zeros((label.shape[0], label.shape[1], 3))
        label3D[np.where(label==0)] = [1., 1., 1.]
        label3D[np.where(label==1)] = [1., 0., 0.]
        label3D[np.where(label==2)] = [0., 1., 0.]
        label3D[np.where(label==3)] = [0., 1., 1.]
        label3D[np.where(label==4)] = [0., 0., 1.]

        # point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        r = lidar[:, :, 4].reshape(-1)
    	
    	#TODO: fix hard-coded label color
        label = label.reshape(-1)
        labelpf = np.zeros((label.shape))

        label[np.where(label == 0)] = 0
        label[np.where(label == 1)] = 150
        label[np.where(label == 2)] = 200
        label[np.where(label == 3)] = 50
        label[np.where(label == 4)] = 120

        # Point cloud shape (6, 32768)
        pc = np.stack((x, y, z, i, r, label)) 

        # Range Map
        rangeMap = Image.fromarray((255 * norm(lidar[:, :, 4])).astype(np.uint8))
        # Intensity map
        intensityMap = Image.fromarray((255 * norm(lidar[:, :, 3])).astype(np.uint8))
       	# Label map
        labelMap = Image.fromarray((255 * norm(label3D)).astype(np.uint8))

        # Messages to publish
        msgIntensity = ImageConverter.toROS(intensityMap)
        msgIntensity.header = VisualizeNode.HEADER
        msgRange = ImageConverter.toROS(rangeMap)
        msgRange.header = VisualizeNode.HEADER
        msgLabel = ImageConverter.toROS(labelMap)
        msgLabel.header = VisualizeNode.HEADER
        msgSegment = pc2.create_cloud(header=VisualizeNode.HEADER, fields=VisualizeNode.FIELDS, points=pc.T)

        self.publisherIntensityMap.publish(msgIntensity)
        self.publisherRangeMap.publish(msgRange)
        self.publisherLabelMap.publish(msgLabel)
        self.publisherPointCloud.publish(msgSegment)
        
        filename = file.strip('.npy').split('/')[-1]
        rospy.loginfo("%s published.", filename)
