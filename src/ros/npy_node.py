#!/usr/bin/python2
"""
author: Ingrid Navarro 
date: July 11th, 2018

	SqueezeSeg Point cloud Visualization

"""
import argparse
import os
import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

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

        rosimage = ImageMsg()
        rosimage.encoding = ImageConverter.ENCODINGMAP_PY_TO_ROS[img.mode]
        (rosimage.width, rosimage.height) = img.size
        rosimage.step = (ImageConverter.PIL_MODE_CHANNELS[img.mode] * rosimage.width)
        rosimage.data = img.tobytes()
        return rosimage

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

    def __init__(self, inpath='./data/lidar_2d', rate=10, 
    		     topicIntensityMap='/squeeze_seg/intensity_map',
                 topicLabelMap='/squeeze_seg/label_map',
                 topicRangeMap='/squeeze_seg/range_map',
                 topicPointCloud='/squeeze_seg/points'):
       	
       	self.path = inpath + "/"
       	self.rate = rate

       	self.publisherPointCloud   = rospy.Publisher(topicPointCloud, PointCloud2, queue_size=1)
       	self.publisherIntensityMap = rospy.Publisher(topicIntensityMap, ImageMsg, queue_size=1)
       	self.publisherRangeMap     = rospy.Publisher(topicRangeMap, ImageMsg, queue_size=1)
       	self.publisherLabelMap     = rospy.Publisher(topicLabelMap, ImageMsg, queue_size=1)

        # Initialize ROS node
        rospy.init_node('npy_node', anonymous=True)
        rospy.loginfo("npy_node started.")
        rospy.loginfo("Publishing pointclouds from %s in '%s'+'%s'+'%s' topics at %d(hz)...", self.path, topicIntensityMap, topicLabelMap, topicPointCloud, self.rate)

        rate     = rospy.Rate(self.rate)
        counter  = 0
        npyFiles = []
       
        if os.path.isdir(self.path):
            for file in os.listdir(self.path):
                if os.path.isdir(file):
                    continue
                else:
                    npyFiles.append(file)
        npyFiles.sort()

        for file in npyFiles:
            if rospy.is_shutdown():
                break

            self.npyPublish(self.path + "/" + file)
            counter += 1
            rate.sleep()

        rospy.logwarn("%d frames published.", counter)

    def npyPublish(self, file):

        # Point cloud shape (X, 512, 6), where X = 64, 32, 16
        npyFile = np.load(file).astype(np.float32, copy=False)
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

def main():
	# Argument parsing 
	ap = argparse.ArgumentParser()
	ap.add_argument("--point_cloud", default="/squeeze_seg/points", help="Topic to publish point clouds.")
	ap.add_argument("--label_map", default="/squeeze_seg/label_map", help="Topic to pusblish label maps.")
	ap.add_argument("--intensity_map", default="/squeeze_seg/intensity_map", help="Topic to pusblish intensity maps.")
	ap.add_argument("--range_map", default="/squeeze_seg/range_map", help="Topic to pusblish range maps.")
	ap.add_argument("--inpath", default="/data/samples", help="Input path")
	ap.add_argument("--rate", default=10, help="Publishing rate.")
	args = ap.parse_args()

	# Create ros node
	node = VisualizeNode(inpath=args.inpath, rate=args.rate, 
				 		   topicIntensityMap=args.intensity_map, 
						   topicLabelMap=args.label_map, 
						   topicRangeMap=args.range_map,
						   topicPointCloud=args.point_cloud)
	
	rospy.logwarn('finished.')

if __name__ == '__main__':
	main()
