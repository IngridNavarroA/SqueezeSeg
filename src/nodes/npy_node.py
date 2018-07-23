#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Training Dataset Visualization
"""

import argparse
import os
import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def makePointField(numFields):
    msgPF1 = pc2.PointField()
    msgPF1.name = np.str('x')
    msgPF1.offset = np.uint32(0)
    msgPF1.datatype = np.uint8(7)
    msgPF1.count = np.uint32(1)

    msgPF2 = pc2.PointField()
    msgPF2.name = np.str('y')
    msgPF2.offset = np.uint32(4)
    msgPF2.datatype = np.uint8(7)
    msgPF2.count = np.uint32(1)

    msgPF3 = pc2.PointField()
    msgPF3.name = np.str('z')
    msgPF3.offset = np.uint32(8)
    msgPF3.datatype = np.uint8(7)
    msgPF3.count = np.uint32(1)

    msgPF4 = pc2.PointField()
    msgPF4.name = np.str('intensity')
    msgPF4.offset = np.uint32(16)
    msgPF4.datatype = np.uint8(7)
    msgPF4.count = np.uint32(1)

    if numFields == 4:
        return [msgPF1, msgPF2, msgPF3, msgPF4]

    msgPF5 = pc2.PointField()
    msgPF5.name = np.str('label')
    msgPF5.offset = np.uint32(20)
    msgPF5.datatype = np.uint8(4)
    msgPF5.count = np.uint32(1)

    return [msgPF1, msgPF2, msgPF3, msgPF4, msgPF5]

class ImageConverter(object):
    """
    Convert images/compressedimages to and from ROS
    From: https://github.com/CURG-archive/ros_rsvp
    """
    _ENCODINGMAP_PY_TO_ROS = {'L': 'mono8', 'RGB': 'rgb8', 'RGBA': 'rgba8', 'YCbCr': 'yuv422'}
    _ENCODINGMAP_ROS_TO_PY = {'mono8': 'L', 'rgb8': 'RGB', 'rgba8': 'RGBA', 'yuv422': 'YCbCr'}
    _PIL_MODE_CHANNELS = {'L': 1, 'RGB': 3, 'RGBA': 4, 'YCbCr': 3}

    @staticmethod
    def toROS(img):
        """
        Convert a PIL/pygame image to a ROS compatible message (sensor_msgs.Image).
        """
        # Everything ok, convert PIL.Image to ROS and return it
        if img.mode == 'P':
            img = img.convert('RGB')

        rosImage = ImageMsg()
        rosImage.encoding = ImageConverter._ENCODINGMAP_PY_TO_ROS[img.mode]
        (rosImage.width, rosImage.height) = img.size
        rosImage.step = (ImageConverter._PIL_MODE_CHANNELS[img.mode] * rosImage.width)
        rosImage.data = img.tobytes()
        return rosImage

class VisualizeNode(object):
    """
    A ros node to publish training set 2D spherical surface image
    """
    def __init__(self, datasetPath='./data/lidar_2d', rate=10, pub='/squeeze_seg/points', 
        topicFeatureMap='/squeeze_seg/feature_map', topicLabelMap='/squeeze_seg/label_map'):
        """
        ros node spin in init function
        :param datasetPath:
        :param topicFeatureMap:
        :param topicLabelMap:
        :param rate:
        """
        self.path = datasetPath + "/"
        self.rate = rate
        # publisher
        self.featureMap = rospy.Publisher(topicFeatureMap, ImageMsg, queue_size=1)
        self.labelMap = rospy.Publisher(topicLabelMap, ImageMsg, queue_size=1)
        self.pub = rospy.Publisher(pub, PointCloud2, queue_size=1)
        
        # ros node init
        rospy.init_node('npy_node', anonymous=True)
        rospy.loginfo("npy_node started.")
        rospy.loginfo("Publishing test %s in '%s'+'%s' topics at %d(hz)...", self.path, topicFeatureMap, topicLabelMap, self.rate)

        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"

        rate = rospy.Rate(self.rate)
        cnt = 0

        npyFiles = []
        if os.path.isdir(self.path):
            for f in os.listdir(self.path):
                if os.path.isdir(f):
                    continue
                else:
                    npyFiles.append(f)
        npyFiles.sort()

        for f in npyFiles:
            if rospy.is_shutdown():
                break

            self.publishImage(self.path + "/" + f, header)
            cnt += 1

            rate.sleep()

        rospy.logwarn("%d frames published.", cnt)

    def publishImage(self, imgFile, header):
        record = np.load(imgFile).astype(np.float32, copy=False)

        lidar = record[:, :, :5]    # x, y, z, intensity, depth
        # print lidar

        label = record[:, :, 5]     # point-wise label

        label3D = np.zeros((label.shape[0], label.shape[1], 3))
        label3D[np.where(label==0)] = [1., 1., 1.]
        label3D[np.where(label==1)] = [1., 0., 0.]
        label3D[np.where(label==2)] = [0., 1., 0.]
        label3D[np.where(label==3)] = [0., 1., 1.]
        label3D[np.where(label==4)] = [0., 0., 1.]

        # insert label into lidar infos
        lidar[np.where(label==1)] = [1., 0., 0., 0., 0.]
        lidar[np.where(label==2)] = [0., 1., 0., 0., 0.]
        lidar[np.where(label==3)] = [0., 1., 1., 0., 0.]
        lidar[np.where(label==4)] = [0., 0., 1., 0., 0.]
        
        # generated feature map from LiDAR data x/y/z
        feature_map = Image.fromarray((255 * normalize(lidar[:, :, 3])).astype(np.uint8))

        # generated label map from LiDAR data
        label_map = Image.fromarray((255 * normalize(label3D)).astype(np.uint8))

        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        cloud = np.stack((x, y, z, i, label))

        msgFeature = ImageConverter.toROS(feature_map)
        msgFeature.header = header
        msgLabel = ImageConverter.toROS(label_map)
        msgLabel.header = header
        msgSegment = pc2.create_cloud(header=header, fields=makePointField(cloud.shape[0]), points=cloud.T)

        self.featureMap.publish(msgFeature)
        self.labelMap.publish(msgLabel)
        self.pub.publish(msgSegment)

        filename = imgFile.strip('.npy').split('/')[-1]
        rospy.loginfo("%s published.", filename)

if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud 2D spherical surface publisher')
    parser.add_argument('--datasetPath', type=str,
                        help='the path of training dataset, default `./data/lidar_2d`',
                        default='./data/lidar_2d')
    parser.add_argument('--rate', type=int,
                        help='the frequency(hz) of image published, default `10`',
                        default=5)
    parser.add_argument('--topicFeatureMap', type=str,
                        help='the 2D spherical surface image message topic to be published, default `/squeeze_seg/feature_map`',
                        default='/squeeze_seg/feature_map')
    parser.add_argument('--topicLabelMap', type=str,
                        help='the corresponding ground truth label image message topic to be published, default `/squeeze_seg/label_map`',
                        default='/squeeze_seg/label_map')
    parser.add_argument('--sub_topic', type=str,
                        help='the pointcloud message topic to be subscribed, default `/kitti/points_raw`',
                        default='/kitti/points_raw')
    parser.add_argument('--topicSegment', type=str,
                        help='the pointcloud message topic to be published, default `/squeeze_seg/points`',
                        default='/squeeze_seg/points')
    args = parser.parse_args()

    # start training_set_node
    node = VisualizeNode(datasetPath=args.datasetPath,
                         rate=args.rate,
                         pub=args.topicSegment, 
                         topicFeatureMap=args.topicFeatureMap,
                         topicLabelMap=args.topicLabelMap)

    rospy.logwarn('finished.')
