#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Training Dataset Visualization
"""

import argparse
# import glob
import os
import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class ImageConverter(object):
    """
    Convert images/compressedimages to and from ROS
    From: https://github.com/CURG-archive/ros_rsvp
    """
    _ENCODINGMAP_PY_TO_ROS = {'L': 'mono8', 'RGB': 'rgb8',
                              'RGBA': 'rgba8', 'YCbCr': 'yuv422'}
    _ENCODINGMAP_ROS_TO_PY = {'mono8': 'L', 'rgb8': 'RGB',
                              'rgba8': 'RGBA', 'yuv422': 'YCbCr'}
    _PIL_MODE_CHANNELS = {'L': 1, 'RGB': 3, 'RGBA': 4, 'YCbCr': 3}

    @staticmethod
    def to_ros(img):
        """
        Convert a PIL/pygame image to a ROS compatible message (sensor_msgs.Image).
        """

        # Everything ok, convert PIL.Image to ROS and return it
        if img.mode == 'P':
            img = img.convert('RGB')

        rosimage = ImageMsg()
        rosimage.encoding = ImageConverter._ENCODINGMAP_PY_TO_ROS[img.mode]
        (rosimage.width, rosimage.height) = img.size
        rosimage.step = (ImageConverter._PIL_MODE_CHANNELS[img.mode]
                         * rosimage.width)
        rosimage.data = img.tobytes()
        return rosimage

    # @classmethod
    # def from_ros(cls, rosMsg):
    #     """
    #     Converts a ROS sensor_msgs.Image or sensor_msgs.CompressedImage to a pygame Surface
    #     :param rosMsg: The message to convert
    #     :return: an alpha-converted pygame Surface
    #     """
    #     pyimg = None
    #     if isinstance(rosMsg, sensor_msgs.msg.Image):
    #         pyimg = pygame.image.fromstring(rosMsg.data, (rosMsg.width, rosMsg.height),
    #                                         cls._ENCODINGMAP_ROS_TO_PY[rosMsg.encoding])
    #     elif isinstance(rosMsg, sensor_msgs.msg.CompressedImage):
    #         pyimg = pygame.image.load(StringIO(rosMsg.data))
    #
    #     if not pyimg:
    #         raise TypeError('rosMsg is not an Image or CompressedImage!')
    #
    #     return pyimg.convert_alpha()

class TrainingSetNode(object):
    """
    A ros node to publish training set 2D spherical surface image
    """

    def __init__(self, dataset_path='./data/lidar_2d',
                 pub_rate=10,
                 pub_feature_map_topic='/squeeze_seg/feature_map',
                 pub_label_map_topic='/squeeze_seg/label_map'):
        """
        ros node spin in init function

        :param dataset_path:
        :param pub_feature_map_topic:
        :param pub_label_map_topic:
        :param pub_rate:
        """

        self._path = dataset_path + "/"
        self._pub_rate = pub_rate
        # publisher
        self._feature_map_pub = rospy.Publisher(pub_feature_map_topic, ImageMsg, queue_size=1)
        self._label_map_pub = rospy.Publisher(pub_label_map_topic, ImageMsg, queue_size=1)
        # ros node init
        rospy.init_node('npy_node', anonymous=True)
        rospy.loginfo("npy_node started.")
        rospy.loginfo("publishing dataset %s in '%s'+'%s' topic with %d(hz)...", self._path,
                      pub_feature_map_topic, pub_label_map_topic, self._pub_rate)

        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"

        rate = rospy.Rate(self._pub_rate)
        cnt = 0

        npy_files = []
        if os.path.isdir(self._path):
            for f in os.listdir(self._path):
                if os.path.isdir(f):
                    continue
                else:
                    npy_files.append(f)
        npy_files.sort()

        # for f in glob.iglob(self.path_):
        for f in npy_files:
            if rospy.is_shutdown():
                break

            self.publish_image(self._path + "/" + f, header)
            cnt += 1

            rate.sleep()

        rospy.logwarn("%d frames published.", cnt)

    def publish_image(self, img_file, header):
        record = np.load(img_file).astype(np.float32, copy=False)

        lidar = record[:, :, :5]    # x, y, z, intensity, depth
        # print lidar

        label = record[:, :, 5]     # point-wise label
        # label = _normalize(label)
        # g=p*R+q*G+t*B, where p=0.2989,q=0.5870,t=0.1140
        # p = 0.2989;q = 0.5870;t = 0.1140
        # label_3d = np.dstack((p*label, q*label, t*label))
        label_3d = np.zeros((label.shape[0], label.shape[1], 3))
        label_3d[np.where(label==0)] = [1., 1., 1.]
        label_3d[np.where(label==1)] = [1., 0., 0.]
        label_3d[np.where(label==2)] = [0., 1., 0.]
        label_3d[np.where(label==3)] = [0., 1., 1.]
        label_3d[np.where(label==4)] = [0., 0., 1.]
        # print label_3d
        # print np.min(label)
        # print np.max(label)

        # insert label into lidar infos
        lidar[np.where(label==1)] = [1., 0., 0., 0., 0.]
        lidar[np.where(label==2)] = [0., 1., 0., 0., 0.]
        lidar[np.where(label==3)] = [0., 1., 1., 0., 0.]
        lidar[np.where(label==3)] = [0., 0., 1., 0., 0.]
        # generated feature map from LiDAR data
        ##x/y/z
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, 0])).astype(np.uint8))
        ##depth map
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, 4])).astype(np.uint8))
        ##intensity map
        feature_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, :3])).astype(np.uint8))
        # generated label map from LiDAR data
        label_map = Image.fromarray(
            (255 * _normalize(label_3d)).astype(np.uint8))

        msg_feature = ImageConverter.to_ros(feature_map)
        msg_feature.header = header
        msg_label = ImageConverter.to_ros(label_map)
        msg_label.header = header

        self._feature_map_pub.publish(msg_feature)
        self._label_map_pub.publish(msg_label)

        file_name = img_file.strip('.npy').split('/')[-1]
        rospy.loginfo("%s published.", file_name)

if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud 2D spherical surface publisher')
    parser.add_argument('--datasetPath', type=str,
                        help='the path of training dataset, default `./data/lidar_2d`',
                        default='./data/lidar_2d')
    parser.add_argument('--pub_rate', type=int,
                        help='the frequency(hz) of image published, default `10`',
                        default=10)
    parser.add_argument('--pub_feature_map_topic', type=str,
                        help='the 2D spherical surface image message topic to be published, default `/squeeze_seg/feature_map`',
                        default='/squeeze_seg/feature_map')
    parser.add_argument('--pub_label_map_topic', type=str,
                        help='the corresponding ground truth label image message topic to be published, default `/squeeze_seg/label_map`',
                        default='/squeeze_seg/label_map')
    args = parser.parse_args()

    # start training_set_node
    node = TrainingSetNode(dataset_path=args.datasetPath,
                           pub_rate=args.pub_rate,
                           pub_feature_map_topic=args.pub_feature_map_topic,
                           pub_label_map_topic=args.pub_label_map_topic)

    rospy.logwarn('finished.')
