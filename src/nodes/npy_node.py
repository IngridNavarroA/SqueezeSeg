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
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    #if num_field == 4:
    #    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

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
        rosimage.step = (ImageConverter._PIL_MODE_CHANNELS[img.mode] * rosimage.width)
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
                 pub_label_map_topic='/squeeze_seg/label_map',
                 sub_topic='/kitti/points_raw',
                 pub_topic='/squeeze_seg/points'):
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
        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)
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

        ## point cloud in filed of view
        # x = np_p_ranged[:, 0].reshape(-1)
        # y = np_p_ranged[:, 1].reshape(-1)
        # z = np_p_ranged[:, 2].reshape(-1)
        # i = np_p_ranged[:, 3].reshape(-1)
        ## point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        # cond = (label!=0)
        # print(cond)
        cloud = np.stack((x, y, z, i, label))


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

        msg_segment = pc2.create_cloud(header=header,
                                       fields=_make_point_field(cloud.shape[0]),
                                       points=cloud.T)


        self._feature_map_pub.publish(msg_feature)
        self._label_map_pub.publish(msg_label)
        self._pub.publish(msg_segment)
        file_name = img_file.strip('.npy').split('/')[-1]
        rospy.loginfo("%s published.", file_name)

if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud semantic segmentation')
    parser.add_argument('--sub_topic', type=str,
                        help='the pointcloud message topic to be subscribed, default `/kitti/points_raw`',
                        default='/kitti/points_raw')
    parser.add_argument('--pub_topic', type=str,
                        help='the pointcloud message topic to be published, default `/squeeze_seg/points`',
                        default='/squeeze_seg/points')
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
                           pub_label_map_topic=args.pub_label_map_topic, 
                           pub_topic=args.pub_topic)

    rospy.logwarn('finished.')
