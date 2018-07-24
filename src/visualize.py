#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Segmentation on LiDAR point cloud using SqueezeSeg Neural Network
"""

import argparse
import tensorflow as tf
import rospy

from ros.npy_visualizer import VisualizeNode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint', './data/SqueezeSeg/model.ckpt-23000', """Path to the model parameter file.""")
tf.app.flags.DEFINE_string('input_path', './data/test64/*', """Input lidar scan to be detected. """)
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string("point_cloud", "/squeeze_seg/points", """Topic to publish point clouds.""")
tf.app.flags.DEFINE_string("label_map", "/squeeze_seg/label_map", """Topic to pusblish label maps.""")
tf.app.flags.DEFINE_string("intensity_map", "/squeeze_seg/intensity_map", """Topic to pusblish intensity maps.""")
tf.app.flags.DEFINE_string("range_map", "/squeeze_seg/range_map", """Topic to pusblish range maps.""")
tf.app.flags.DEFINE_string("rate", 20, """Publishing rate.""")

def main(argv=None):
    
    # Create ros node
    node = VisualizeNode(FLAGS=FLAGS) 
    rospy.logwarn('finished.')

if __name__ == '__main__':
    tf.app.run()