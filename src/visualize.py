#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Segmentation on LiDAR point cloud using SqueezeSeg Neural Network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import rospy

from ros.npy_visualizer import VisualizeNode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint', './data/SqueezeSeg/model.ckpt-23000', """Path to the model parameter file.""")
tf.app.flags.DEFINE_string('input_path', './data/test/g/g_vlp64_v1/*', """Input lidar scan to be detected. """)
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string("topic_pcl", "/squeeze_seg/points", """Topic to publish point clouds.""")
tf.app.flags.DEFINE_string("topic_label", "/squeeze_seg/label_map", """Topic to pusblish label maps.""")
tf.app.flags.DEFINE_string("topic_intensity", "/squeeze_seg/intensity_map", """Topic to pusblish intensity maps.""")
tf.app.flags.DEFINE_string("topic_range", "/squeeze_seg/range_map", """Topic to pusblish range maps.""")
tf.app.flags.DEFINE_integer("rate", 20, """Publishing rate.""")

def main(argv=None):
    
    # Create ros node
    node = VisualizeNode(FLAGS=FLAGS) 
    rospy.logwarn('finished.')

if __name__ == '__main__':
    tf.app.run()