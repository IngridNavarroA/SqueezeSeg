# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from imdb import kitti
from utils.util import *
from nets import *
# from sklearn.metrics import confusion_matrix

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI', """Currently support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val', """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/bichen/logs/squeezeSeg/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', '',"""Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1, """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg', """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string('extended', 'y', """Extended classes.""")
tf.app.flags.DEFINE_string('restore', 'n', """Start from checkpoint""")

# def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
#     """pretty print for confusion matrixes"""
#     columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
#     empty_cell = " " * columnwidth
#     # Print header
#     print("    " + empty_cell, end=" ")
#     for label in labels:
#         print("%{0}s".format(columnwidth) % label, end=" ")
#     print()
#     # Print rows
#     for i, label1 in enumerate(labels):
#         print("    %{0}s".format(columnwidth) % label1, end=" ")
#         for j in range(len(labels)):
#             cell = "%{0}.4f".format(columnwidth) % cm[i, j]
#             if hide_zeroes:
#                 cell = cell if float(cm[i, j]) != 0 else empty_cell
#             if hide_diagonal:
#                 cell = cell if i != j else empty_cell
#             if hide_threshold:
#                 cell = cell if cm[i, j] > hide_threshold else empty_cell
#             print(cell, end=" ")
#         print()

def eval_once(saver, ckpt_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb, model):
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Restores from checkpoint
    saver.restore(sess, ckpt_path)
    # Assuming model_checkpoint_path looks something like:
    #   /ckpt_dir/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt_path.split('/')[-1].split('-')[-1]

    mc = model.mc
    mc.DATA_AUGMENTATION = False

    num_images = len(imdb.image_idx)

    _t = {
        'detect': Timer(),
        'read': Timer(),
        'eval': Timer()
    }

    tot_error_rate, tot_rmse, tot_th_correct = 0.0, 0.0, 0.0

    # class-level metrics
    tp_sum = np.zeros(mc.NUM_CLASS)
    fn_sum = np.zeros(mc.NUM_CLASS)
    fp_sum = np.zeros(mc.NUM_CLASS)
    # instance-level metrics
    itp_sum = np.zeros(mc.NUM_CLASS)
    ifn_sum = np.zeros(mc.NUM_CLASS)
    ifp_sum = np.zeros(mc.NUM_CLASS)
    # instance-level object matching metrics
    otp_sum = np.zeros(mc.NUM_CLASS)
    ofn_sum = np.zeros(mc.NUM_CLASS)
    ofp_sum = np.zeros(mc.NUM_CLASS)

    # if mc.EVAL_ON_ORG:
    #   cm = np.zeros([4, 4])
    # else: 
    #   cm = np.zeros([mc.NUM_CLASS, mc.NUM_CLASS])

    for i in xrange(int(num_images/mc.BATCH_SIZE)):
      offset = max((i+1)*mc.BATCH_SIZE - num_images, 0)
      
      _t['read'].tic()
      lidar_per_batch, lidar_mask_per_batch, label_per_batch, _ \
          = imdb.read_batch(shuffle=False)
      _t['read'].toc()

      _t['detect'].tic()
      pred_cls = sess.run(
          model.pred_cls, 
          feed_dict={
              model.lidar_input:lidar_per_batch,
              model.keep_prob: 1.0,
              model.lidar_mask:lidar_mask_per_batch
          }
      )
      _t['detect'].toc()

      _t['eval'].tic()

      # if  mc.EVAL_ON_ORG:
      #   print('Translating labels...')
      #   pred_cls[(pred_cls!=0)*(pred_cls!=7)*(pred_cls!=6)*(pred_cls!=9)] =0 
      #   pred_cls[pred_cls==6]=2
      #   pred_cls[pred_cls==9]=3
      #   pred_cls[pred_cls==7]=1           
      #   cm+=confusion_matrix(y_true=label_per_batch.flatten(), y_pred=pred_cls.flatten(), labels=range(4))
      # else:
      #   cm+=confusion_matrix(y_true=label_per_batch.flatten(), y_pred=pred_cls.flatten(), labels=range(mc.NUM_CLASS))
      
      # Evaluation
      iou, tps, fps, fns = evaluate_iou(
          label_per_batch[:mc.BATCH_SIZE-offset],
          pred_cls[:mc.BATCH_SIZE-offset] \
          * np.squeeze(lidar_mask_per_batch[:mc.BATCH_SIZE-offset]),
          mc.NUM_CLASS
      )

      tp_sum += tps
      fn_sum += fns
      fp_sum += fps

      _t['eval'].toc()

      print ('detect: {:d}/{:d} im_read: {:.3f}s '
          'detect: {:.3f}s evaluation: {:.3f}s'.format(
                (i+1)*mc.BATCH_SIZE-offset, num_images,
                _t['read'].average_time/mc.BATCH_SIZE,
                _t['detect'].average_time/mc.BATCH_SIZE,
                _t['eval'].average_time/mc.BATCH_SIZE))

    ious = tp_sum.astype(np.float)/(tp_sum + fn_sum + fp_sum + mc.DENOM_EPSILON)
    pr = tp_sum.astype(np.float)/(tp_sum + fp_sum + mc.DENOM_EPSILON)
    re = tp_sum.astype(np.float)/(tp_sum + fn_sum + mc.DENOM_EPSILON)

    print ('Evaluation summary:')
    print ('  Timing:')
    print ('    read: {:.3f}s detect: {:.3f}s'.format(
        _t['read'].average_time/mc.BATCH_SIZE,
        _t['detect'].average_time/mc.BATCH_SIZE
    ))

    eval_sum_feed_dict = {
        eval_summary_phs['Timing/detect']:_t['detect'].average_time/mc.BATCH_SIZE,
        eval_summary_phs['Timing/read']:_t['read'].average_time/mc.BATCH_SIZE,
    }

    # if mc.EVAL_ON_ORG:
    #     print ('  Accuracy:')
    #     classes=['unknown' ,  'car', 'pedestrian'  ,  'cyclist'] 
    #     for i in range(1, 4):
    #         print ('    {}:'.format(classes[i]))
    #         print ('\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}'.format(
    #             pr[i], re[i], ious[i]))

    print ('  Accuracy:')
    for i in range(1, mc.NUM_CLASS):
      if not mc.EVAL_ON_ORG:
          print ('    {}:'.format(mc.CLASSES[i]))
          print ('\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}'.format(pr[i], re[i], ious[i]))
      eval_sum_feed_dict[
          eval_summary_phs['Pixel_seg_accuracy/'+mc.CLASSES[i]+'_iou']] = ious[i]
      eval_sum_feed_dict[
          eval_summary_phs['Pixel_seg_accuracy/'+mc.CLASSES[i]+'_precision']] = pr[i]
      eval_sum_feed_dict[
          eval_summary_phs['Pixel_seg_accuracy/'+mc.CLASSES[i]+'_recall']] = re[i]

    eval_summary_str = sess.run(eval_summary_ops, feed_dict=eval_sum_feed_dict)
    for sum_str in eval_summary_str:
      summary_writer.add_summary(sum_str, global_step)
    summary_writer.flush()
    # cm = cm / cm.sum(axis=1)[:, np.newaxis] 
    # if  mc.EVAL_ON_ORG:
    #     print_cm(cm, ['unknown' ,  'car', 'pedestrian'  ,  'cyclist'])
    # else:
    #     print_cm(cm, mc.CLASSES[i])

def evaluate():
  """Evaluate."""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only supports KITTI dataset'

  if FLAGS.gpu == '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "" # Train only with CPU

  with tf.Graph().as_default() as g:

    assert FLAGS.net == 'squeezeSeg' or FLAGS.net == 'squeezeSeg32' or FLAGS.net == 'squeezeSeg16', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    if FLAGS.net == 'squeezeSeg':  
      if FLAGS.extended == 'y':
        mc = kitti_squeezeSeg_config_ext() # Added ground class
      else:
        mc = kitti_squeezeSeg_config()   # Original training set
      mc.LOAD_PRETRAINED_MODEL = False
      mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
      model = SqueezeSeg(mc)
    elif FLAGS.net == 'squeezeSeg32':    
      if FLAGS.extended == 'y':
        mc = kitti_squeezeSeg32_config_ext() # Added ground class
      else:
        mc = kitti_squeezeSeg32_config()   # Original training set

      mc.LOAD_PRETRAINED_MODEL = False #FLAGS.pretrained_model_path
      mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
      model = SqueezeSeg32(mc)
    else:    
      if FLAGS.extended == 'y':
        mc = kitti_squeezeSeg16_config_ext() # Added ground class
      else:
        mc = kitti_squeezeSeg16_config()   # Original training set

      mc.LOAD_PRETRAINED_MODEL = False #FLAGS.pretrained_model_path
      mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
      model = SqueezeSeg32(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    eval_summary_ops = []
    eval_summary_phs = {}

    eval_summary_names = [
        'Timing/read', 
        'Timing/detect',
    ]
    for i in range(1, mc.NUM_CLASS):
      eval_summary_names.append('Pixel_seg_accuracy/'+mc.CLASSES[i]+'_iou')
      eval_summary_names.append('Pixel_seg_accuracy/'+mc.CLASSES[i]+'_precision')
      eval_summary_names.append('Pixel_seg_accuracy/'+mc.CLASSES[i]+'_recall')

    for sm in eval_summary_names:
      ph = tf.placeholder(tf.float32)
      eval_summary_phs[sm] = ph
      eval_summary_ops.append(tf.summary.scalar(sm, ph))

    saver = tf.train.Saver(model.model_params)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    
    ckpts = set() 
    while True:
      if FLAGS.run_once:
        # When run_once is true, checkpoint_path should point to the exact
        # checkpoint file.
        eval_once(
            saver, FLAGS.checkpoint_path, summary_writer, eval_summary_ops,
            eval_summary_phs, imdb, model)
        return
      else:
        # When run_once is false, checkpoint_path should point to the directory
        # that stores checkpoint files.
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
          if ckpt.model_checkpoint_path in ckpts:
            # Do not evaluate on the same checkpoint
            print ('Wait {:d}s for new checkpoints to be saved ... '
                      .format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)
          else:
            ckpts.add(ckpt.model_checkpoint_path)
            print ('Evaluating {}...'.format(ckpt.model_checkpoint_path))
            eval_once(
                saver, ckpt.model_checkpoint_path, summary_writer,
                eval_summary_ops, eval_summary_phs, imdb, model)
        else:
          print('No checkpoint file found')
          if not FLAGS.run_once:
            print ('Wait {:d}s for new checkpoints to be saved ... '
                      .format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir) and FLAGS.restore == 'n':
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
