# from util import load_image
from detector import Detector
import model_param


import skimage.color 
import tensorflow as tf
import numpy as np
import pickle
import re

from os import listdir
from os.path import isfile, join
from skimage.transform import resize





class Forward_model:
  """Loads the caltech model in memory and forwards images through it"""
  
  def __init__(self, mod_param, epochNb=3):
    """Load the tensorflow model of VGG16-GAP trained on caltech
    
    Keyword arguments:
      epochNb -- iteration of the model to consider
    """
    tf.reset_default_graph()
    weight_path    = '../caffe_layers_value.pickle'
    model_path     = mod_param.paths["save_model"]
    model_path    += '-'+str(epochNb)
    self.mod_param = mod_param
    self.labels    = mod_param.labels
    self.n_labels  = mod_param.n_labels
    
    # Initialize some tensorflow variables
    self.images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
    self.labels_tf = tf.placeholder( tf.int64, [None], name='labels')
    
    detector = Detector(mod_param)
    c1,c2,c3,c4,conv5, self.conv6, gap, self.output = detector.inference( self.images_tf )
    if re.match(".*CAM._W_S", mod_param.mod_type):
      self.classmap = detector.get_classmap( self.labels_tf, self.conv6 )
    
    self.sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore( self.sess, model_path )
  
  def _visualize_cam_w_s(self, imgs):
    feed_dict = {}
    feed_dict[self.images_tf] = imgs
    conv6_val, output_val = self.sess.run([self.conv6, self.output],feed_dict=feed_dict)
    preds = output_val
    preds_order = preds.argsort( axis=1 )[:,::-1]
    best_preds  = preds.argmax(  axis=1 )
    
    classmap_vals = self.sess.run(
                        self.classmap,
                        feed_dict={
                          self.labels_tf: best_preds,
                          self.conv6: conv6_val
                        })
    
    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
    named_preds  = map(self.to_named_pred, preds)
    return named_preds, np.array(classmap_vis)[0]
  
  def _visualize_cam_s(self, imgs):
    feed_dict = {}
    feed_dict[self.images_tf] = imgs
    conv6_val, output_val = self.sess.run([self.conv6, self.output],feed_dict=feed_dict)
    preds = output_val
    named_preds = map(self.to_named_pred, preds)
    return named_preds, conv6_val
  
  def _check_image(self, img):
    if img.shape[-1] == 1:
      img = skimage.color.greytorgb(img)
    if img.shape[-3:] != (224,224,3):
      print "WARNING: FUCKED UP SHAPE, not (224*224*3) !!"
    return img
  
  def to_named_pred(self, preds):
    named_preds = [(self.labels[idx], p) for idx,p in enumerate(preds)]
    return named_preds
  
  def forward_image(self, img):
    """Retrieve prediction's text and visualisation
    
      Keyword arguments:
        img -- image to feed forward as numpy array
        vis -- saliency map to visualize (-1 for best prediction)
    """
    # Reshape image
    img = img.reshape(1,224,224,3)
    return self.forward_images(img)[0]
  
  def forward_images(self, imgs, visualize=False):
    """Retrieve prediction's text and visualisation
    
      Keyword arguments:
        imgs -- image to feed forward as numpy array
        vis -- saliency map to visualize (-1 for best prediction)
      
      returns :
        named_preds -- predictions with their names
        visualisation -- a 2D saliency map
    """
    # Reshape image
    imgs = self._check_image(imgs)
  
    if visualize == False:
      feed_dict = {}
      feed_dict[self.images_tf] = imgs
      output_val = self.sess.run(self.output,feed_dict=feed_dict)
      preds = output_val
      named_preds = map(self.to_named_pred, preds)
      return named_preds
    
    if self.mod_param.mod_type == "VGG16_CAM_W_S":
      return self._visualize_cam_w_s(imgs)
  
    if self.mod_param.mod_type in ("VGG16_CAM_S", 'VGG16_CAM5_S', 'VGG16_CAM7_S', 'VGG16P_CAM3_S', "VGG16P_CAM5_S" ):
      return self._visualize_cam_s(imgs)
  



if __name__ == '__main__':
  import sys; sys.path.append("/home/cuda/datasets/VOC2012")
  from voc2012_getter import get_batch
  a = get_batch('test')
  batch = a.next()
  imgs = batch[0]

  mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM7_S", 'rmsProp', 0.000008, 0.00005)
  model = Forward_model(mod_param, 6)
  model.forward_images(imgs, True)




