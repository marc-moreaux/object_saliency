# from util import load_image
from detector import Detector

import tensorflow as tf
import numpy as np
import pickle

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
    weight_path    = '../caffe_layers_value.pickle'
    model_path     = mod_param.paths["save_model"]
    model_path    += '-'+str(epochNb)
    self.mod_param = mod_param
    self.labels    = mod_param.labels
    n_labels       = mod_param.n_labels
    
    # Initialize some tensorflow variables
    self.images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
    self.labels_tf = tf.placeholder( tf.int64, [None], name='labels')
    
    detector = Detector(mod_param)
    c1,c2,c3,c4,conv5, self.conv6, gap, self.output = detector.inference( self.images_tf )

    if mod_param.mod_type == "VGG16_CAM_W_S":
      self.classmap = detector.get_classmap( self.labels_tf, self.conv6 )
    
    self.sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore( self.sess, model_path )
  
  def forward_image(self, img, vis=-1):
    """Retrieve prediction's text and visualisation
    
    Keyword arguments:
      img -- image to feed forward as numpy array
      vis -- saliency map to visualize (-1 for best prediction)
    """
    # Reshape image
    if img.shape != (224,224,3):
      print "Image might have wierd shape..."
      img = resize(img, [224,224])
    if img.shape != (1,224,224,3):
      img = img.reshape(1,224,224,3)
    
    # Feed network with the image
    feed_dict = {}
    feed_dict[self.images_tf] = img
    conv6_val, output_val = self.sess.run([self.conv6, self.output],feed_dict=feed_dict)
    preds = output_val
    preds_order = preds.argsort( axis=1 )[:,::-1]
    best_pred   = preds.argmax(  axis=1 ) if vis == -1 else [vis]

    if self.mod_param.mod_type == "VGG16_CAM_W_S":
      classmap_vals = self.sess.run(
                          self.classmap,
                          feed_dict={
                            self.labels_tf: best_pred,
                            self.conv6: conv6_val
                          })
      
      classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
      named_preds = [(self.labels[i], preds[0][i]) for i in preds_order[0]]
      return named_preds, np.array(classmap_vis)[0]

    if self.mod_param.mod_type in ("VGG16_CAM_S", 'VGG16_CAM5_S', 'VGG16_CAM7_S', 'VGG16P_CAM3_S', "VGG16P_CAM5_S" ):
      named_preds = [(self.labels[idx], p) for idx,p in enumerate(preds[0])]
      return named_preds, conv6_val



  def forward_images(self, imgs, visualize=False):
      """Retrieve prediction's text and visualisation
      
      Keyword arguments:
        imgs -- image to feed forward as numpy array
        vis -- saliency map to visualize (-1 for best prediction)
      """
      # Reshape image
      if imgs.shape[-3:] != (224,224,3):
        print "FUCKED UP SHAPE !!!"

      
      # Feed network with the image
      feed_dict = {}
      feed_dict[self.images_tf] = imgs
      conv6_val, output_val = self.sess.run([self.conv6, self.output],feed_dict=feed_dict)
      preds = output_val

      if visualize == False:
        return preds
      
      best_pred   = preds.argmax( axis=1 )
      classmap_vals = self.sess.run(
                    self.classmap,
                    feed_dict={
                      self.labels_tf: best_pred,
                      self.conv6: conv6_val
                    })
      

      return classmap_vals, named_preds
      
      
      # classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
      # named_preds = [(self.labels[i], preds[0][i]) for i in preds_order[0]]
      # return named_preds, np.array(classmap_vis)[0]





