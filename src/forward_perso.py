# from util import load_image
from detector import Detector

import tensorflow as tf
import numpy as np
import pickle

from os import listdir
from os.path import isfile, join
from skimage.transform import resize


class Forward_perso:
  """Loads the caltech model in memory and forwards images through it"""
  
  def __init__(self, modelNb=3):
    """Load the tensorflow model of VGG16-GAP trained on caltech
    
    Keyword arguments:
      modelNb -- iteration of the model to consider
    """
    dataset_path = '/home/cuda/datasets/perso_db/'
    trainset_path = dataset_path+'train.pkl'
    testset_path  = dataset_path+'test.pkl'
    weight_path = '../caffe_layers_value.pickle'
    model_path = '../models/perso/model-'+str(modelNb)
    
    # load labels
    testset    = pickle.load( open(testset_path,  "rb") )
    self.label_dict = testset.keys()
    n_labels = len(self.label_dict)
    
    # Initialize some tensorflow variables
    batch_size = 1
    self.images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
    self.labels_tf = tf.placeholder( tf.int64, [None], name='labels')
    
    detector = Detector( weight_path, n_labels )
    c1,c2,c3,c4,conv5, self.conv6, gap, self.output = detector.inference( self.images_tf )
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
    best_pred   = preds.argmax( axis=1 ) if vis == -1 else [vis]

    classmap_vals = self.sess.run(
                        self.classmap,
                        feed_dict={
                          self.labels_tf: best_pred,
                          self.conv6: conv6_val
                        })
    
    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
    

    named_preds = [(self.label_dict[i], preds[0][i]) for i in preds_order[0]]

    
    return named_preds, np.array(classmap_vis)[0]







