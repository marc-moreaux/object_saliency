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



f_sotfmax = lambda arr : (np.exp(arr).transpose() / np.sum(np.exp(arr), axis=1)).transpose()


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
    self.images_tf = tf.placeholder( tf.float32, [None, None, None, 3], name="images")
    self.labels_tf = tf.placeholder( tf.int64, [None], name='labels')
    
    detector = Detector(mod_param)
    c1,c2,c3,c4,conv5, self.conv6, self.gap, self.output = detector.inference( self.images_tf )
    self.detector = detector
    
    self.sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore( self.sess, model_path )

    if re.match(".*CAM.*_W_S", self.mod_param.mod_type):
      self.classmap = self.detector.get_classmap( self.labels_tf, self.conv6 )
    
  
  def _get_bounding_boxes(self, imgs, summed_viz, threshold_value=.7):
    """
    Compute the bounding box around saliency <summmed_viz>
    for each classes and each elements of batch <summed_vis>
    
    Params:
    summed_viz -- 
    threshold_value -- only consider <summed_viz> values over this threshold
                       as a percentage of itself
    
    Returns:
    ls,ts,rs,bs -- lefts, tops, rights, bottoms 
                   for each classes of each batchs
    """
    self.viz = summed_viz # for debug
    viz = summed_viz
    n_batchs  = viz.shape[ 0]
    n_classes = viz.shape[-1]
    
    # viz.shape (100,14,14,20) => (14,14,100,20)
    viz = viz.swapaxes(0,2); viz = viz.swapaxes(0,1)
  
    # Normalize <viz>, image per image (to be in range [-1,1])
    viz = viz / np.max(np.abs(viz), axis=(0,1))
    viz = (viz+1)/2 # range[0,1]
  
    # Resize each summed_viz to its original size (size of input image)
    if viz.shape[:2] != imgs.shape[1:3]:
      viz = np.array(
              [ skimage.transform.resize(viz[:,:,idx], imgs[idx].shape[1:3])
                for idx in range(len(imgs))
                if viz.shape[0] != imgs.shape[1]
              ] )
      viz = viz.swapaxes(0,2); viz = viz.swapaxes(0,1)
  
    # Threshold <viz>s to keep values over 70% of its max values
    m_max = threshold_value * viz.max(axis=(0,1))
    viz = viz * (m_max < viz)
  
    # We want a 2d boundind box, so project threshold in xs and ys
    xxs = viz.sum(axis=0)
    yys = viz.sum(axis=1)
  
    # Get some non-thresholded values (left, top... of bounding boxes)
    get_lefts   = lambda b_id, c_idx: xxs[:,b_id,c_idx].nonzero()[0][ 0]
    get_tops    = lambda b_id, c_idx: yys[:,b_id,c_idx].nonzero()[0][-1]
    get_rights  = lambda b_id, c_idx: xxs[:,b_id,c_idx].nonzero()[0][-1]
    get_bottoms = lambda b_id, c_idx: yys[:,b_id,c_idx].nonzero()[0][ 0]

    # Debug
    # def get_lefts (b_id, c_idx): 
    #   print xxs[:,b_id,c_idx].nonzero()
    #   xxs[:,b_id,c_idx].nonzero()[0][ 0]
  
    # Build the 2d array with first or lasts positions of zeros
    # INNER FUNCTION
    def _get_border_array(f_border=get_lefts):
      return np.array(
               [ map(f_border, [b_idx]*n_classes, range(n_classes))
                 for b_idx in range(n_batchs) ]
               )
  
    lefts   = _get_border_array(get_lefts)
    tops    = _get_border_array(get_tops)
    rights  = _get_border_array(get_rights)
    bottoms = _get_border_array(get_bottoms)
  
    return lefts, tops, rights, bottoms
  
  def _visualize_cam_w_s(self, imgs, names="2DPositions"):
    """
    Returns a list with the named_predictions (or more depending on <names>)
    Also returns a saliency map of the imgs
    
    Params:
    imgs -- batch of images, the ones to analize
    names -- <named_preds> is going to change accordingly
             "Predictions" only retrieves class name and its probability
             "2DPositions" adds x and y center coordinates to <named_preds>
             "BoundingBoxes" adds left, top, right, bottom to <named_preds>
    """
    feed_dict = {}
    feed_dict[self.images_tf] = imgs
    conv6_val, output_val = self.sess.run([self.conv6, self.output],feed_dict=feed_dict)
    preds = output_val
    preds_order = preds.argsort( axis=1 )[:,::-1]
    best_preds  = preds.argmax(  axis=1 )


    if names == "Predictions":
      self.classmap = self.detector.get_classmap( self.labels_tf, self.conv6, list(imgs.shape[1:3]), eval_all=False )
      classmap_vals = self.sess.run(
                          self.classmap,
                          feed_dict={
                            self.labels_tf: best_preds,
                            self.conv6: conv6_val
                          })
      
      classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
      self.classmap_vis = classmap_vis # for Debug
      named_preds  = map(self.to_named_pred, preds)

    if names == "BoundingBoxes":
      softmaxs = f_sotfmax(preds)
      self.classmap = self.detector.get_classmap( self.labels_tf, self.conv6, list(imgs.shape[1:3]), eval_all=True )
      classmap_vals = self.sess.run(
                          self.classmap,
                          feed_dict={
                            self.labels_tf: best_preds*0-1,
                            self.conv6: conv6_val
                          })
      classmap_vis = classmap_vals
      ls,ts,rs,bs = self._get_bounding_boxes(imgs, classmap_vals, threshold_value=.8)
      named_preds = map(self.to_named_pred, softmaxs, ls, ts, rs, bs)


    return named_preds, np.array(classmap_vis)[0]
  

  def _visualize_cam_s(self, imgs, summed_filters=True, names="2DPositions"):
    """
    Returns a list with the named_predictions (or more depending on <names>)
    Also returns a saliency map of the imgs
    
    Params:
    imgs -- batch of images, the ones to analize
    summed_filters -- Boolean affecting return values
    names -- <named_preds> is going to change accordingly
             "Predictions" only retrieves class name and its probability
             "2DPositions" adds x and y center coordinates to <named_preds>
             "BoundingBoxes" adds left, top, right, bottom to <named_preds>
    """
    feed_dict = {}
    feed_dict[self.images_tf] = imgs
    conv6_val, output_val = self.sess.run([self.conv6, self.output],feed_dict=feed_dict)
    preds = output_val

    # <preds> to confidence (to  softmax)
    softmaxs = f_sotfmax(preds)
  
    # Computes the position of the max per axis [-1;1]
    summed_viz = self.summed_vis(conv6_val)
    if names == "2DPositions":
      max_pos = lambda arr: (arr.argmax(axis=1) / float(arr.shape[-2]) *2)-1
      xxs     = max_pos( summed_viz.sum(axis=1) )
      yys     = max_pos( summed_viz.sum(axis=2) )
      named_preds = map(self.to_named_pred, preds, xxs, yys)
  
    # Computes the bounding boxes coordinates of classes
    if names == "BoundingBoxes":
      ls,ts,rs,bs = self._get_bounding_boxes(imgs, summed_viz)
      named_preds = map(self.to_named_pred, softmaxs, ls, ts, rs, bs)
  
    if summed_filters == True:
      conv6_val = summed_viz 
  
    return named_preds, conv6_val
  
  def _check_image(self, img):
    if img.shape[-1] == 1:
      img = skimage.color.greytorgb(img)
    if img.shape[-3:] != (224,224,3):
      print "WARNING: FUCKED UP SHAPE, not (224*224*3) !!"
    if len(img.shape) == 3:
      img = img.reshape([1,]+list(img.shape))
    return img
  
  def to_named_pred(self, preds, xxs=None, yys=None, xxws=None, yyhs=None):
    """
    Retrieves a list with :
    [("CLASS_NAME", pred, -1, -1, -1, -1), ..]
    [("CLASS_NAME", pred,  x,  y, -1, -1), ..] where x, y are the ped's centers
    [("CLASS_NAME", pred,  l,  t,  r,  b), ..] with left, top, right, bottom of bounding box
  
    Params:
    preds -- 
    xxs  -- predicted centered xs's positions of classes
            or predicted left's positions of classes
    yys  -- predicted centered ys's positions of classes
            or predicted bottom's positions of classes
    xxws -- predicted right's positions of classes
    yyhs -- predicted top's positions of classes
    """
    if xxs == None :
      named_preds = [(self.labels[idx], p,-1,-1,-1,-1) for idx,p in enumerate(preds)]
    elif xxws == None:
      named_preds = [(self.labels[idx], p, x, y,-1,-1) for idx,(p,x,y) in enumerate(zip(preds, xxs, yys))]
    else :
      named_preds = [(self.labels[idx], p, l, t, r, b) for idx,(p,l,t,r,b) in enumerate(zip(preds, xxs, yys, xxws, yyhs))]
    return named_preds
  
  def forward_image(self, img, visualize=False, do_resize=True, names="2DPositions"):
    """
    Retrieve prediction's text and visualisation
    
    Keyword arguments:
    img -- image to feed forward as numpy array
    vis -- saliency map to visualize (-1 for best prediction)
    names -- <named_preds> is going to change accordingly
             "Predictions" only retrieves class name and its probability
             "2DPositions" adds x and y center coordinates to <named_preds>
             "BoundingBoxes" adds left, top, right, bottom to <named_preds>
    """
    # Reshape image
    if do_resize == True:
      # crop from center 
      short_edge = min(img.shape[:2]) 
      yy = int((img.shape[0] - short_edge) / 2)
      xx = int((img.shape[1] - short_edge) / 2)
      img = img[yy: yy + short_edge, xx: xx + short_edge]
  
      # resize
      img = resize(img, [224,224])
      img = img.reshape(1,224,224,3)
    tup = self.forward_images(img, visualize, names=names)
    if visualize == False:
      return tup[0]
    return tup[0][0], tup[1]
  
  def forward_images(self, imgs, visualize=False, names="2DPositions"):
    """
    Retrieve prediction's text and visualisation
    
    Params:
    imgs -- image to feed forward as numpy array
    vis -- saliency map to visualize (-1 for best prediction)
    names -- <named_preds> is going to change accordingly
             "Predictions" only retrieves class name and its probability
             "2DPositions" adds x and y center coordinates to <named_preds>
             "BoundingBoxes" adds left, top, right, bottom to <named_preds>
    
    returns:
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
    
    if re.match(".*CAM.*_W_S", self.mod_param.mod_type):
      return self._visualize_cam_w_s(imgs, names=names)
  
    else :
      return self._visualize_cam_s(imgs, names=names)
  
  def summed_vis(self, vis):
    # Retrieve model's CAM's properties 
    # eg: 'CALTECH256.VGG16_CAM5b_S.rmsProp.1e-5' => '5b'
    name = self.mod_param.get_name()
    suffix = [ s for s in name.split("_") if "CAM" in s][0][3:]
    if len(suffix) < 1:
      suffix = '3'
    if len(suffix) < 2:
      suffix += 'a'
      vis = [vis]
    
    filter_sizes = []
    for idx in range(0, 2, len(suffix)):
      alphabet_index = ord(suffix[idx+1]) - ord('a') + 1
      filter_sizes.append((int(suffix[idx]), alphabet_index))
    
    # Concatenate all the CAMs to <end_viz>
    end_viz = None
    for idx,filter_size in enumerate(filter_sizes):
      cam_size  = filter_size[0]
      n_cam     = filter_size[1]
      new_shape = list(vis[idx].shape[:-1])+[self.n_labels]+[n_cam]
      m_vis     = vis[idx].reshape(new_shape)
      m_vis     = m_vis.sum(axis=len(new_shape)-1)
      end_viz   = m_vis if end_viz == None else sum(end_viz, m_vis)
    
    return end_viz



if __name__ == '__main__':
  import sys; sys.path.append("/home/cuda/datasets/VOC2012")
  from voc2012_getter import get_batch
  a = get_batch('test')
  batch = a.next()
  imgs = batch[0]
  
  mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-7)
  model  = Forward_model(mod_param, 19)
  named_preds,viz = model.forward_images(imgs, True)







