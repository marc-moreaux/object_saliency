import tensorflow as tf
import cPickle as pickle
import os



class Model_type():
  VGG16         = 'VGG16'
  VGG16_CAM_W_S = 'VGG16_CAM_W_S'
  VGG16_CAM_S   = 'VGG16_CAM_S'
  VGG16_CAM7_S  = 'VGG16_CAM7_S'



class DB_type():
  PERSO      = 'PERSO'
  CALTECH256 = 'CALTECH256'
  ACTION40   = 'ACTION40'



class Model_params():
  def __init__(self, dataset, mod_type, optimizer, learning_rate, is_l2=False, l2_weight=.0005):
    """ 
      
      parameters:
      - dataset : one of DB_type
      - mod_type : one of Model_type
      - optimizer : rmsProp, adam or SDG
      - learning_rate : floating number
      - is_l2 : True or False
    """
    self.dataset   = dataset
    self.mod_type  = mod_type
    self.optimizer = optimizer
    self.lr        = learning_rate
    self.is_l2     = is_l2
    self.l2_weight = l2_weight
    self.paths     = self._set_paths()
    self.labels    = self._set_labels()
    self.n_labels  = len(self.labels)  
  
  
  #####################################
  ### Hidden functions for __init__
  def _set_paths(self):
    if self.dataset == DB_type.PERSO :
      paths = {
        "dataset"    : '/home/cuda/datasets/'+self.dataset.lower()+'/'             ,
        "trainset"   : '/home/cuda/datasets/'+self.dataset.lower()+'/'+'train.pkl' ,
        "testset"    : '/home/cuda/datasets/'+self.dataset.lower()+'/'+'test.pkl'  ,
        "save_model" : '../models/'+self.get_name()+'/model'                       ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    else :
      raise AttributeError('unknow parameter +' self.dataset)

    # Make <paths["save_model"]> a directory if not existing
    if not os.path.isdir(paths["save_model"]):
      os.mkdir(paths["save_model"])

    self.paths = paths
    return self.paths
  
  def _set_labels(self):
    if self.dataset == DB_type.PERSO :
      testset       = pickle.load( open(self.paths["testset"] , "rb") )
    self.labels = testset.keys()
    return self.labels
  
  
  #####################################
  ### General purpose functions
  def get_datasets(self):
    """ Return dictionnaries of pahts and the model"""
    dataset_path = '/home/cuda/datasets/'+self.dataset.lower()+'/'
  
    if self.dataset == DB_type.PERSO :
      trainset_path = dataset_path+'train.pkl'
      testset_path  = dataset_path+'test.pkl'
      trainset      = pickle.load( open(trainset_path,  "rb") )
      testset       = pickle.load( open(testset_path ,  "rb") )
  
  
    if self.dataset == DB_type.ACTION40 :
      print "NOT IMPLEMENTED YET"
  
    if self.dataset == DB_type.CALTECH256 :
      print "NOT IMPLEMENTED YET"
  
    return trainset, testset
  
  def get_optimizer(self, tf_learning_rate, loss_tf):
    # Switch on self.optimizer
    optimizer = {
      "adam"   : tf.train.AdamOptimizer(tf_learning_rate)    ,
      "rmsProp": tf.train.RMSPropOptimizer(tf_learning_rate) ,
      "SDG"    : tf.train.MomentumOptimizer(tf_learning_rate, .9) 
      }.get(self.optimizer)
    
    grads_and_vars = optimizer.compute_gradients( loss_tf )
    
    # Switch on self.mod_type
    grads_and_vars = {
      Model_type.VGG16 : [(gv[0], gv[1]) if ('fc6' in gv[1].name or 'fc7' in gv[1].name or 'fc8' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
      Model_type.VGG16_CAM_W_S : [(gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
      Model_type.VGG16_CAM_S   : [(gv[0], gv[1]) if ('fc6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
      Model_type.VGG16_CAM7_S  : [(gv[0], gv[1]) if ('fc6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
    }.get(self.mod_type)
  
    train_op = optimizer.apply_gradients( grads_and_vars )
    return optimizer, train_op
  
  def get_name(self):
    txt  = ""
    txt += self.dataset+'.'
    txt += self.mod_type+'.'
    txt += self.optimizer+(".%1.e"%self.lr).replace("0","")
    if self.is_l2 == True :
      txt += ".l2"
    return txt

  def get_loss(self, tf_output, tf_labels):
    tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( tf_output, tf_labels ))
    if self.is_l2 == True :
      weights_only  = [x for x in tf.trainable_variables() if x.name.endswith('W:0')]
      weight_decay  = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * self.l2_weight
      tf_loss    += weight_decay
    return tf_loss



if __name__ == '__main__':
  a = Model_params("PERSO", "VGG16_CAM_W_S", 'rmsProp', 0.001)
  a.get_name()
  a.paths["log_file"]