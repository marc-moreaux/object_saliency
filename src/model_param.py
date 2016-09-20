import tensorflow as tf
import cPickle as pickle
import os



class Model_type():
  VGG16         = 'VGG16'
  VGG16_CAM_W_S = 'VGG16_CAM_W_S'
  VGG16_CAM_S   = 'VGG16_CAM_S'
  VGG16_CAM5_S  = 'VGG16_CAM5_S'
  VGG16_CAM7_S  = 'VGG16_CAM7_S'
  VGG16P_CAM3_S = 'VGG16P_CAM3_S'
  VGG16P_CAM5_S = 'VGG16P_CAM5_S'

class DB_type():
  PERSO      = 'PERSO'
  CALTECH256 = 'CALTECH256'
  ACTION40   = 'ACTION40'

class Labels_names():
  PERSO      = ['eyeglasses', 'gardening', 'people', 'playing_violin', 'applauding', 'fixing_a_car', 'fire-extinguisher', 'socks', 'cutting_vegetables', 'running', 'laptop-101', 'grand-piano-101', 'computer-mouse', 'brushing_teeth', 'computer-monitor', 'hamburger', 'head-phones', 'riding_a_bike', 't-shirt', 'waving_hands', 'washing-machine', 'playing-card', 'frying-pan', 'backpack', 'toaster', 'fixing_a_bike', 'writing_on_a_book', 'microwave', 'coffee-mug', 'washing_dishes', 'pouring_liquid', 'knife', 'hot-dog', 'texting_message', 'mushroom', 'playing_guitar', 'cleaning_the_floor', 'wine-bottle', 'using_a_computer', 'phoning', 'jumping', 'ice-cream-cone', 'grapes', 'umbrella-101', 'smoking', 'drinking', 'watching_TV', 'teapot', 'refrigerator', 'computer-keyboard', 'taking_photos', 'cereal-box', 'fried-egg', 'spoon', 'reading', 'cooking', 'chess-board', 'dog', 'walking_the_dog', 'beer-mug', 'soda-can', 'holding_an_umbrella', 'bathtub']
  CALTECH256 = []
  ACTION40   = []




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
        "save_model" : '../models/'+self.dataset+'/'+self.get_name()+'/model'      ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    else :
      raise AttributeError('dataset should be one of DB_type '+ self.dataset)

    # Make <paths["save_model"]> a directory if not existing
    save_dir = "/".join(paths["save_model"].split('/')[:-1])
    if not os.path.isdir(save_dir):
      print "Created a new directory at : "+save_dir
      os.mkdir(save_dir)

    self.paths = paths
    return self.paths
  
  def _set_labels(self):
    if os.path.isfile(self.paths["testset"]):
      if self.dataset == DB_type.PERSO :
        testset  = pickle.load( open(self.paths["testset"] , "rb") )
      self.labels = testset.keys()
      return self.labels
    else :
      print "There is no %s on this computer"%self.paths["testset"]
      self.labels = Labels_names.PERSO
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
      Model_type.VGG16_CAM5_S  : [(gv[0], gv[1]) if ('fc6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
      Model_type.VGG16_CAM7_S  : [(gv[0], gv[1]) if ('fc6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
      Model_type.VGG16P_CAM3_S : [(gv[0], gv[1]) if ('fc6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
      Model_type.VGG16P_CAM5_S : [(gv[0], gv[1]) if ('fc6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars] ,
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
