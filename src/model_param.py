import tensorflow as tf
import cPickle as pickle
import sys
import os


class Model_type:
  VGG16             = 'VGG16'
  VGG16_CAM3_W_S    = 'VGG16_CAM3_W_S'
  VGG16_CAM_W_S     = 'VGG16_CAM_W_S'
  VGG16_CAM3_S      = 'VGG16_CAM3_S'
  VGG16_CAM5_S      = 'VGG16_CAM5_S'
  VGG16_CAM7_S      = 'VGG16_CAM7_S'

  VGG16_CAM3b_S     = 'VGG16_CAM3b_S'
  VGG16_CAM3d_S     = 'VGG16_CAM3d_S'
  VGG16_CAM3e_S     = 'VGG16_CAM3e_S'
  VGG16_CAM5b_S     = 'VGG16_CAM5b_S'
  VGG16_CAM5d_S     = 'VGG16_CAM5d_S'
  VGG16_CAM5e_S     = 'VGG16_CAM5e_S'
  VGG16_CAM7b_S     = 'VGG16_CAM7b_S'
  VGG16_CAM7e_S     = 'VGG16_CAM7e_S'
  VGG16_CAM7d_S     = 'VGG16_CAM7d_S'
  VGG16_CAM9b_S     = 'VGG16_CAM9b_S'
  VGG16_CAM9d_S     = 'VGG16_CAM9d_S'

  VGG16_CAM5a7a_S   = 'VGG16_CAM5a7a_S'
  VGG16_CAM5b7a_S   = 'VGG16_CAM5b7a_S'

  VGG16_CAM3a5a7a_S = 'VGG16_CAM3a5a7a_S'

  VGG16P_CAM3_S     = 'VGG16P_CAM3_S'
  VGG16P_CAM5_S     = 'VGG16P_CAM5_S'
  VGG16P_CAM7_S     = 'VGG16P_CAM7_S'

class DB_type:
  PERSO      = 'PERSO'
  PERSO1     = 'PERSO1'
  CALTECH256 = 'CALTECH256'
  ACTION40   = 'ACTION40'
  VOC2012    = 'VOC2012'
  EXT_MNIST  = 'EXT_MNIST'

class Labels_names:
  PERSO      = "eyeglasses,gardening,people,playing_violin,applauding,fixing_a_car,fire-extinguisher,socks,cutting_vegetables,running,laptop-101,grand-piano-101,computer-mouse,brushing_teeth,computer-monitor,hamburger,head-phones,riding_a_bike,t-shirt,waving_hands,washing-machine,playing-card,frying-pan,backpack,toaster,fixing_a_bike,writing_on_a_book,microwave,coffee-mug,washing_dishes,pouring_liquid,knife,hot-dog,texting_message,mushroom,playing_guitar,cleaning_the_floor,wine-bottle,using_a_computer,phoning,jumping,ice-cream-cone,grapes,umbrella-101,smoking,drinking,watching_TV,teapot,refrigerator,computer-keyboard,taking_photos,cereal-box,fried-egg,spoon,reading,cooking,chess-board,dog,walking_the_dog,beer-mug,soda-can,holding_an_umbrella,bathtub".split(',')
  CALTECH256 = "ak47,american-flag,backpack,baseball-bat,baseball-glove,basketball-hoop,bat,bathtub,bear,beer-mug,billiards,binoculars,birdbath,blimp,bonsai-101,boom-box,bowling-ball,bowling-pin,boxing-glove,brain-101,breadmaker,buddha-101,bulldozer,butterfly,cactus,cake,calculator,camel,cannon,canoe,car-tire,cartman,cd,centipede,cereal-box,chandelier-101,chess-board,chimp,chopsticks,cockroach,coffee-mug,coffin,coin,comet,computer-keyboard,computer-monitor,computer-mouse,conch,cormorant,covered-wagon,cowboy-hat,crab-101,desk-globe,diamond-ring,dice,dog,dolphin-101,doorknob,drinking-straw,duck,dumb-bell,eiffel-tower,electric-guitar-101,elephant-101,elk,ewer-101,eyeglasses,fern,fighter-jet,fire-extinguisher,fire-hydrant,fire-truck,fireworks,flashlight,floppy-disk,football-helmet,french-horn,fried-egg,frisbee,frog,frying-pan,galaxy,gas-pump,giraffe,goat,golden-gate-bridge,goldfish,golf-ball,goose,gorilla,grand-piano-101,grapes,grasshopper,guitar-pick,hamburger,hammock,harmonica,harp,harpsichord,hawksbill-101,head-phones,helicopter-101,hibiscus,homer-simpson,horse,horseshoe-crab,hot-air-balloon,hot-dog,hot-tub,hourglass,house-fly,human-skeleton,hummingbird,ibis-101,ice-cream-cone,iguana,ipod,iris,jesus-christ,joy-stick,kangaroo-101,kayak,ketch-101,killer-whale,knife,ladder,laptop-101,lathe,leopards-101,license-plate,lightbulb,light-house,lightning,llama-101,mailbox,mandolin,mars,mattress,megaphone,menorah-101,microscope,microwave,minaret,minotaur,motorbikes-101,mountain-bike,mushroom,mussels,necktie,octopus,ostrich,owl,palm-pilot,palm-tree,paperclip,paper-shredder,pci-card,penguin,people,pez-dispenser,photocopier,picnic-table,playing-card,porcupine,pram,praying-mantis,pyramid,raccoon,radio-telescope,rainbow,refrigerator,revolver-101,rifle,rotary-phone,roulette-wheel,saddle,saturn,school-bus,scorpion-101,screwdriver,segway,self-propelled-lawn-mower,sextant,sheet-music,skateboard,skunk,skyscraper,smokestack,snail,snake,sneaker,snowmobile,soccer-ball,socks,soda-can,spaghetti,speed-boat,spider,spoon,stained-glass,starfish-101,steering-wheel,stirrups,sunflower-101,superman,sushi,swan,swiss-army-knife,sword,syringe,tambourine,teapot,teddy-bear,teepee,telephone-box,tennis-ball,tennis-court,tennis-racket,theodolite,toaster,tomato,tombstone,top-hat,touring-bike,tower-pisa,traffic-light,treadmill,triceratops,tricycle,trilobite-101,tripod,t-shirt,tuning-fork,tweezer,umbrella-101,unicorn,vcr,video-projector,washing-machine,watch-101,waterfall,watermelon,welding-mask,wheelbarrow,windmill,wine-bottle,xylophone,yarmulke,yo-yo,zebra,airplanes-101,car-side-101,faces-easy-101,greyhound,tennis-shoes,toad,clutter".split(",")
  ACTION40   = []
  VOC2012    = "aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor".split(",")
  EXT_MNIST  = "0,1,2,3,4,5,6,7,8,9".split(',')


class Model_params:
  def __init__(self, dataset, mod_type, optimizer, learning_rate, l2_weight=.0, l2_gap=0, l1_gap=0):
    """ 
      
      parameters:
      - dataset : one of DB_type
      - mod_type : one of Model_type
      - optimizer : rmsProp, adam or SDG
      - learning_rate : floating number
      - l2_weight : value of the l2 on the weights
      - l2_gap : value of the l2 on the GAP
    """
    self.dataset   = dataset
    self.mod_type  = mod_type
    self.optimizer = optimizer
    self.lr        = learning_rate
    self.l2_weight = l2_weight
    self.l2_gap    = l2_gap
    self.l1_gap    = l1_gap
    self.paths     = self._set_paths()
    self.labels    = self._set_labels()
    self.n_labels  = len(self.labels)
  
  
  #####################################
  ### Hidden functions for __init__
  def _set_paths(self):
    if self.dataset == DB_type.PERSO :
      if os.path.isdir("/home/cuda/datasets/perso/"):
        sys.path.append("/home/cuda/datasets/perso")
        from perso_getter import get_batch
        self.get_batch = get_batch
      else :
        print "WARNING: could not import perso_getter"
      paths = {
        "save_model" : '../models/'+self.dataset+'/'+self.get_name()+'/model' ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    if self.dataset == DB_type.PERSO1 :
      if os.path.isdir("/home/cuda/datasets/perso1/"):
        sys.path.append("/home/cuda/datasets/perso1")
        import perso1_getter
        self.get_batch = perso1_getter.get_batch
        self.labels = perso1_getter.getter_inst.classes
      else :
        print "WARNING: could not import perso_getter"
      paths = {
        "save_model" : '../models/'+self.dataset+'/'+self.get_name()+'/model' ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    elif self.dataset == DB_type.VOC2012 :
      if os.path.isdir("/home/cuda/datasets/VOC2012/"):
        sys.path.append("/home/cuda/datasets/VOC2012")
        from voc2012_getter import get_batch
        self.get_batch = get_batch
      else :
        print "WARNING: could not import VOC2012_getter"
      paths = {
        "save_model" : '../models/'+self.dataset+'/'+self.get_name()+'/model' ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    elif self.dataset == DB_type.CALTECH256 :
      if os.path.isdir("/home/cuda/datasets/caltech256/"):
        sys.path.append("/home/cuda/datasets/caltech256/")
        from caltech256_getter import get_batch
        self.get_batch = get_batch
      else :
        print "WARNING: could not import caltech256_getter"
      paths = {
        "save_model" : '../models/'+self.dataset+'/'+self.get_name()+'/model' ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    elif self.dataset == DB_type.EXT_MNIST :
      if os.path.isdir("/home/cuda/datasets/mnist/"):
        sys.path.append("/home/cuda/datasets/mnist/")
        import mnist_getter
        self.get_batch  = mnist_getter.get_batch_224
        self.epoch_size = mnist_getter.EPOCH_SIZE
      else :
        print "WARNING: could not import mnist_getter"
      paths = {
        "save_model" : '../models/'+self.dataset+'/'+self.get_name()+'/model' ,
        "log_file"   : "../results/"+self.get_name()+".txt"
      }
    else :
      raise AttributeError('dataset should be one of DB_type '+ self.dataset)
   
    # Make <paths["save_model"]> a directory if not existing
    save_dir = "/".join(paths["save_model"].split('/')[:-1])
    if not os.path.isdir(save_dir):
      print "Created a new directory at : "+save_dir
      os.makedirs(save_dir)
     
    self.paths = paths
    return self.paths
  
  def _set_labels(self):
    
    if   self.dataset == DB_type.PERSO :
      self.labels = Labels_names.PERSO
      return self.labels
    if   self.dataset == DB_type.PERSO1 :
      return self.labels
    elif self.dataset == DB_type.VOC2012 :
      self.labels = Labels_names.VOC2012
      return self.labels
    elif self.dataset == DB_type.CALTECH256 :
      self.labels = Labels_names.CALTECH256
      return self.labels
    elif self.dataset == DB_type.EXT_MNIST :
      self.labels = Labels_names.EXT_MNIST
      return self.labels
    else :
      raise NotImplementedError
      
      
  
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
      raise NotImplementedError
  
    if self.dataset == DB_type.CALTECH256 :
      print "NOT IMPLEMENTED YET"
      raise NotImplementedError
  
    return trainset, testset
  
  def get_optimizer(self, tf_learning_rate, loss_tf, label_idx=-1):
    """
    Returns the weights to 

    Params:
    tf_learning_rate -- Learning rate tensor
    loss_tf -- The loss tensor
    label_idx -- The label idx if we want a training on a single GAP
    """
    # Switch on self.optimizer
    optimizer = {
      "adam"   : tf.train.AdamOptimizer(tf_learning_rate)    ,
      "rmsProp": tf.train.RMSPropOptimizer(tf_learning_rate) ,
      "SDG"    : tf.train.MomentumOptimizer(tf_learning_rate, .9) 
      }.get(self.optimizer)
    
    grads_and_vars = optimizer.compute_gradients( loss_tf )
    print [gv[1].name for gv in grads_and_vars]
    
    # set <grads_and_vars>
    if   self.mod_type == Model_type.VGG16:
      grads_and_vars = [(gv[0], gv[1]) if ('fc6' in gv[1].name or 'fc7' in gv[1].name or 'fc8' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars]
    elif "_W_S" in self.mod_type: # == Model_type.VGG16_CAM3_W_S:
      grads_and_vars = [(gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars]
    else :
      grads_and_vars = [(gv[0], gv[1]) if ('conv6' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars]

    # Only modify the desired idx, nothing else changes in the network
    if label_idx >= 0:
      grads_and_vars = [(gv[0], gv[1]) if ('conv6' in gv[1].name) else (gv[0]*0, gv[1]) for gv in grads_and_vars]
  
    train_op = optimizer.apply_gradients( grads_and_vars )
    return optimizer, train_op
  
  def get_name(self):
    txt  = ""
    txt += self.dataset+'.'
    txt += self.mod_type+'.'
    txt += self.optimizer+(".%1.e"%self.lr).replace("0","")+'.'
    if self.l2_weight > 0 :
      txt += "l2w"+(".%1.e"%self.l2_weight).replace("0","")+'.'
    if self.l2_gap > 0 :
      txt += "l2gap"+(".%1.e"%self.l2_gap).replace("0","")+'.'
    if self.l1_gap > 0 :
      txt += "l1gap"+(".%1.e"%self.l1_gap).replace("0","")+'.'
    return txt[:-1]
  
  def get_loss(self, tf_output, tf_labels, conv6=None):
    """
    Returns the loss of a network

    Params:
    tf_output -- The output tensor 
    tf_labels -- The label tensor 
    conv6 -- the conv6 layer (the one which is going to be Meaned)
    """
    if label_idx == -1:
      tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( tf_output, tf_labels ))
    else 
      tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( tf_output, tf_labels ))
    if self.l2_weight > 0 :
      weights_only  = [x for x in tf.trainable_variables() if x.name.endswith('W:0') and "conv6" not in x.name]
      weight_decay  = tf.reduce_mean(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * self.l2_weight
      tf_loss      += weight_decay
    if self.l2_gap > 0:
      if conv6 == None:
        print "WARNING : found no conv6 vars layers on model..."
      if type(conv6) != type(list()):
        conv6 = [conv6]
      print "Applying the conv6 loss (l2)"+str(conv6)
      tf_loss += tf.reduce_mean(tf.pack([tf.nn.l2_loss(x) for x in conv6])) * self.l2_gap
    if self.l1_gap > 0:
      if conv6 == None:
        print "WARNING : found no conv6 vars layers on model..."
      if type(conv6) != type(list()):
        conv6 = [conv6]
      print "Applying the conv6 loss (l1)"+str(conv6)
      tf_loss += tf.reduce_mean(tf.pack([tf.abs(x) for x in conv6])) * self.l1_gap
    return tf_loss


if __name__ == '__main__':
  a = Model_params("VOC2012", "VGG16_CAM_S", 'rmsProp', 0.001)
  a.get_name()
  a.paths["log_file"]



