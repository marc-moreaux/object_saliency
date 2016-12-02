# from util import load_image
from PIL import Image
from detector import Detector
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import listdir
from os.path import isfile, join



# Declare paths
model_label_path = "../models/caltech_labels.pkl"
weight_path = '../caffe_layers_value.pickle'
model_path = '../models/caltech256/model-3'
jpg_folder_path = "../img_test"

imgPath = [join(jpg_folder_path, f) 
           for f in listdir(jpg_folder_path) 
             if isfile(join(jpg_folder_path, f))
             and ('jpg' or 'jpeg' or 'png') in f ]

# load the caltech model
with open(model_label_path, 'rb') as f:
  label_dict = pickle.load(f)

n_labels = len(label_dict)
batch_size = 1

images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector( weight_path, n_labels )
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference( images_tf )
classmap = detector.get_classmap( labels_tf, conv6 )

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore( sess, model_path )


# Fast forward images & save them
for idx,imgP in enumerate(imgPath):
    img = Image.open(imgP)
    img = img.resize([224,224])
    img = np.array(img)
    img = img.reshape(1,224,224,3)
    
    feed_dict = {}
    feed_dict[images_tf] = img
    conv6_val, output_val = sess.run([conv6, output],feed_dict=feed_dict)
    label_predictions = output_val.argmax( axis=1 )
    label_ordered     = output_val.argsort( axis=1 )[:,::-1]
    
    classmap_vals = sess.run(
            classmap,
            feed_dict={
                labels_tf: label_predictions,
                conv6: conv6_val
                })
    
    classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap_vals)
    
    for vis, ori, ori_path, l_class in zip(classmap_vis, img, imgP, label_predictions):
        _str = ''
        for _a in label_ordered[0,:5]:
            _str+=label_dict[_a]+', '
        plt.title(_str)
        plt.imshow( ori )
        plt.imshow( vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
        # plt.show()
        plt.savefig('../img_results/img'+str(idx)+'.png')


