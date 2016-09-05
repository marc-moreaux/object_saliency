from util import load_image
import tensorflow as tf
import cPickle as pickle
import numpy as np
import time
import my_images

from detector import Detector
import os

###################################
### Variables initialisation
###################################
weight_path = '../caffe_layers_value.pickle'
model_path = '../models/perso/'
pretrained_model_path = '../models/perso/model-perso-0'
n_epochs = 10000
init_learning_rate = 0.01
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 50

dataset_path = '/home/cuda/datasets/perso_db/'
trainset_path = dataset_path+'train.pkl'
testset_path  = dataset_path+'test.pkl'


###################################
### Load dataset's paths
###################################
trainset   = pickle.load( open(trainset_path, "rb") )
testset    = pickle.load( open(testset_path,  "rb") )
n_labels = len(testset.keys())

def get_batch(mdict, batch_size=10):
  """load the dict images as batch and perform some 
  reshaping operations on them. Also, the images are
  returned such that there is an equal number of 
  images per classes
  
  return:
  images -- array of <batch_size> images
  labels -- array of <batch_size> labels
  """
  # Some variables
  max_len   = max(map(len, mdict.values()))
   
  # We'll return n_keys*max(n_values) indices (fake array) 
  idxs = list(np.random.permutation(len(mdict.keys())*max_len))
  idxs = [ [idxs.pop() for _ in range(batch_size)] \
                      for _ in range(len(idxs)/batch_size)  ]
  
  # Yield batch of images
  for batch_idxs in idxs:
    img_list   = []
    label_list = []
    for idx in batch_idxs:
      key  = idx/max_len
      label_list.append(key)
      key  = mdict.keys()[key]
      val  = idx%max_len # val in fake array
      path = mdict[key][val%len(mdict[key])]
      tmp = my_images.load_augmented_image(path)
      img_list.append(tmp)
   
    yield np.stack(img_list), np.stack(label_list)





###################################
### Begin with Tensorflow :D
###################################
learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(weight_path, n_labels)
p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( output, labels_tf ))
weights_only = [x for x in tf.trainable_variables() if x.name.endswith('W:0')]
weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
loss_tf += weight_decay
sess = tf.InteractiveSession()
saver = tf.train.Saver( max_to_keep=50 )
optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = [(gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]) for gv in grads_and_vars]
#grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )
tf.initialize_all_variables().run()
# if pretrained_model_path:
#     print("Pretrained")
#     saver.restore(sess, pretrained_model_path)


f_log = open('../results/log.perso.txt', 'w')

batch_images, batch_labels = get_batch(trainset,batch_size).next()
output.eval(feed_dict={learning_rate: init_learning_rate, images_tf: batch_images})

iterations = 0
loss_list = []
n_iter = len(trainset.keys())*max(map(len, trainset.values()))
for epoch in range(n_epochs):
  ##################################
  ### Training 
  ##################################
  for batch_images, batch_labels in get_batch(trainset,batch_size):
    _, loss_val, output_val = sess.run(
          [train_op, loss_tf, output],
          feed_dict={
              learning_rate: init_learning_rate,
              images_tf: batch_images,
              labels_tf: batch_labels
              })
    loss_list.append( loss_val )
    
    iterations += 1
    if iterations % 5 == 0:
      label_predictions = output_val.argmax(axis=1)
      acc = (label_predictions == batch_labels).sum()
      print("======================================")
      print("Epoch", epoch, "Iteration", iterations)
      print("Processed", iterations*batch_size, '/', n_iter)
      print("Accuracy:", acc, '/', len(batch_labels))
      print("Training Loss:", np.mean(loss_list))
      print("\n")
      loss_list = []
   
  ##################################
  ### Statistics on testset
  ##################################
  n_correct = 0
  n_data = 0
  for batch_images, batch_labels in get_batch(testset,batch_size):
    output_vals = sess.run(
            output,
            feed_dict={images_tf:batch_images})
    
    label_predictions = output_vals.argmax(axis=1)
    acc = (label_predictions == batch_labels).sum()
    
    n_correct += acc
    n_data += len(batch_labels)
    
  test_acc = n_correct / float(n_data)
  f_log.write('epoch:'+str(epoch)+'\tacc:'+str(test_acc) + '\n')
  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
  print('epoch:'+str(epoch)+'\tacc:'+str(test_acc) + '\n')
  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
  
  saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)
  
  init_learning_rate *= 0.99




###################################
### Plot the amazing dataset
###################################
import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3,5)
for ax in [ax for l in axs for ax in l ]:
  ax.imshow(my_images.load_augmented_image('/home/cuda/Pictures/creepy_max.png'))

def animate(data):
  batch_img,batch_labels = data
  for idx,ax in enumerate([ax for l in axs for ax in l ]):
    ax.imshow(batch_img[idx])
    ax.set_title(testset.keys()[batch_labels[idx]])

ani = animation.FuncAnimation(fig, animate, get_batch(testset,15), interval=1000)
plt.show()


###################################
### Stats on loading time
###################################
import time
t_start = time.time()
count = 0
for a in get_batch(testset,15):
  count += 15
  pass

t_end = time.time()
print "%.3f sec to load %d images"%(t_end-t_start, count)
print "=> %.3fsec to load 70 imgs"%(t_end-t_start/count*70)
