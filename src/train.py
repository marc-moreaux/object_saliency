# train from util import load_image
import tensorflow as tf
import cPickle as pickle
import numpy as np
import time
import my_images

import os
from detector import Detector
import model_param

###################################
### Variables initialisation
###################################
# tensorflow 
n_epochs = 20
momentum = 0.9
batch_size = 10









###################################
### Begin with Tensorflow :D
###################################

def print_train(epoch, iterations, accuracy, train_loss, batch_size, n_iter, log_file):
  # log_file.write('epoch:'+str(epoch)+' iter:'+str(iterations)+' loss:'+str(train_loss) + '\n')
  log_file.write('epoch:%2d - iter:%3d - loss:%2.7f\n'%(epoch, iterations, train_loss))
  print "======================================"
  print "Epoch", epoch, "Iteration", iterations
  print "Processed", iterations*batch_size, '/', n_iter
  print "Accuracy:", accuracy, '/', batch_size
  print "Training Loss:", train_loss
  print " "

def print_test(epoch, test_accuracy, log_file):
  # log_file.write('epoch:'+str(epoch)+' acc:'+str(test_accuracy) + '\n')
  log_file.write('epoch:%2d - acc:%.3f\n'%(epoch, test_accuracy))
  log_file.write('---------------\n')
  print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
  print 'epoch:'+str(epoch)+'\tacc:'+str(test_accuracy)
  print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"


def my_train(mod_param, n_epochs=15, mod_param_old=None, param_idx_old=5, single_lbl_learning=-1):
  with open(mod_param.paths["log_file"], 'w',0) as log_file:
    # Tensorflow vars
    log_file =open(mod_param.paths["log_file"], 'w',0) 
    tf.reset_default_graph()
    tf_learning_rate = tf.placeholder( tf.float32, [])
    images_tf = tf.placeholder( tf.float32, [None, None, None, 3], name="images")
    labels_tf = tf.placeholder( tf.int64, [None], name='labels')
    detector  = Detector(mod_param)
    p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)
    
    # Init Tensorflow
    sess     = tf.InteractiveSession()
    saver    = tf.train.Saver( max_to_keep=50 )
    if mod_param_old != None:
      print "Debug: Restoring an other version of the net"
      saver.restore(sess, mod_param_old.paths["save_model"]+"-"+str(param_idx_old))
    
    
    # train vars
    model_path = mod_param.paths["save_model"]
    n_labels   = mod_param.n_labels
    loss_tf               = mod_param.get_loss(output, labels_tf, conv6)
    optimizer, train_op   = mod_param.get_optimizer(tf_learning_rate, loss_tf,single_lbl_learning)
    
    # Save classes names (for varing classes classifiers)
    with open(mod_param.paths["save_model"]+"-classes.txt", "w") as file:
      file.write(",".join(mod_param.labels))
    
    # loop variables
    tf.initialize_all_variables().run()
    loss_list = []
    iterations = 0
    n_iter = 600*20
    log_file = open(mod_param.paths["log_file"], 'w',0)
    for epoch in range(n_epochs):
      ##################################
      ### Training 
      ##################################
      iterations = 0
      for batch_images, batch_labels in mod_param.get_batch('train',batch_size):
        _, loss_val, output_val = sess.run(
              [train_op, loss_tf, output],
              feed_dict={
                  tf_learning_rate: mod_param.lr,
                  images_tf: batch_images,
                  labels_tf: batch_labels
                  })
        loss_list.append( loss_val )
        print loss_val
        
        iterations += 1
        if iterations % 5 == 0:
          label_predictions = output_val.argmax(axis=1)
          accuracy = (label_predictions == batch_labels).sum()
          print_train(epoch, iterations, accuracy, np.mean(loss_list), batch_size, n_iter, log_file)
          loss_list = []
       
      ##################################
      ### Statistics on testset
      ##################################
      n_correct = 0
      n_data = 0
      for batch_images, batch_labels in mod_param.get_batch('test',batch_size):
        output_vals = sess.run(
                output,
                feed_dict={images_tf:batch_images})
        
        label_predictions = output_vals.argmax(axis=1)
        acc = (label_predictions == batch_labels).sum()
        n_correct += acc
        n_data += len(batch_labels)
        
      test_accuracy = n_correct / float(n_data)
      print_test(epoch, test_accuracy, log_file)
      
      ##################################
      ### Save weights and go to 
      ###   next batch
      saver.save( sess, mod_param.paths["save_model"], global_step=epoch)
      mod_param.lr *= 0.9







mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM_W_S", 'rmsProp',   8e-6, 5e-5)
my_train(mod_param, 20)


# mod_param  = model_param.Model_params('PERSO1',   "VGG16_CAM5_S", 'rmsProp', 1e-5, 5e-5, 2e-7)
# my_train(mod_param, 20)

mod_param = model_param.Model_params('PERSO1',   "VGG16_CAM5_S", 'rmsProp', 1e-5, 5e-5, 2e-7)
mod_param_old = model_param.Model_params('PERSO1',   "VGG16_CAM5_S", 'rmsProp', 1e-5, 5e-5, 2e-7)
my_train(mod_param, 20, mod_param_old, 4, 2)


# mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM_W_S", 'rmsProp',   8e-6, 5e-5)
# my_train(mod_param, 20)

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM3b_S", 'rmsProp',   1e-5, 5e-5, 0, 1e-7)
my_train(mod_param, 20)

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-6, 5e-5, 1e-8)
my_train(mod_param, 20)

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-6, 5e-5, 1e-7)
my_train(mod_param, 20)

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 0, 1e-8)
my_train(mod_param, 20)

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5)
my_train(mod_param, 20)

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 4e-5)
my_train(mod_param, 20)

# VOC2012.VGG16_CAM5b_S.rmsProp.1e-5.l2w.5e-5.l2gap.1e-7.txt        => 0.614 at epoch 19
# VOC2012.VGG16_CAM_W_S.rmsProp.8e-6.l2.txt                         => 0.610 at epoch  4
# VOC2012.VGG16_CAM7_S.rmsProp.8e-6.l2.txt                          => 0.582 at epoch  6
# VOC2012.VGG16_CAM5b_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.575 at epoch 17
# VOC2012.VGG16_CAM5_S.rmsProp.8e-6.l2.txt                          => 0.572 at epoch 10
# VOC2012.VGG16_CAM_S.rmsProp.8e-6.l2.txt                           => 0.572 at epoch 16
# VOC2012.VGG16P_CAM7_S.rmsProp.8e-6.l2.txt                         => 0.571 at epoch 13
# VOC2012.VGG16_CAM3b_S.rmsProp.1e-5.l2w.5e-5.l2gap.2e-7.txt        => 0.571 at epoch 12
# VOC2012.VGG16_CAM7b_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.570 at epoch 19
# VOC2012.VGG16_CAM_W_S.rmsProp.8e-6.l2w.5e-5.txt                   => 0.569 at epoch  2
# VOC2012.VGG16_CAM5b_S.rmsProp.1e-5.l2w.5e-5.l2gap.1e-8.txt        => 0.568 at epoch 13
# VOC2012.VGG16_CAM9b_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.568 at epoch 23
# VOC2012.VGG16_CAM3b_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.568 at epoch  6
# VOC2012.VGG16P_CAM5_S.rmsProp.8e-6.l2.txt                         => 0.566 at epoch 15
# VOC2012.VGG16_CAM5d_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.565 at epoch 16
# VOC2012.VGG16_CAM7b_S.rmsProp.1e-5.l2w.5e-5.l2gap.2e-7.txt        => 0.564 at epoch 10
# VOC2012.VGG16_CAM9d_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.564 at epoch 19
# VOC2012.VGG16_CAM7d_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.564 at epoch 29
# VOC2012.VGG16_CAM3b_S.rmsProp.1e-5.l2w.5e-5.l1gap.1e-7.txt        => 0.563 at epoch 19
# VOC2012.VGG16_CAM5b_S.rmsProp.1e-5.l2w.5e-5.l2gap.2e-7.txt        => 0.560 at epoch 11
# VOC2012.VGG16_CAM5b_S.rmsProp.1e-5.l2w.5e-5.txt                   => 0.560 at epoch  7
# VOC2012.VGG16_CAM5b_S.rmsProp.1e-5.l2w.5e-5.l1gap.1e-8.txt        => 0.559 at epoch 10
# VOC2012.VGG16_CAM3d_S.rmsProp.1e-4.l2w.5e-5.l2gap.5e-7.txt        => 0.557 at epoch 22
# VOC2012.VGG16P_CAM3_S.rmsProp.8e-6.l2.txt                         => 0.555 at epoch 30
# VOC2012.VGG16_CAM3b_S.rmsProp.1e-5.l2w.5e-5.l2gap.5e-7.txt        => 0.542 at epoch  8





# mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-4, 5e-5, 5e-7)
# my_train(mod_param, 45)

# mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 2e-7)
# my_train(mod_param, 45)

# mod_param  = model_param.Model_params("EXT_MNIST", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-8)
# my_train(mod_param, 15)

# mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-4, 5e-5, 5e-7)
# my_train(mod_param, 45)

# mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-8)
# my_train(mod_param, 15)

# mod_param  = model_param.Model_params("PERSO", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-7)
# my_train(mod_param, 45)

# mod_param  = model_param.Model_params("PERSO", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-8)
# my_train(mod_param, 15)







# ###################################
# ### Plot the amazing dataset
# ###################################
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(3,5)
# for ax in [ax for l in axs for ax in l ]:
#   ax.imshow(my_images.load_augmented_image('/home/cuda/Pictures/creepy_max.png'))

# def animate(data):
#   batch_img,batch_labels = data
#   for idx,ax in enumerate([ax for l in axs for ax in l ]):
#     ax.imshow(batch_img[idx])
#     ax.set_title(testset.keys()[batch_labels[idx]])

# ani = animation.FuncAnimation(fig, animate, get_batch(testset,15), interval=1000)
# plt.show()


# ###################################
# ### Stats on loading time
# ###################################
# import time
# t_start = time.time()
# count = 0
# for a in get_batch(testset,15):
#   count += 15
#   pass

# t_end = time.time()
# print "%.3f sec to load %d images"%(t_end-t_start, count)
# print "=> %.3fsec to load 70 imgs"%(t_end-t_start/count*70)






