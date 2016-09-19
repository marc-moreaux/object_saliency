
from util import load_image
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
batch_size = 50


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
      tmp  = my_images.load_augmented_image(path)
      img_list.append(tmp)
   
    yield np.stack(img_list), np.stack(label_list)


###################################
### Begin with Tensorflow :D
###################################



# if pretrained_model_path:
#     print("Pretrained")
#     saver.restore(sess, pretrained_model_path)



def print_train(epoch, iterations, accuracy, train_loss, batch_size, n_iter, log_file):
  # log_file.write('epoch:'+str(epoch)+' iter:'+str(iterations)+' loss:'+str(train_loss) + '\n')
  log_file.write('epoch:%2d - iter:%3d - loss:%2.7f\n'%(epoch, iterations, train_loss))
  print("======================================")
  print("Epoch", epoch, "Iteration", iterations)
  print("Processed", iterations*batch_size, '/', n_iter)
  print("Accuracy:", accuracy, '/', batch_size)
  print("Training Loss:", train_loss)
  print("\n")

def print_test(epoch, test_accuracy, log_file):
  # log_file.write('epoch:'+str(epoch)+' acc:'+str(test_accuracy) + '\n')
  log_file.write('epoch:%2d - acc:%.3f\n'%(epoch, test_accuracy))
  log_file.write('---------------\n')
  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
  print('epoch:'+str(epoch)+'\tacc:'+str(test_accuracy) + '\n')
  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

def my_train(mod_param, n_epochs=15):
  with open(mod_param.paths["log_file"], 'w',0) as log_file:
    # train vars
    model_path = mod_param.paths["save_model"]
    n_labels   = mod_param.n_labels
    pretrained_model_path = model_path+'-3'
    trainset, testset     = mod_param.get_datasets()
    loss_tf               = mod_param.get_loss(output, labels_tf)
    optimizer, train_op   = mod_param.get_optimizer(tf_learning_rate, loss_tf)
    
    # loop varaibles
    tf.initialize_all_variables().run()
    loss_list = []
    iterations = 0
    n_iter = len(trainset.keys())*max(map(len, trainset.values()))
    for epoch in range(n_epochs):
      ##################################
      ### Training 
      ##################################
      iterations = 0
      for batch_images, batch_labels in get_batch(trainset,batch_size):
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
      for batch_images, batch_labels in get_batch(testset,batch_size):
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
      mod_param.lr *= 0.99
  




# ###################################
# ### Setup 1st model to train

# model vars
mod_param  = model_param.Model_params("PERSO", "VGG16_CAM_S", 'rmsProp', 0.000005, True)

# Tensorflow vars
tf_learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')
detector  = Detector(mod_param)
p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)

# loss function
sess     = tf.InteractiveSession()
saver    = tf.train.Saver( max_to_keep=50 )

my_train(mod_param, 40)




# ###################################
# ### Setup 2nd model to train
tf.reset_default_graph()

# model vars
mod_param  = model_param.Model_params("PERSO", "VGG16_CAM_S", 'rmsProp', 0.00001, True)

# Tensorflow vars
tf_learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')
detector  = Detector(mod_param)
p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)

# loss function
sess     = tf.InteractiveSession()
saver    = tf.train.Saver( max_to_keep=50 )

my_train(mod_param, 40)













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





