import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import model_param
# import ipdb

class Detector():
  '''
  image_mean : precomputed image means
  n_labels   : nb of labels
  pretrained_weights : weights from VGG or pretrained model
  '''
  def __init__(self, mod_param):
      self.image_mean  = [103.939, 116.779, 123.68]
      self.mod_param   = mod_param
      self.n_labels    = mod_param.n_labels
      weight_file_path ='../caffe_layers_value.pickle'
      

      with open(weight_file_path,'rb') as f:
          # self.pretrained_weights = pickle.load(f,encoding='latin1')
          self.pretrained_weights = pickle.load(f)

  def get_weight( self, layer_name):
      layer = self.pretrained_weights[layer_name]
      return layer[0]

  def get_bias( self, layer_name ):
      layer = self.pretrained_weights[layer_name]
      return layer[1]

  def get_conv_weight( self, name ):
      f = self.get_weight( name )
      return f.transpose(( 2,3,1,0 ))

  def conv_layer( self, bottom, name ):
      with tf.variable_scope(name) as scope:

          w = self.get_conv_weight(name)
          b = self.get_bias(name)

          conv_weights = tf.get_variable(
                  "W",
                  shape=w.shape,
                  initializer=tf.constant_initializer(w)
                  )
          conv_biases = tf.get_variable(
                  "b",
                  shape=b.shape,
                  initializer=tf.constant_initializer(b)
                  )

          conv = tf.nn.conv2d( bottom, conv_weights, [1,1,1,1], padding='SAME')
          bias = tf.nn.bias_add( conv, conv_biases )
          relu = tf.nn.relu( bias, name=name )

      return relu

  def new_conv_layer( self, bottom, filter_shape, name ):
      with tf.variable_scope( name ) as scope:
          w = tf.get_variable(
                  "W",
                  shape=filter_shape,
                  initializer=tf.random_normal_initializer(0., 0.01))
          b = tf.get_variable(
                  "b",
                  shape=filter_shape[-1],
                  initializer=tf.constant_initializer(0.))

          conv = tf.nn.conv2d( bottom, w, [1,1,1,1], padding='SAME')
          bias = tf.nn.bias_add(conv, b)

      return bias #relu

  def fc_layer(self, bottom, name, create=False):
      shape = bottom.get_shape().as_list()
      dim = np.prod( shape[1:] )
      x = tf.reshape(bottom, [-1, dim])

      cw = self.get_weight(name)
      b = self.get_bias(name)

      if name == "fc6":
          cw = cw.reshape((4096, 512, 7,7))
          cw = cw.transpose((2,3,1,0))
          cw = cw.reshape((25088,4096))
      else:
          cw = cw.transpose((1,0))

      with tf.variable_scope(name) as scope:
          cw = tf.get_variable(
                  "W",
                  shape=cw.shape,
                  initializer=tf.constant_initializer(cw))
          b = tf.get_variable(
                  "b",
                  shape=b.shape,
                  initializer=tf.constant_initializer(b))

          fc = tf.nn.bias_add( tf.matmul( x, cw ), b, name=scope)

      return fc

  def new_fc_layer( self, bottom, input_size, output_size, name ):
      shape = bottom.get_shape().to_list()
      dim = np.prod( shape[1:] )
      x = tf.reshape( bottom, [-1, dim])

      with tf.variable_scope(name) as scope:
          w = tf.get_variable(
                  "W",
                  shape=[input_size, output_size],
                  initializer=tf.random_normal_initializer(0., 0.01))
          b = tf.get_variable(
                  "b",
                  shape=[output_size],
                  initializer=tf.constant_initializer(0.))
          fc = tf.nn.bias_add( tf.matmul(x, w), b, name=scope)

      return fc

  def _VGG16(self):
    pool5   = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool')
    fc6     = self.new_fc_layer( pool5, 1024,          'fc6' )
    fc7     = self.new_fc_layer( pool5, 1024,          'fc7' )
    fc8     = self.new_fc_layer( pool5, self.n_labels, 'fc8' )
    output  = fc8
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, None, None, self.output

  def _VGG16_CAM_W_S(self):
    conv6 = self.new_conv_layer( self.conv5_3, [3,3,512,1024], "conv6")
    gap   = tf.reduce_mean( conv6, [1,2] )
    with tf.variable_scope("GAP"):
        gap_w = tf.get_variable(
                "W",
                shape=[1024, self.n_labels],
                initializer=tf.random_normal_initializer(0., 0.01))
    output = tf.matmul( gap, gap_w)
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, conv6, gap, output

  def _VGG16_CAMX_S(self, cam_filter_size, end_pooling=False):
    if end_pooling == True:
      pool5   = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool')
      conv6  = self.new_conv_layer( pool5, [cam_filter_size,cam_filter_size,512,self.n_labels], "conv6")
    else :
      conv6  = self.new_conv_layer( self.conv5_3, [cam_filter_size,cam_filter_size,512,self.n_labels], "conv6")
    gap    = tf.reduce_mean( conv6, [1,2] )
    output = gap
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, conv6, gap, output


  def inference( self, rgb, train=False ):
      rgb *= 255.
      r, g, b = tf.split(3, 3, rgb)
      bgr = tf.concat(3,
          [
              b-self.image_mean[0],
              g-self.image_mean[1],
              r-self.image_mean[2]
          ])

      conv1_1 = self.conv_layer( bgr, "conv1_1" )
      conv1_2 = self.conv_layer( conv1_1, "conv1_2" )
      pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

      conv2_1 = self.conv_layer(pool1, "conv2_1")
      conv2_2 = self.conv_layer(conv2_1, "conv2_2")
      pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool2')

      conv3_1 = self.conv_layer( pool2, "conv3_1")
      conv3_2 = self.conv_layer( conv3_1, "conv3_2")
      conv3_3 = self.conv_layer( conv3_2, "conv3_3")
      pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool3')

      conv4_1 = self.conv_layer( pool3, "conv4_1")
      conv4_2 = self.conv_layer( conv4_1, "conv4_2")
      conv4_3 = self.conv_layer( conv4_2, "conv4_3")
      pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool4')

      conv5_1 = self.conv_layer( pool4, "conv5_1")
      conv5_2 = self.conv_layer( conv5_1, "conv5_2")
      conv5_3 = self.conv_layer( conv5_2, "conv5_3")

      # Store all in <self>
      conv1_1=self.conv1_1 ; conv1_2=self.conv1_2 ; pool1  =self.pool1
      conv2_1=self.conv2_1 ; conv2_2=self.conv2_2 ; pool2  =self.pool2
      conv3_1=self.conv3_1 ; conv3_2=self.conv3_2 ; conv3_3=self.conv3_3 ; pool3=self.pool3
      conv4_1=self.conv4_1 ; conv4_2=self.conv4_2 ; conv4_3=self.conv4_3 ; pool4=self.pool4
      conv5_1=self.conv5_1 ; conv5_2=self.conv5_2 ; conv5_3=self.conv5_3


      switcher = {
        model_param.Model_type.VGG16         = self._VGG16()
        model_param.Model_type.VGG16_CAM_W_S = self._VGG16_CAM_W_S()
        model_param.Model_type.VGG16_CAM_S   = self._VGG16_CAMX_S(3)
        model_param.Model_type.VGG16_CAM5_S  = self._VGG16_CAMX_S(5)
        model_param.Model_type.VGG16_CAM7_S  = self._VGG16_CAMX_S(7)
        model_param.Model_type.VGG16P_CAM3_S = self._VGG16P_CAMX_S(3, True)
        model_param.Model_type.VGG16P_CAM5_S = self._VGG16P_CAMX_S(5, True)
      }.get(self.mod_param.mod_type)


  def get_classmap(self, label, conv6):
      conv6_resized = tf.image.resize_bilinear( conv6, [224, 224] )
      with tf.variable_scope("GAP", reuse=True):
          label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
          label_w = tf.reshape( label_w, [-1, 1024, 1] ) # [batch_size, 1024, 1]

      conv6_resized = tf.reshape(conv6_resized, [-1, 224*224, 1024]) # [batch_size, 224*224, 1024]

      classmap = tf.batch_matmul( conv6_resized, label_w )
      classmap = tf.reshape( classmap, [-1, 224,224] )
      return classmap

  def plot_conv_weights(self, layer_name):
    w = detector.get_weight( layer_name )
    dim = np.sqrt(w.shape[0])
    dim = int(dim+.5)
    fig, ax = plt.subplots(dim,dim)
    for idx,graph in enumerate(ax.flatten()):
        graph.imshow(w[idx])
    plt.show()








