import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import model_param
# import ipdb

def split_backward_comp(value, num_or_size_splits, axis):
    # Tensorflow versions < 0.12.0:  tf.split(axis, num_or_size_splits, value)
    # > tf.split(value, num_or_size_splits, axis)
    if tf.__version__ <= '0.12.0':
        return tf.split(axis, num_or_size_splits, value)
    return tf.split(value, num_or_size_splits, axis)

def concat_backward_comp(values, axis, name):
    # Tensorflow versions < 0.12.0:  tf.split(axis, num_or_size_splits, value)
    # > tf.split(value, num_or_size_splits, axis)
    if tf.__version__ <= '0.12.0':
        return tf.concat(axis, values, name)
    return tf.concat(values, axis, name):

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
          self.pretrained_weights = pickle.load(f, encoding='latin1') # encoding='utf-8': for compat with python3 (explicit encoding)

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
    """
    End of the original VGG16 network
    """
    pool5   = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool')
    fc6     = self.new_fc_layer( pool5, 1024,          'fc6' )
    fc7     = self.new_fc_layer( pool5, 1024,          'fc7' )
    fc8     = self.new_fc_layer( pool5, self.n_labels, 'fc8' )
    output  = fc8
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, None, None, self.output

  def _VGG16_CAM_W_S(self, cam_size):
    """
    End of the VGG16 - CAM network, as in the paper

    Params:
    cam_size -- 2Dsize of the cnn kernel
    """
    conv6 = self.new_conv_layer( self.conv5_3, [cam_size,cam_size,512,1024], "conv6")
    gap   = tf.reduce_mean( conv6, [1,2] )
    with tf.variable_scope("GAP"):
        gap_w = tf.get_variable(
                "W",
                shape=[1024, self.n_labels],
                initializer=tf.random_normal_initializer(0., 0.01))
    output = tf.matmul( gap, gap_w)
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, conv6, gap, output

  def _VGG16_CAMX_S(self, cam_filter_size, end_pooling=False):
    """
    End of the my version of VGG16-CAM net with one ccn

    Params:
    cam_filter_size -- 2Dsize of the cnn kernel
    end_polling -- add a pooling after conv6, before GAP
    """
    if end_pooling == True:
      pool5   = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')
      conv6  = self.new_conv_layer( pool5, [cam_filter_size,cam_filter_size,512,self.n_labels], "conv6")
    else :
      conv6  = self.new_conv_layer( self.conv5_3, [cam_filter_size,cam_filter_size,512,self.n_labels], "conv6")
    gap    = tf.reduce_mean( conv6, [1,2], name="gap"+str(cam_filter_size) )
    output = gap
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, conv6, gap, output

  def _VGG16_CAMXX_S(self, cam_filter_size, end_pooling=False):
    """
    End of the my version of VGG16-CAM net with many ccns

    Params:
    cam_filter_size -- 2Dsize of the cnn kernel
    end_polling -- add a pooling after conv6, before GAP
    """
    in_to_conv6 = self.conv5_3
    if end_pooling == True:
      pool5   = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')
      in_to_conv6 = pool5

    conv6 = []
    ccn_s = []
    for m_filter in cam_filter_size:
      cam_size = m_filter[0]
      n_cam    = m_filter[1]
      conv6_tmp = self.new_conv_layer( in_to_conv6, 
                                        [cam_size,cam_size,512,self.n_labels*n_cam], 
                                        "conv6_"+str(cam_size))
      conv6.append( conv6_tmp )
      gap = tf.reduce_mean( conv6_tmp, [1,2], name="gap"+str(cam_size))
      ccn = tf.reshape(gap,[-1,self.n_labels,n_cam])
      ccn = tf.reduce_mean(ccn, 2)
      ccn_s.append(ccn)
    
    output = sum(ccn_s)
    return self.pool1, self.pool2, self.pool3, self.pool4, self.conv5_3, conv6, ccn_s, output

  def inference( self, rgb, train=False ):
      rgb *= 255.
      r, g, b = tf.split_backward_comp(rgb, 3, 3) # tf.split(3, 3, rgb) => tf.split(rgb, 3, 3) # 2017-12-07 Alma: for tensorflow > 0.12.0 
      bgr = tf.concat_backward_comp(
          [
              b-self.image_mean[0],
              g-self.image_mean[1],
              r-self.image_mean[2]
          ],3) # 2017-12-07 Alma: change order after tf 1.0

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
      self.conv1_1=conv1_1 ; self.conv1_2=conv1_2 ; self.pool1  =pool1
      self.conv2_1=conv2_1 ; self.conv2_2=conv2_2 ; self.pool2  =pool2
      self.conv3_1=conv3_1 ; self.conv3_2=conv3_2 ; self.conv3_3=conv3_3 ; self.pool3=pool3
      self.conv4_1=conv4_1 ; self.conv4_2=conv4_2 ; self.conv4_3=conv4_3 ; self.pool4=pool4
      self.conv5_1=conv5_1 ; self.conv5_2=conv5_2 ; self.conv5_3=conv5_3


      m_type = self.mod_param.mod_type
      if m_type == model_param.Model_type.VGG16          : return self._VGG16()
      if m_type == model_param.Model_type.VGG16_CAM_W_S  : return self._VGG16_CAM_W_S(3)
      if m_type == model_param.Model_type.VGG16_CAM3_W_S : return self._VGG16_CAM_W_S(3)
      if m_type == model_param.Model_type.VGG16_CAM3_S   : return self._VGG16_CAMX_S(3)
      if m_type == model_param.Model_type.VGG16_CAM5_S   : return self._VGG16_CAMX_S(5)
      if m_type == model_param.Model_type.VGG16_CAM7_S   : return self._VGG16_CAMX_S(7)
      if m_type == model_param.Model_type.VGG16P_CAM3_S  : return self._VGG16_CAMX_S(3, True)
      if m_type == model_param.Model_type.VGG16P_CAM5_S  : return self._VGG16_CAMX_S(5, True)
      if m_type == model_param.Model_type.VGG16P_CAM7_S  : return self._VGG16_CAMX_S(7, True)

      if m_type == model_param.Model_type.VGG16_CAM3b_S   : return self._VGG16_CAMXX_S( [(3,2)] )
      if m_type == model_param.Model_type.VGG16_CAM3d_S   : return self._VGG16_CAMXX_S( [(3,4)] )
      if m_type == model_param.Model_type.VGG16_CAM3e_S   : return self._VGG16_CAMXX_S( [(3,5)] )
      if m_type == model_param.Model_type.VGG16_CAM5b_S   : return self._VGG16_CAMXX_S( [(5,2)] )
      if m_type == model_param.Model_type.VGG16_CAM5d_S   : return self._VGG16_CAMXX_S( [(5,4)] )
      if m_type == model_param.Model_type.VGG16_CAM5e_S   : return self._VGG16_CAMXX_S( [(5,5)] )
      if m_type == model_param.Model_type.VGG16_CAM7b_S   : return self._VGG16_CAMXX_S( [(7,2)] )
      if m_type == model_param.Model_type.VGG16_CAM7d_S   : return self._VGG16_CAMXX_S( [(7,4)] )
      if m_type == model_param.Model_type.VGG16_CAM7e_S   : return self._VGG16_CAMXX_S( [(7,5)] )
      if m_type == model_param.Model_type.VGG16_CAM9b_S   : return self._VGG16_CAMXX_S( [(9,2)] )
      if m_type == model_param.Model_type.VGG16_CAM9d_S   : return self._VGG16_CAMXX_S( [(9,4)] )
      if m_type == model_param.Model_type.VGG16_CAM5a7a_S : return self._VGG16_CAMXX_S( [(5,1),(7,1)] )
      if m_type == model_param.Model_type.VGG16_CAM5b7a_S : return self._VGG16_CAMXX_S( [(5,2),(7,1)] )
      if m_type == model_param.Model_type.VGG16_CAM3a5a7a_S : return self._VGG16_CAMXX_S( [(3,1),(5,1),(7,1)] )

  def get_classmap(self, label, conv6, end_shape=[224,224], eval_all=False):
      """
      Retrieves a preview of the CAM layer, for desired label

      Params:
      label -- The label to visualize
               if <-1>, returns all the classmaps
      conv6 -- The conv6 weights results
      end_shape -- the desired shape of visualisation
      """
      conv6_resized = tf.image.resize_bilinear( conv6, tuple(end_shape) )

      # returns one visualisation
      if eval_all == False:
        with tf.variable_scope("GAP", reuse=True):
            label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
            label_w = tf.reshape( label_w, [-1, 1024, 1] ) # [batch_size, 1024, 1]

        conv6_resized = tf.reshape(conv6_resized, [-1, np.prod(end_shape), 1024]) # [batch_size, 224*224, 1024]
        classmap = tf.batch_matmul( conv6_resized, label_w )
        classmap = tf.reshape( classmap, [-1,]+end_shape )
      else :
        # Return all 20 visualisations (for PASCAL)
        with tf.variable_scope("GAP", reuse=True):
          label_w = tf.get_variable("W")
          label_w = tf.reshape( label_w, [-1, 1024, self.n_labels] ) # [batch_size, 1024, 1]

        conv6_resized = tf.reshape(conv6_resized, [1, -1, 1024]) # [batch_size, 224*224, 1024]
        classmap = tf.batch_matmul( conv6_resized, label_w )
        classmap = tf.reshape( classmap, [-1]+end_shape+[self.n_labels] )

      return classmap

  def plot_conv_weights(self, layer_name):
    w = detector.get_weight( layer_name )
    dim = np.sqrt(w.shape[0])
    dim = int(dim+.5)
    fig, ax = plt.subplots(dim,dim)
    for idx,graph in enumerate(ax.flatten()):
        graph.imshow(w[idx])
    plt.show()





if __name__ == '__main__':
  mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM_W_S", 'rmsProp',   8e-6, 5e-5)
  tf.reset_default_graph()
  images_tf = tf.placeholder( tf.float32, [None, None, None, 3], name="images")
  detector  = Detector(mod_param)
  detector.inference(images_tf)


