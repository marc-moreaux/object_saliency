from skimage.transform import resize
import matplotlib.pyplot as plt
import my_images

import model_param
from forward_model import Forward_model
import numpy as np
import os

import sys
sys.path.append("/home/cuda/work/py-faster-rcnn/lib")
import datasets.factory


sys.path.append("/home/cuda/datasets/VOC2012")
import voc2012_getter

def gaussian(height, center_x, center_y, width_x, width_y):
  """
  Returns a gaussian function with the given parameters
  """
  width_x = float(width_x)
  width_y = float(width_y)
  return lambda x,y: height*exp(
              -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
  """
  Returns (height, x, y, width_x, width_y)
  the gaussian parameters of a 2D distribution by calculating its
  moments 
  """
  total = data.sum()
  X, Y = indices(data.shape)
  x = (X*data).sum()/total
  y = (Y*data).sum()/total
  col = data[:, int(y)]
  width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
  row = data[int(x), :]
  width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
  height = data.max()
  return height, x, y, width_x, width_y

def fitgaussian(data):
  """
  Returns (height, x, y, width_x, width_y)
  the gaussian parameters of a 2D distribution found by a fit
  """
  params = moments(data)
  errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) - data)
  p, success = optimize.leastsq(errorfunction, params)
  p, success = optimize.leastsq(errorfunction, params, maxfev=50)
  return p

def border_data(data, threshold_value=.7):
  """
  Retrun the largest array containing
  """
  m_max = threshold_value * np.abs(data).max()
  data = data * (m_max < data)
  xx = data.sum(axis=0)
  yy = data.sum(axis=1)
  left   = xx.nonzero()[0][0]
  right  = xx.nonzero()[0][-1]
  bottom = yy.nonzero()[0][0]
  top    = yy.nonzero()[0][-1]
  print data.shape
  print "%s - %s"%(bottom,top)
  print "%s - %s"%(left, right)
  data[bottom:top, left:right] = 1.
  return data


######################################
###  Parameters
######################################
voc_valid  = datasets.factory.__sets['voc_2012_val']()

mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM_W_S", 'rmsProp',   8e-6, 5e-5)
model  = Forward_model(mod_param, 4)
# mod_param  = model_param.Model_params("VOC2012", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-8)
# model  = Forward_model(mod_param, 9)


# gen = mod_param.get_batch('valid',10, im_size=0, official_splits=True)
# batch_images, batch_labels = gen.next()

# for idx in range(10):
#   orig_img = batch_images[idx]
#   named_pred, summed_viz = model.forward_image(orig_img, visualize=True, do_resize=False, names="BoundingBoxes")
  
#   best_pred_idx = np.argmax([a[1] for a in named_pred])
#   name,pred,l,t,r,b = named_pred[best_pred_idx]
#   mask = np.zeros(orig_img.shape)
#   mask[b:t,l:r] = 1
#   plt.imshow(batch_images[idx])
#   plt.imshow(mask, alpha=.5)
#   plt.axes().set_title(name+" - "+str(pred))
#   plt.show()




import my_images

# Compute all the predictions desired
# all_boxes[class][image] = [] or np.array of shape #dets x 5
all_boxes  = [[] for _ in range(model.n_labels)]
all_boxes2 = [[] for _ in range(model.n_labels)]


for img_idx in range(voc_valid.num_images):
  print img_idx
  img_path = voc_valid.image_path_at(img_idx)
  img = my_images.load_image(img_path, resize=False)
  named_pred, summed_viz = model.forward_image(img, visualize=True, do_resize=False, names="BoundingBoxes")
  named_pred2 = model.forward_image(img, visualize=False, do_resize=True, names="Predictions")
  
  best_idx  = np.argmax([p[1] for p in named_pred])
  best_idx2 = np.argmax([p[1] for p in named_pred2])
  # best_pred = sorted(named_pred, key=lambda a:a[1], reverse=True)[0]
  # Fill all_boxes[class][image]
  for c_idx,pred in enumerate(named_pred):
    if c_idx == best_idx:
      print pred
      name,p,l,t,r,b = pred
      t = img.shape[0] - t
      b = img.shape[0] - b
      all_boxes[c_idx].append(np.array([l,t,r,b,p]).reshape(1,-1))
    else:
      all_boxes[c_idx].append([])
    if c_idx == best_idx2:
      print pred
      name,p,l,t,r,b = pred
      t = img.shape[0] - t
      b = img.shape[0] - b
      all_boxes2[c_idx].append(np.array([l,t,r,b,p]).reshape(1,-1))
    else:
      all_boxes2[c_idx].append([])


all_boxes  = [None] + all_boxes
all_boxes2 = [None] + all_boxes2


voc_valid.evaluate_detections(all_boxes2, output_dir='output')


