# from util import load_image
from skimage.transform import resize
from stream import VideoStream
import my_images

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import model_param
from forward_model import Forward_model
from time import sleep
import numpy as np
import os


######################################
###  Parameter
######################################
mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 1e-5)
model  = Forward_model(mod_param, 14)
labels = model.mod_param.labels
PEPPER_IP  = "10.0.165.29"  # jmot.local
PEPPER_IP  = "10.0.161.43" # jarc.local
LOCAL_IP   = "10.0.164.160"
LOCAL_PORT = 8081
LOCAL_SERVER = "http://"+LOCAL_IP+":8081/"
LOCAL_SERVER_FOLDER = '/home/cuda/work/cm_perso/py/image_server/'

######################################
### Load video stream of pepper
######################################
vStream = VideoStream( PEPPER_IP )
img = vStream.getFrame()

# Load naoqi tablet service to display image
from naoqi import ALProxy
tabletService = ALProxy("ALTabletService", PEPPER_IP, 9559)
tts           = ALProxy("ALTextToSpeech" , PEPPER_IP, 9559)
led           = ALProxy("ALLeds"         , PEPPER_IP, 9559)


######################################
### To visualize on pepper,
###  launch an HTTP server in :
###  /home/cuda/work/cm_perso/py/image_server
###  $ python2 -m SimpleHTTPServer 8081
######################################





######################################
### Functions to manage bar chart
### Live visualisation of predictions
######################################
def make_bars(axis, ys, labels):
  bars = axis.bar(range(len(ys)), ys,  align = 'center')
  axis.set_ylim(0, 20)
  anots = []
  for x, (label, y) in enumerate(zip(labels, ys)):
    anots.append(axis.text(x, y+.05, label, horizontalalignment='center'))
  return bars, anots

def update_bars(axis, ys, labels, colors):
  # Update anotations
  for anot, y, label in zip(anots, ys, labels):
    anot.set_text(label)
    anot.set_y(y+.05)
  
  # Update bars
  for bar, y, label in zip(bars, ys, labels):
    bar.set_height(y)
    c = colors[model.labels.index(label)]
    bar.set_color(c)

def print_colors(colors):
  import math
  import numpy as np
  
  N = len(RGB_tuples)
  n = int(round(math.sqrt(N)+.5))
  color_viz = np.zeros((n*30,n*30,3))
  for idx,color in enumerate(RGB_tuples):
      x = (idx%n)*30
      y = (idx/n)*30
      color_viz[y:y+30,x:x+30] = color
  plt.imshow(color_viz)
  plt.show()


# build the bar's colors
import colorsys
N = len(model.labels)
HSV_tuples = [(x*1.0/N, .8 if x%2 else 1, .8 if x%2 else 1) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
colors = RGB_tuples

def print_sorted_preds(named_preds):
  for name, p in sorted(named_preds, key=lambda a:a[1])[-5:]:
    print "%-15s %3.3f"%(name,p)
  print '--'

def reshape_vis(vis, index):
  max_val = 30
  if len(vis.shape)>3:
    vis  = vis[0,:,:,index]
  print vis.max()
  vis += 10
  vis  = vis * (vis>0) # ReLu
  vis  = np.minimum(vis,max_val)
  vis /= max_val
  vis  = resize(vis,[224,224])
  return vis

def summed_vis(model, vis):
  # Retrieve model's CAM's properties 
  # eg: 'CALTECH256.VGG16_CAM5b_S.rmsProp.1e-5' => '5b'
  name = model.mod_param.get_name()
  suffix = [ s for s in name.split("_") if "CAM" in s][0][3:]
  if len(suffix) < 1:
    suffix = '3'
  if len(suffix) < 2:
    suffix += 'a'
    vis = [vis]
  
  filter_sizes = []
  for idx in range(0, 2, len(suffix)):
    alphabet_index = ord(suffix[idx+1]) - ord('a') + 1
    filter_sizes.append((int(suffix[idx]), alphabet_index))
  
  # Concatenate all the CAMs to <end_viz>
  end_viz = None
  for idx,filter_size in enumerate(filter_sizes):
    cam_size  = filter_size[0]
    n_cam     = filter_size[1]
    new_shape = list(vis[idx].shape[:-1])+[model.n_labels]+[n_cam]
    m_vis     = vis[idx].reshape(new_shape)
    m_vis     = m_vis.sum(axis=len(new_shape)-1)
    end_viz   = m_vis if end_viz == None else sum(end_viz, m_vis)
  
  return end_viz

def get_img_pred_and_vis(do_resize=True):
  img = vStream.getFrame()
  if resize == True :
    img = my_images.crop_from_center(img)
    resize_shape = [224,224]
  else:
    resize_shape = img.shape[:2]
  img = resize(img, resize_shape)
  named_preds, vis = model.forward_image(img, True)
  return img, named_preds, vis



def print_vis_stat(vis, idx=None):
  """Expected shape is (14,14,x)"""
  idx = range(vis.shape[-1]) if idx == None else idx
  for idx in range(vis.shape[-1]):
    _v = vis[:,:,:,idx]
    print "[%9.2f ; %9.2f ; %9.2f]"%(_v.min(), _v.mean(), _v.max()),
    if idx %3 == 0:
      print "%10s " % model.labels[idx][:10]
    else : 
      print "%10s " % model.labels[idx][:10],



# import sys; sys.path.append("/home/cuda/datasets/mnist/")
# import mnist_getter

# batch = mnist_getter.get_batch_224(batch_size=10).next()

# for img_idx in range(1):
#   img   = batch[0][img_idx]
#   lbl   = batch[1][img_idx]
#   named_preds, vis = model.forward_image(img, True)
  
#   summed_viz = summed_vis(model, vis)
#   print_vis_stat(summed_viz)
  
#   print lbl
#   fig, axs = plt.subplots(3,10)
#   for idx, ax in enumerate([b for a in axs for b in a]):
#     if idx < 10*2:
#       tmp_idx = (idx*2)%20+idx/10
#       ax.imshow(vis[0][0,:,:,tmp_idx], vmin=-15, vmax=15)
#     elif idx < 10*3:
#       ax.imshow(summed_viz[0,:,:,idx-20], vmin=-15, vmax=15)
  
#   plt.show()





######################################
### Plotting time !!
######################################
img, named_preds, vis = get_img_pred_and_vis()
summed_viz = summed_vis(model, vis)

lbl_idx = []
lbl_idx.append( labels.index("head-phones") )
lbl_idx.append( labels.index("people") )
lbl_idx.append( labels.index("soda-can") )
lbl_idx.append( labels.index("cereal-box") )
lbl_idx.append( labels.index("coffee-mug") )
lbl_idx.append( labels.index("eyeglasses") )
lbl_idx.append( labels.index("computer-keyboard") )


# First plot
fig, axs = plt.subplots(2, (len(lbl_idx)+1)/2 )
axs = [a for b in axs for a in b]
ims = []
ims.append( axs[0].imshow(img, animated=True)  )
for i in range(len(lbl_idx)):
  ims.append( axs[i+1].imshow(summed_viz[0,:,:,0], vmin=0, vmax=20000) )
  axs[i+1].set_title(labels[lbl_idx[i]])


# update function
def updatefig(*args):
    img, named_preds, vis = get_img_pred_and_vis()
    summed_viz = summed_vis(model, vis)
    
    # Update the axis
    ims[0].set_array( img )
    for i in range(len(lbl_idx)):
      data = np.exp( summed_viz[0,:,:,lbl_idx[i]] )
      ims[i+1].set_array( data )
      
      # if max value is over threshold,  show class
      if data.max() > 20000.:
        print data.max() > 20000.
        print "%d => %f"%(i, data.max())
        params = fitgaussian(data)
        fit    = gaussian(*params)
        axs[i+1].contourf(fit(*indices(data.shape)), cmap=cm.copper)
    
    print_vis_stat(summed_viz, lbl_idx)
    print_sorted_preds(named_preds)
    
    return ims

ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
fig.show()








from numpy import *
from scipy import optimize

def gaussian(height, center_x, center_y, width_x, width_y):
  """Returns a gaussian function with the given parameters"""
  width_x = float(width_x)
  width_y = float(width_y)
  return lambda x,y: height*exp(
              -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
  """Returns (height, x, y, width_x, width_y)
  the gaussian parameters of a 2D distribution by calculating its
  moments """
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
  """Returns (height, x, y, width_x, width_y)
  the gaussian parameters of a 2D distribution found by a fit"""
  params = moments(data)
  errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) -
                               data)
  p, success = optimize.leastsq(errorfunction, params, maxfev=20)
  return p








plt.contour(fit(*indices(data.shape)), cmap=cm.copper)
ax = gca()
(height, x, y, width_x, width_y) = params

plt.text(0.95, 0.05, """
x : %.1f
y : %.1f
width_x : %.1f
width_y : %.1f""" %(x, y, width_x, width_y),
        fontsize=16, horizontalalignment='right',
        verticalalignment='bottom', transform=ax.transAxes)

plt.show()




