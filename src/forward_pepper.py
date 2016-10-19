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
mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-5, 5e-5, 5e-5)
model      = Forward_model(mod_param, 5)
PEPPER_IP  = "10.0.165.29"  # jmot.local
PEPPER_IP  = "10.0.160.236" # jarc.local
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





# ######################################
# ### Plotting time !!
# ######################################
img = vStream.getFrame()
img = my_images.crop_from_center(img)
img = resize(img, [224,224])
named_preds, vis = model.forward_image(img,-1)
vis = reshape_vis(vis, model.mod_param.labels.index("soda-can"))

# First plot
fig, ax = plt.subplots(1,1)
im1 = ax.imshow(img, animated=True)
im2 = ax.imshow(vis, cmap=plt.cm.jet, alpha=.5, interpolation='nearest', vmin=0, vmax=1)


# update function
def updatefig(*args):
    img = vStream.getFrame()
    img = my_images.crop_from_center(img)
    img = resize(img, [224,224])
    named_preds, vis = model.forward_image(img,-1)
    tmp = vis[0,:,:,model.mod_param.labels.index("soda-can")]
    print "min %2.3f, max %2.3f, mean %2.3f"%(tmp.min(),tmp.max(),tmp.mean())
    vis = reshape_vis(vis, model.mod_param.labels.index("soda-can"))
    
    # Update the axis
    im1.set_array(img)
    im2.set_array(vis)
    
    print_sorted_preds(named_preds)
    
    return im1, im2

ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
fig.show()





