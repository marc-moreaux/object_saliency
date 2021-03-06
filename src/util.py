import skimage.io
import skimage.transform
# import ipdb

import numpy as np


def load_image( path ):
  try:
    img = skimage.io.imread( path ).astype( float )
  except:
    return None
  
  if img is None: return None
  if len(img.shape) < 2: return None
  if len(img.shape) == 4: return None
  if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
  if img.shape[2] == 4: img=img[:,:,:3]
  if img.shape[2] > 4: return None
  
  img /= 255.
  
  short_edge = min( img.shape[:2] )
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
  resized_img = skimage.transform.resize( crop_img, [224,224] )
  return resized_img


def load_image2( path ):
  try:
    img = skimage.io.imread( path ).astype( float )
  except:
    return None
  
  if img is None:         _print_err(path,"img is none") ; return None
  if len(img.shape) < 2:  _print_err(path,"len(img.shape) is 4 :/") ; return None
  if len(img.shape) == 4: _print_err(path,"len(img.shape) is 2 :/") ; return None
  if len(img.shape) == 2:  img=np.tile(img[:,:,None], 3)
  if img.shape[2] == 4:    img=img[:,:,:3] #removes last chanel
  if img.shape[2] > 4:    _print_err(path,"more than 4 chanels - da fuck") ; return None
  
  img /= 255.
  return np.array(img)


def _print_err(img_p, err=None):
  print("in util -> load_image")
  print("could not load image : "+img_p)
  if err is not None : print(err)





##########################################
###  Plotting funtions
##########################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import my_images

def plot_histogram(vStream, model, idx ):

    # idx = model.mod_param.labels.index("soda-can")
    def update_hist(*args):
        plt.cla()
        img = vStream.getFrame()
        img = my_images.crop_from_center(img)
        img = resize(img, [224,224])
        named_preds, vis = model.forward_image(img, idx)
        plt.hist(vis.reshape(-1), bins=50, range=(-600,50))
        plt.plot([1400])
    
    fig = plt.figure()
    
    hist = plt.hist([0,0,0,0,0], bins=50, range=(-600,50))
    plt.plot([1400])
    
    import matplotlib.animation as animation 
    animation = animation.FuncAnimation(fig, update_hist, interval=10)
    plt.show()


# plot_histogram(vStream, model, 2)
