import skimage.io
import skimage.transform
import numpy as np
import random

from os import listdir
from os.path import isfile, join
from os.path import expanduser
from numpy.random import random as rand
from skimage.transform import resize, rescale


datasets_path = expanduser("~")
datasets_path = join(datasets_path, "Desktop/datasets")

def get_img_in_folder(folder_path):
  ''' returns all the paths of images in a folder'''
  imgsPath = [join(folder_path, f) 
           for f in listdir(folder_path) 
             if isfile(join(folder_path, f))
             and ('jpg' or 'jpeg' or 'png') in f ]
  return imgsPath


def image_paths(dataset_name="VOC2012"):
  '''Return the images paths of a dataset'''
  if   dataset_name == ("voc" or "VOC2014"):
    imgsPaths = get_img_in_folder(join(datasets_path,"VOC2012/JPEGImages"))
  elif dataset_name == "coco":
    imgsPaths = get_img_in_folder(join(datasets_path,"coco/train2014"))
  elif dataset_name == "action40":
    imgsPaths = get_img_in_folder(join(datasets_path,"actions40"))
  elif dataset_name == "action40":
    imgsPaths = get_img_in_folder(join(datasets_path,"actions40"))
  # Missing caltech256, pano
  return imgsPaths



def crop_from_center(img):
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  img = img[yy: yy + short_edge, xx: xx + short_edge]
  return img



def load_image(path, resize=True):
  '''returns image of shape [224, 224, 3]
  [height, width, depth]'''
  # load image
  img = skimage.io.imread(path)
  img = skimage.color.gray2rgb(img)
  if img.max > 1:
    img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  # print "Original Image Shape: ", img.shape
  # we crop image from center
  if resize is True:
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    img = skimage.transform.resize(img, (224, 224))
  return img


def load_image_light(path):
  '''returns image of shape [224, 224, 3]
  [height, width, depth]'''
  # load image
  img = skimage.io.imread(path)
  # if img.max > 1:
  #   img = img / 255.0
  # assert (0 <= img).all() and (img <= 1.0).all()
  # print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  img = img[yy: yy + short_edge, xx: xx + short_edge]
  # resize to 224, 224
  # img = skimage.transform.resize(img, (224, 224))
  return img


def load_random_image(dataset_name="voc"):
  '''loads a random image of a dataset'''
  img_path = random.choice(image_paths(dataset_name))
  return load_image(img_path)



def load_augmented_image(path):
  """Return a image given path 
  .33 proba to be original image
  .33 proba to be flipped image
  .33 proba to be shrinked 
    --> (image between 200&100px)
  """
  proba = rand()
  if proba < .33:
    img = load_image(path, resize=True)
    return img
  elif proba < .66:
    img = load_image(path, resize=True)
    return np.fliplr(img)
  else:
    # Load the background and the original image
    img_back = np.ones([224,224,3]) * rand(3)
    img_orig = load_image(path, resize=False)

    # Maybe flip the image
    if rand()>.5:
      img_orig = np.fliplr(img_orig)

    # Reshape original image (to fit to a max size of 200px)
    max_new_shape = max(img_orig.shape)
    downscale_factor =  round(rand()*100+100) / max_new_shape
    img_orig = rescale(img_orig, downscale_factor)

    # Put img_orig on the background
    yy,xx,_ = img_orig.shape
    y, x ,_ = img_back.shape
    y = int(rand()*(y-yy))
    x = int(rand()*(x-xx))
    img_back[y:yy+y,x:xx+x] = img_orig
    return img_back