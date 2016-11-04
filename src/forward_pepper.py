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
### Functions to manage bar chart
### Live visualisation of prediction
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


######################################
### Print functions for debug
######################################
def print_pred(pred):
  # visual print of singular prediction
  name,p,x,y = pred
  print "%-15s %3.3f at (%1.1f;%1.1f)"%(name,p,x,y)

def print_sorted_preds(named_preds):
  for pred in sorted(named_preds, key=lambda a:a[1])[-5:]:
    print_pred(pred)
  print '--'

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
  print ''

def print_over_threshold(summed_viz, thresholds, named_preds):
  over_thresh = []
  for idx, thresh in enumerate(thresholds.values()):
    if np.max(summed_viz[...,idx]) > np.log(thresh):
      print_pred(named_preds[idx])
      over_thresh.append(named_preds[idx])
  return over_thresh
  



######################################
### Functions related to conv6-filters
###  The viz's functions
######################################
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
  named_preds, vis = model.forward_image(img, visualize=True, do_resize=do_resize)
  return img, named_preds, vis


def main():


######################################
###  Parameter
######################################
mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-4, 5e-5, 5e-7)
model  = Forward_model(mod_param, 44)
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


# build the bar's colors
import colorsys
N = len(model.labels)
HSV_tuples = [(x*1.0/N, .8 if x%2 else 1, .8 if x%2 else 1) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
colors = RGB_tuples


######################################
### Plotting time !!
######################################
img, named_preds, vis = get_img_pred_and_vis(do_resize=False)
summed_viz = summed_vis(model, vis)

lbl_idx = []
lbl_idx.append( labels.index("head-phones") )
lbl_idx.append( labels.index("people") )
lbl_idx.append( labels.index("soda-can") )
lbl_idx.append( labels.index("cereal-box") )
lbl_idx.append( labels.index("coffee-mug") )
lbl_idx.append( labels.index("eyeglasses") )
lbl_idx.append( labels.index("computer-keyboard") )

thresholds = {'airplanes-101': 15.539495, 'ak47': 11.614439, 'american-flag': 10.979799, 'backpack': 12.223079, 'baseball-bat': 9.5437927, 'baseball-glove': 13.739741, 'basketball-hoop': 6.8361239, 'bat': 9.3481636, 'bathtub': 11.226589, 'bear': 12.300051, 'beer-mug': 10.193258, 'billiards': 13.462406, 'binoculars': 14.34924, 'birdbath': 8.5118666, 'blimp': 10.230799, 'bonsai-101': 14.111417, 'boom-box': 9.4953127, 'bowling-ball': 12.180779, 'bowling-pin': 13.188916, 'boxing-glove': 12.254652, 'brain-101': 10.274245, 'breadmaker': 12.863352, 'buddha-101': 9.4291439, 'bulldozer': 12.71928, 'butterfly': 12.233453, 'cactus': 11.847659, 'cake': 9.6894112, 'calculator': 11.544846, 'camel': 10.984636, 'cannon': 11.45732, 'canoe': 8.5938473, 'car-side-101': 15.239647, 'car-tire': 10.235177, 'cartman': 13.4251, 'cd': 14.570348, 'centipede': 11.146159, 'cereal-box': 9.5662346, 'chandelier-101': 14.05705, 'chess-board': 14.479905, 'chimp': 11.409027, 'chopsticks': 9.7986164, 'clutter': 6.7525487, 'cockroach': 11.884217, 'coffee-mug': 9.8205605, 'coffin': 7.9233661, 'coin': 13.396132, 'comet': 13.854059, 'computer-keyboard': 9.3863621, 'computer-monitor': 12.101433, 'computer-mouse': 9.4048815, 'conch': 10.171927, 'cormorant': 13.560222, 'covered-wagon': 10.693968, 'cowboy-hat': 11.713959, 'crab-101': 7.8341355, 'desk-globe': 11.701681, 'diamond-ring': 14.189276, 'dice': 12.145127, 'dog': 10.916739, 'dolphin-101': 13.114321, 'doorknob': 5.0762424, 'drinking-straw': 11.298171099057921, 'duck': 8.0495834, 'dumb-bell': 10.45462, 'eiffel-tower': 12.312577, 'electric-guitar-101': 12.958621, 'elephant-101': 14.492549, 'elk': 13.485504, 'ewer-101': 9.8520899, 'eyeglasses': 12.613165, 'faces-easy-101': 14.780043, 'fern': 14.704705, 'fighter-jet': 8.7353821, 'fire-extinguisher': 9.8947344, 'fire-hydrant': 11.806909, 'fire-truck': 13.965515, 'fireworks': 13.296287, 'flashlight': 8.1808243, 'floppy-disk': 7.1887569, 'football-helmet': 10.615582, 'french-horn': 13.687181, 'fried-egg': 10.432892, 'frisbee': 11.636504, 'frog': 9.7874126, 'frying-pan': 13.031361, 'galaxy': 13.235175, 'gas-pump': 9.2886629, 'giraffe': 11.326241, 'goat': 11.352355, 'golden-gate-bridge': 10.316961, 'goldfish': 9.4651136, 'golf-ball': 13.796005, 'goose': 10.911978, 'gorilla': 14.111794, 'grand-piano-101': 12.888706, 'grapes': 15.015621, 'grasshopper': 11.786924, 'greyhound': 10.316483, 'guitar-pick': 13.68029, 'hamburger': 15.062843, 'hammock': 9.5732231, 'harmonica': 9.8693962, 'harp': 11.529077, 'harpsichord': 11.165341, 'hawksbill-101': 11.578585, 'head-phones': 13.229425, 'helicopter-101': 10.342123, 'hibiscus': 15.14535, 'homer-simpson': 12.982221, 'horse': 10.339157, 'horseshoe-crab': 8.7140741, 'hot-air-balloon': 11.959567, 'hot-dog': 7.6435122, 'hot-tub': 14.082646, 'hourglass': 8.5736504, 'house-fly': 10.245955, 'human-skeleton': 11.28433, 'hummingbird': 13.007446, 'ibis-101': 13.071635, 'ice-cream-cone': 8.9562511, 'iguana': 10.795525, 'ipod': 11.828806, 'iris': 13.975347, 'jesus-christ': 7.8586192, 'joy-stick': 10.704171, 'kangaroo-101': 10.39012, 'kayak': 10.973958, 'ketch-101': 14.624043, 'killer-whale': 10.278771, 'knife': 7.0219631, 'ladder': 10.223049, 'laptop-101': 14.154712, 'lathe': 10.909991, 'leopards-101': 13.462743, 'license-plate': 9.1783524, 'light-house': 13.667336, 'lightbulb': 9.0387936, 'lightning': 14.409019, 'llama-101': 13.470165, 'mailbox': 6.5361915, 'mandolin': 7.0408216, 'mars': 15.681169, 'mattress': 13.038775, 'megaphone': 9.5789099, 'menorah-101': 11.231406, 'microscope': 13.210646, 'microwave': 12.780502, 'minaret': 14.352889, 'minotaur': 8.7425909, 'motorbikes-101': 14.512385, 'mountain-bike': 11.489552, 'mushroom': 9.2215405, 'mussels': 9.5242481, 'necktie': 12.587179, 'octopus': 10.618414, 'ostrich': 13.737724, 'owl': 12.664935, 'palm-pilot': 10.367587, 'palm-tree': 13.79882, 'paper-shredder': 10.851973, 'paperclip': 7.6281219, 'pci-card': 13.879761, 'penguin': 12.622005, 'people': 10.127394, 'pez-dispenser': 10.151289, 'photocopier': 12.675723, 'picnic-table': 11.054971, 'playing-card': 8.0703611, 'porcupine': 14.769379, 'pram': 8.6249542, 'praying-mantis': 7.9761715, 'pyramid': 9.2024403, 'raccoon': 13.569774, 'radio-telescope': 9.3716202, 'rainbow': 14.275431, 'refrigerator': 10.21283, 'revolver-101': 9.7003441, 'rifle': 7.8262658, 'rotary-phone': 8.7905035, 'roulette-wheel': 13.088307, 'saddle': 12.905226, 'saturn': 13.530177, 'school-bus': 14.30421, 'scorpion-101': 11.310881, 'screwdriver': 8.5278797, 'segway': 9.8139648, 'self-propelled-lawn-mower': 13.614111, 'sextant': 10.938291, 'sheet-music': 12.781863, 'skateboard': 5.3709097, 'skunk': 12.385101, 'skyscraper': 10.209162, 'smokestack': 8.4563723, 'snail': 10.329622, 'snake': 11.788112, 'sneaker': 11.271756, 'snowmobile': 12.525217, 'soccer-ball': 13.321915, 'socks': 12.545236, 'soda-can': 5.3792729, 'spaghetti': 14.648598, 'speed-boat': 8.1863899, 'spider': 10.452264, 'spoon': 7.7456422, 'stained-glass': 12.208684, 'starfish-101': 9.5304241, 'steering-wheel': 10.538023, 'stirrups': 9.0607805, 'sunflower-101': 15.126546, 'superman': 10.417388, 'sushi': 10.138167, 'swan': 13.547839, 'swiss-army-knife': 12.918182, 'sword': 6.741322, 'syringe': 11.86831, 't-shirt': 12.696012, 'tambourine': 11.732224, 'teapot': 11.243834, 'teddy-bear': 13.073026, 'teepee': 13.556503, 'telephone-box': 12.293881, 'tennis-ball': 13.999656, 'tennis-court': 13.988853, 'tennis-racket': 11.597626, 'tennis-shoes': 12.119856, 'theodolite': 8.5096235, 'toad': 14.345215, 'toaster': 8.2721653, 'tomato': 13.720393, 'tombstone': 8.7411394, 'top-hat': 10.413415, 'touring-bike': 13.964041, 'tower-pisa': 14.130401, 'traffic-light': 9.1431026, 'treadmill': 11.544156, 'triceratops': 9.1150694, 'tricycle': 7.9400773, 'trilobite-101': 9.9814043, 'tripod': 12.860129, 'tuning-fork': 8.508173, 'tweezer': 12.786347, 'umbrella-101': 14.133526, 'unicorn': 8.989871, 'vcr': 13.953634, 'video-projector': 13.71447, 'washing-machine': 13.383274, 'watch-101': 14.873101, 'waterfall': 11.931628, 'watermelon': 13.523896, 'welding-mask': 9.4707451, 'wheelbarrow': 6.9907722, 'windmill': 8.115716, 'wine-bottle': 11.967635, 'xylophone': 10.102029, 'yarmulke': 10.422739, 'yo-yo': 8.1963339, 'zebra': 12.6486}
thresholds = {'airplanes-101': 5.5431083e+14,'ak47': 3.0300088e+16,'american-flag': 3.5716338e+16,'backpack': 1.9385098e+12,'baseball-bat': 1.5624532e+09,'baseball-glove': 1.6674185e+13,'basketball-hoop': 21648198.0,'bat': 5.3302285e+08,'bathtub': 4.081619e+13,'bear': 5.1944895e+13,'beer-mug': 1.2813283e+09,'billiards': 1.8982378e+19,'binoculars': 2.4670087e+11,'birdbath': 86495976.0,'blimp': 5.466241e+10,'bonsai-101': 1.2646443e+14,'boom-box': 3.2918062e+09,'bowling-ball': 2.4476167e+11,'bowling-pin': 3.577533e+16,'boxing-glove': 6.7875977e+13,'brain-101': 9.9880776e+09,'breadmaker': 3.2294382e+12,'buddha-101': 3.8930813e+10,'bulldozer': 6.2439028e+16,'butterfly': 1.0646077e+11,'cactus': 8.1894794e+09,'cake': 7.40256e+09,'calculator': 3.5660982e+11,'camel': 1.664745e+10,'cannon': 6.9733033e+09,'canoe': 50444132.0,'car-side-101': 3.7557727e+10,'car-tire': 3.0141978e+08,'cartman': 6.5723494e+13,'cd': 6.4437289e+10,'centipede': 1.2770371e+12,'cereal-box': 2.6240285e+08,'chandelier-101': 8.2653332e+13,'chess-board': 2.2851635e+14,'chimp': 7.410362e+11,'chopsticks': 2.4793016e+11,'clutter': 8339.2676,'cockroach': 5.4527033e+10,'coffee-mug': 1.480877e+08,'coffin': 5039319.5,'coin': 1.3312232e+11,'comet': 2.0706677e+10,'computer-keyboard': 1.0867948e+09,'computer-monitor': 7.1442136e+12,'computer-mouse': 47223632.0,'conch': 1.8807533e+08,'cormorant': 3.2918025e+15,'covered-wagon': 5.9376696e+12,'cowboy-hat': 2.4143793e+09,'crab-101': 4341659.0,'desk-globe': 9.4681735e+10,'diamond-ring': 1.9219957e+14,'dice': 1.4127444e+10,'dog': 4.6726865e+10,'dolphin-101': 5.9940569e+11,'doorknob': 13459.243,'drinking-straw': 1.4986623698626245e+19,'duck': 3049474.0,'dumb-bell': 4.0482092e+11,'eiffel-tower': 3.5857907e+19,'electric-guitar-101': 4.0460525e+16,'elephant-101': 1.2808333e+15,'elk': 1.1112351e+13,'ewer-101': 1.4056063e+11,'eyeglasses': 1.9051025e+11,'faces-easy-101': 2.2937489e+11,'fern': 7.1044002e+12,'fighter-jet': 7798978.0,'fire-extinguisher': 1.719988e+10,'fire-hydrant': 9.6152668e+14,'fire-truck': 3.3854999e+14,'fireworks': 6.6339655e+12,'flashlight': 1.5764622e+08,'floppy-disk': 139934.2,'football-helmet': 6.6389512e+10,'french-horn': 9.3023247e+15,'fried-egg': 4.9036191e+09,'frisbee': 1.0446183e+08,'frog': 6.0765036e+09,'frying-pan': 3.2934395e+16,'galaxy': 1.2823036e+11,'gas-pump': 4.143339e+09,'giraffe': 3.8628279e+12,'goat': 7.5328881e+11,'golden-gate-bridge': 7.8824473e+10,'goldfish': 1.0932602e+09,'golf-ball': 1.3780492e+20,'goose': 6.2447514e+12,'gorilla': 2.1993976e+12,'grand-piano-101': 2.8961075e+15,'grapes': 1.5072703e+13,'grasshopper': 1.8578018e+10,'greyhound': 1.8462416e+13,'guitar-pick': 5.1328466e+13,'hamburger': 4.2002609e+14,'hammock': 3.3486584e+09,'harmonica': 9.5709587e+08,'harp': 8.1178002e+13,'harpsichord': 3.7312467e+10,'hawksbill-101': 7.3944072e+10,'head-phones': 3.4009946e+13,'helicopter-101': 2.1896231e+10,'hibiscus': 1.8412439e+12,'homer-simpson': 1.5423778e+12,'horse': 5.6068927e+10,'horseshoe-crab': 4.2088189e+09,'hot-air-balloon': 3.2325625e+13,'hot-dog': 7944667.0,'hot-tub': 8.2769019e+14,'hourglass': 7.5616689e+11,'house-fly': 3.1041532e+10,'human-skeleton': 5.0458438e+08,'hummingbird': 1.7925873e+15,'ibis-101': 1.9916933e+17,'ice-cream-cone': 63787136.0,'iguana': 2.2303822e+10,'ipod': 1.207799e+13,'iris': 1.0276757e+13,'jesus-christ': 14475644.0,'joy-stick': 7.2560395e+12,'kangaroo-101': 4.879786e+10,'kayak': 1.1078073e+12,'ketch-101': 3.8059018e+12,'killer-whale': 2.6241966e+09,'knife': 19238.672,'ladder': 1.5208937e+09,'laptop-101': 1.1846513e+16,'lathe': 1.7692549e+12,'leopards-101': 1.6646305e+09,'license-plate': 1.1828102e+12,'light-house': 3.6549751e+21,'lightbulb': 19177000.0,'lightning': 1.4482883e+12,'llama-101': 6.7335113e+13,'mailbox': 2773471.5,'mandolin': 18860346.0,'mars': 1.6402625e+11,'mattress': 3.1016109e+15,'megaphone': 4.2023346e+10,'menorah-101': 3.3722458e+12,'microscope': 4.1738928e+13,'microwave': 8.5503303e+13,'minaret': 3.1610025e+18,'minotaur': 32122660.0,'motorbikes-101': 2.8167275e+14,'mountain-bike': 5.9001851e+09,'mushroom': 53559256.0,'mussels': 2.2473357e+08,'necktie': 3.1209993e+11,'octopus': 2.6431208e+08,'ostrich': 1.7842261e+15,'owl': 6.5421483e+13,'palm-pilot': 2.3683363e+12,'palm-tree': 7.7542704e+12,'paper-shredder': 1.3108989e+08,'paperclip': 1352188.0,'pci-card': 1.9177542e+14,'penguin': 3.0658302e+14,'people': 5.2147927e+09,'pez-dispenser': 6.3548403e+14,'photocopier': 1.0440469e+13,'picnic-table': 3.8073414e+10,'playing-card': 5538763.0,'porcupine': 6.9250853e+12,'pram': 18372230.0,'praying-mantis': 1.560399e+08,'pyramid': 4.8814873e+11,'raccoon': 1.655157e+17,'radio-telescope': 2.3465157e+10,'rainbow': 8.7877876e+14,'refrigerator': 44751656.0,'revolver-101': 1.8511545e+12,'rifle': 9.0105306e+08,'rotary-phone': 2.1230793e+09,'roulette-wheel': 3.5503577e+12,'saddle': 3.9173898e+11,'saturn': 9.6435749e+10,'school-bus': 2.8633875e+16,'scorpion-101': 2.0427739e+10,'screwdriver': 26136550.0,'segway': 1.8123455e+14,'self-propelled-lawn-mower': 2.1806294e+14,'sextant': 8.092094e+09,'sheet-music': 3.0781233e+10,'skateboard': 14456.855,'skunk': 1.5204825e+16,'skyscraper': 42357828.0,'smokestack': 3.6834967e+09,'snail': 3.002801e+08,'snake': 1.6795233e+10,'sneaker': 2.9544725e+12,'snowmobile': 1.7613053e+13,'soccer-ball': 2.3668703e+15,'socks': 1.2557998e+12,'soda-can': 4422.9951,'spaghetti': 9.0263502e+13,'speed-boat': 6.3242834e+09,'spider': 2.2383148e+09,'spoon': 17250952.0,'stained-glass': 4.3774906e+13,'starfish-101': 4.5136595e+08,'steering-wheel': 2.6580664e+09,'stirrups': 5.438729e+08,'sunflower-101': 2.0106238e+15,'superman': 1.3998532e+12,'sushi': 3.5312387e+09,'swan': 1.5102679e+14,'swiss-army-knife': 2.9733163e+13,'sword': 97893.102,'syringe': 2.4944873e+11,'t-shirt': 8.4770243e+09,'tambourine': 1.0646093e+10,'teapot': 1.3279314e+10,'teddy-bear': 3.1996138e+12,'teepee': 4.4722962e+16,'telephone-box': 1.7343836e+13,'tennis-ball': 1.129395e+13,'tennis-court': 1.6999828e+15,'tennis-racket': 4.39301e+13,'tennis-shoes': 2.5194111e+10,'theodolite': 47372728.0,'toad': 8.3253107e+12,'toaster': 10031872.0,'tomato': 5.2412243e+12,'tombstone': 15937872.0,'top-hat': 4.3982733e+08,'touring-bike': 2.7450732e+11,'tower-pisa': 7.2208215e+13,'traffic-light': 6.6413796e+11,'treadmill': 7.972144e+12,'triceratops': 1.0741414e+09,'tricycle': 11951269.0,'trilobite-101': 1.2566726e+09,'tripod': 4.0014282e+16,'tuning-fork': 356258.0,'tweezer': 6.7144122e+11,'umbrella-101': 4.9185605e+13,'unicorn': 6.2804467e+08,'vcr': 1.2763661e+14,'video-projector': 3.2985965e+12,'washing-machine': 2.8531178e+13,'watch-101': 1.230518e+13,'waterfall': 2.739926e+11,'watermelon': 1.5243077e+11,'welding-mask': 2.7199916e+10,'wheelbarrow': 97438696.0,'windmill': 8.450663e+08,'wine-bottle': 6.4735841e+14,'xylophone': 2.0981789e+10,'yarmulke': 2.365763e+10,'yo-yo': 2.1747139e+08,'zebra': 4.6917522e+13} 
# for key in thresholds: thresholds[key] *= 1.5

# First plot
fig, axs = plt.subplots(2, (len(lbl_idx)+1)/2 )
axs = [a for b in axs for a in b]
ims = []
ims.append( axs[0].imshow(img, animated=True)  )
for i in range(len(lbl_idx)):
  ims.append( axs[i+1].imshow(summed_viz[0,:,:,lbl_idx[i]], vmin=0, vmax=thresholds.values()[lbl_idx[i]])  ) 
  axs[i+1].set_title(labels[lbl_idx[i]])


# update function
def updatefig(*args):
    img, named_preds, vis = get_img_pred_and_vis(do_resize=False)
    summed_viz = summed_vis(model, vis)
    
    # Update the axis
    ims[0].set_array( img )
    for i in range(len(lbl_idx)):
      data = np.exp( summed_viz[0,:,:,lbl_idx[i]] )
      ims[i+1].set_array( data )      
      
    
    # print_vis_stat(summed_viz, lbl_idx)
    # print_sorted_preds(named_preds)
    print_over_threshold(summed_viz, thresholds, named_preds)
    
    return ims

ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
fig.show()






if __name__ == '__main__':
  main()