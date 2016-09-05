# from util import load_image
from skimage.transform import resize
from stream import VideoStream
import my_images

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import sleep


######################################
###  Parameter
######################################
MODEL_NAME = "perso"
# MODEL_NAME = "caltech"
PEPPER_IP = "10.0.165.29"  # jmot.local
PEPPER_IP = "10.0.160.236" # jarc.local
LOCAL_IP  = "10.0.164.204"
LOCAL_SERVER = "http://"+LOCAL_IP+":8081/"
LOCAL_SERVER_FOLDER = '/home/cuda/work/cm_perso/py/image_server/'



######################################
# Load video stream of pepper
######################################
vStream = VideoStream( PEPPER_IP )
img = vStream.getFrame()

# Load naoqi tablet service to display image
from naoqi import ALProxy
tabletService = ALProxy("ALTabletService", PEPPER_IP, 9559);


######################################
### To visualize on pepper,
###  launch an HTTP server in :
###  /home/cuda/work/cm_perso/py/image_server
###  $ python2 -m SimpleHTTPServer 8081
######################################



######################################
# Load tensorflow's caltech model
######################################
if MODEL_NAME == "perso":
    from forward_perso import Forward_perso as Forward_model
else:
    from forward_perso import Forward_caltech as Forward_model

model = Forward_model(3)





######################################
### Functions to manage bar chart
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




######################################
### Plotting time !!
######################################
# Retrieve the predictions
img = vStream.getFrame()
img = my_images.crop_from_center(img)
img = resize(img, [224,224])
named_preds, vis = model.forward_image(img,-1)
labels = [n[0] for n in named_preds[:5]]
preds  = [n[1] for n in named_preds[:5]]

# First plot
fig, ax = plt.subplots(1,2)
im1  = ax[0].imshow(img, animated=True)
im2  = ax[0].imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
bars, anots = make_bars(ax[1], preds, labels)

# update function
def updatefig(*args):
    img = vStream.getFrame()
    img = my_images.crop_from_center(img)
    img = resize(img, [224,224])
    named_preds, vis = model.forward_image(img,-1)
    labels = [n[0] for n in named_preds[:5]]
    preds  = [n[1] for n in named_preds[:5]]
    
    # Update the axis
    im1.set_array(img)
    im2.set_array(vis)
    update_bars(ax[1], preds, labels, colors)
    fig.canvas.draw()
    
    # Upload to pepper
    fig.savefig( LOCAL_SERVER_FOLDER+'new.jpg' )
    # tabletService.showImageNoCache("http://10.0.164.204:8081/new.jpg")

ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
tabletService.loadUrl(LOCAL_SERVER+"imageViewer.html")
tabletService.showWebview()
fig.show()



