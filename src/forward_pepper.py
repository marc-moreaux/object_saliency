# from util import load_image
from forward_perso import Forward_perso
from skimage.transform import resize
from stream import VideoStream
import my_images

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import sleep

# Load tensorflow's caltech model
model = Forward_perso(3)


# Load video stream of pepper
vStream = VideoStream("10.0.160.236")
img = vStream.getFrame()

# Load naoqi tablet service to display image
from naoqi import ALProxy
tabletService = ALProxy("ALTabletService","10.0.160.236",9559);
url = "http://10.0.164.204:8081/new.jpg"

######################################
### To visualize on pepper,
###  launch an HTTP server in :
###  /home/cuda/work/cm_perso/py/image_server
###  $ python2 -m SimpleHTTPServer 8081
######################################




# compute preductions of a succession of images
img = vStream.getFrame()
img = my_images.crop_from_center(img)
img = resize(img, [224,224])
named_preds, vis = model.forward_image(img,-1)
labels = [n[0] for n in named_preds[:5]]
preds  = [n[1] for n in named_preds[:5]]
ys = preds



# Functions to manage bar chart
def make_bars(axis, ys, labels):
    bars = axis.bar(range(len(ys)), ys,  align = 'center')
    axis.set_ylim(0, 20)
    anots = []
    for x, (label, y) in enumerate(zip(labels, ys)):
        anots.append(axis.text(x, y+.05, label, horizontalalignment='center'))
    return bars, anots

def update_bars(axis, ys, labels):
    # Update anotations
    for anot, y, label in zip(anots, ys, labels):
        anot.set_text(label)
        anot.set_y(y+.05)
    
    # Update bars
    for bar,y in zip(bars, ys):
        bar.set_height(y)




import time


fig, ax = plt.subplots(1,2)
im1  = ax[0].imshow(img, animated=True)
im2  = ax[0].imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )

bars, anots = make_bars(ax[1], preds, labels)

def updatefig(*args):
    img = vStream.getFrame()
    img = my_images.crop_from_center(img)
    img = resize(img, [224,224])
    named_preds, vis = model.forward_image(img,-1)
    labels = [n[0] for n in named_preds[:5]]
    preds  = [n[1] for n in named_preds[:5]]
    
    im1.set_array(img)
    im2.set_array(vis)
    
    update_bars(ax[1], preds, labels)
    
    fig.canvas.draw()
    fig.savefig('/home/cuda/work/cm_perso/py/image_server/new.jpg')
    # sleep(0.5)
    tabletService.showImageNoCache("http://10.0.164.204:8081/new.jpg")

ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)

fig.show()



