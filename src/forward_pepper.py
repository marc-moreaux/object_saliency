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
mod_param  = model_param.Model_params('PERSO', "VGG16_CAM7_S", 'rmsProp', 0.000008)
PEPPER_IP  = "10.0.165.29"  # jmot.local
PEPPER_IP  = "10.0.160.236" # jarc.local
LOCAL_IP   = "10.0.164.204"
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
# Load tensorflow's caltech model
######################################
model = Forward_model(mod_param, 10)




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
# def forward_one_image():
# img = vStream.getFrame()
# img = my_images.crop_from_center(img)
# img = resize(img, [224,224])
# named_preds, vis = model.forward_image(img,-1)
# labels = [n[0] for n in named_preds[:5]]
# preds  = [n[1] for n in named_preds[:5]]

# # First plot
# fig, ax = plt.subplots(1,2)
# im1  = ax[0].imshow(img, animated=True)
# im2  = ax[0].imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
# bars, anots = make_bars(ax[1], preds, labels)

# # update function
# def updatefig(*args):
#     img = vStream.getFrame()
#     img = my_images.crop_from_center(img)
#     img = resize(img, [224,224])
#     named_preds, vis = model.forward_image(img,-1)
#     labels = [n[0] for n in named_preds[:5]]
#     preds  = [n[1] for n in named_preds[:5]]
    
#     # Update the axis
#     im1.set_array(img)
#     im2.set_array(vis)
#     update_bars(ax[1], preds, labels, colors)
#     fig.canvas.draw()
    
#     # Upload to pepper
#     fig.savefig( LOCAL_SERVER_FOLDER+'new.jpg' )
#     # tabletService.showImageNoCache("http://10.0.164.204:8081/new.jpg")

# ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
# tabletService.loadUrl(LOCAL_SERVER+"imageViewer.html")
# tabletService.showWebview()
# fig.show()













# def forward_image_as_batch():

#######################################
### Detection using 3 frames
#######################################
# import numpy as np
# import time
# import itertools
# from threshold import d_detect


# def detect(d_detect, pred):
#     for key in d_detect:
#         idx = model.labels.index(key)
#         if pred[idx] > d_detect[key][0]:
#             print "-- %s, %.3f"%(key,pred[idx])
#         if pred[idx] > d_detect[key][0] and not d_detect[key][2]:
#             d_detect[key][2] = True
#             tts.say("I see a %s !"%(key))
#         elif pred[idx] < d_detect[key][1]:
#             d_detect[key][2] = False

# t_preds = []
# while True:
#     img = vStream.getFrame()
#     imgs = [resize(img, (224,224))]
#     width = 480
#     for x in [0, 640-width]:
#         n_img = img[0:0+width, x:x+width]
#         n_img = resize(n_img, (224, 224))
#         imgs.append(n_img)
    
#     imgs = np.stack(imgs)
#     preds = model.forward_images(imgs)
    
#     for pred in preds:
#         t_preds.append(pred)
#     if len(t_preds) > 20*3:
#         for _ in range(3):
#             _=t_preds.pop(0)
    
#     actual_pred = np.array(t_preds).mean(axis=0)
    
#     detect(d_detect, actual_pred)
    
#     print d_detect["coffee-mug"]
#     print "%s, %.2f"%(model.labels[actual_pred.argmax()], actual_pred.max())
#     print "%s, %.2f"%("texting_message", actual_pred[model.labels.index("texting_message")]) 
#     print "--"







# #######################################
# ### Detection using 3*4 frames
# #######################################
# import numpy as np
# import time
# import itertools
# import matplotlib.pyplot as plt
# from threshold import d_detect

# led.fade("EarLeds", 0, .1)


# good_classes = ["eyeglasses","people","applauding","laptop-101","brushing_teeth","computer-monitor","computer-keyboard",'computer-mouse',\
# "coffee-mug","texting_message","phoning","umbrella-101","drinking","reading","soda-can","holding_an_umbrella","cereal-box", "head-phones"]

# sentense = ["bottom", "top",    "right",  "left"]
# # mask   = [idx/4==2, idx/4==0, idx%4==3, idx%4==0]

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0) 


# def detect(d_detect, pred, winNb):
#     for key in d_detect:
#         idx = model.labels.index(key)
#         if d_detect[key][2][winNb] is True :
#             print "%2d-- %s, %.3f"%(winNb,key,pred[idx])
#         if pred[idx] > d_detect[key][0] and not d_detect[key][2][winNb]:
#             d_detect[key][2][winNb] = True
#             mask  = [winNb/4==2, winNb/4==0, winNb%4==3, winNb%4==0]
#             pos   = [s for s,m in zip(sentense,mask) if m]
#             if pos == []:
#                 pos = ["center"]
#             pos   = " ".join(pos)
#             led.fade("EarLeds", 1, .1)
#             tts.say("I see a %s at %s !"%(key, pos))
#             led.fade("EarLeds", 0, 1)
#         elif pred[idx] < d_detect[key][1]:
#             d_detect[key][2][winNb] = False



# fig, axs = plt.subplots(3,4)
# im = np.zeros((224,224,3))
# plt_im = []

# # for ax in [ ax for sub_axs in axs for ax in sub_axs ]
# for ax in itertools.chain(*axs):
#     plt_im.append(ax.imshow(im, animated=True))

# win_width = 224
# n_x_split = 4
# n_y_split = 3
# # 3*4 lists
# t_preds   = [ [] for _ in range(n_x_split * n_y_split) ]

# def updatefig(*args):
#     img   = vStream.getFrame()
#     imgs  = []
#     im_h  = img.shape[0]
#     im_w  = img.shape[1]
#     for y in range(n_y_split):
#         y = y*( im_h-win_width )/(n_y_split-1)
#         for x in range(n_x_split):
#             x = x*( im_w-win_width )/(n_x_split-1)
#             n_img = img[y:y+win_width, x:x+win_width]
#             #n_img = resize(n_img, (224, 224))
#             imgs.append(n_img)
    
#     imgs = np.stack(imgs)
#     preds = model.forward_images(imgs)
    
#     for idx, pred in enumerate(preds):
#         t_preds[idx].append(pred)
#         while len(t_preds[idx]) > 5:
#             _=t_preds[idx].pop(0)
#         actual_pred = np.array(t_preds[idx]).mean(axis=0)
#         detect(d_detect, actual_pred, idx)
#         plt_im[idx].set_array(imgs[idx])
    
#     # for idx in range(len(preds)):
#     #     actual_pred = np.array(t_preds[idx]).mean(axis=0)
#     #     print "%s, %.2f"%(model.labels[actual_pred.argmax()], actual_pred.max())
    
#     actual_pred = np.array(t_preds[8]).mean(axis=0)
#     # actual_pred = softmax(actual_pred)
#     actual_pred = actual_pred/actual_pred.sum(axis=0)
#     for c in good_classes:
#         idx = model.labels.index(c)
#         print "%s \t- %.3f"%(c, actual_pred[idx])
    
    
#     print '--'
    
#     fig.savefig( LOCAL_SERVER_FOLDER+'new.jpg' )
     
#     return imgs[0], imgs[1], imgs[2], imgs[3],\
#            imgs[4], imgs[5], imgs[6], imgs[7],\
#            imgs[8], imgs[9], imgs[10],imgs[11]
    

# ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
# tabletService.loadUrl(LOCAL_SERVER+"imageViewer.html")
# tabletService.showWebview()
# fig.show()






# vis  = vis[0,:,:,model.mod_param.labels.index("people")]
# vis *= 1/vis.max()
# vis  = vis.clip(min=0)
# vis  = resize(vis,[224,224])









# def reshape_vis(vis):
#     vis = vis[0,:,:,:5]
#     vis = vis.swapaxes(0,2)
#     vis = vis.swapaxes(1,2)
#     vis = vis * (vis>0) # ReLu
#     vis /= vis.max(axis=(0,1))
#     vis = np.array([resize(viz,[224,224]) for viz in vis])
#     return vis


# # ######################################
# # ### Plotting time !!
# # ######################################
# # Retrieve the predictions
# # def forward_one_image():
# img = vStream.getFrame()
# img = my_images.crop_from_center(img)
# img = resize(img, [224,224])
# named_preds, vis = model.forward_image(img,-1)
# vis = reshape_vis(vis)


# # First plot
# ims1 = []
# ims2 = []
# fig, axs = plt.subplots(2,3)
# for viz,ax in zip(vis, [ax for a in axs for ax in a]):
#   ims1.append( ax.imshow(img, animated=True) )
#   ims2.append( ax.imshow(viz, cmap=plt.cm.jet, alpha=.5, interpolation='nearest'))


# # update function
# def updatefig(*args):
#     img = vStream.getFrame()
#     img = my_images.crop_from_center(img)
#     img = resize(img, [224,224])
#     named_preds, vis = model.forward_image(img,-1)
#     vis  = reshape_vis(vis)
#     # vis  = vis[0,:,:,model.mod_param.labels.index("people")]
    
#     # Update the axis
#     for viz,im1,im2 in zip(vis, ims1, ims2):
#         im1.set_array(img)
#         im2.set_array(viz)
#     # Upload to pepper
#     # fig.savefig( LOCAL_SERVER_FOLDER+'new.jpg' )
#     # tabletService.showImageNoCache("http://10.0.164.204:8081/new.jpg")
#     return ims1[0],ims2[0],ims1[1],ims2[1],ims1[2],ims2[2],ims1[3],ims2[3],ims1[4],ims2[4]
#     # return ims1[0],ims2[0],ims1[1],ims2[1],ims1[2],ims2[2],ims1[3],ims2[3],ims1[4],ims2[4],ims1[5],ims2[5],ims1[6],ims2[6],ims1[7],ims2[7],ims1[8],ims2[8],ims1[9],ims2[9],ims1[10],ims2[10],ims1[11],ims2[11],ims1[12],ims2[12],ims1[13],ims2[13],ims1[14],ims2[14],ims1[15],ims2[15],ims1[16],ims2[16],ims1[17],ims2[17],ims1[18],ims2[18],ims1[19],ims2[19],ims1[20],ims2[20],ims1[21],ims2[21],ims1[22],ims2[22],ims1[23],ims2[23],ims1[24],ims2[24],ims1[25],ims2[25],ims1[26],ims2[26],ims1[27],ims2[27],ims1[28],ims2[28],ims1[29],ims2[29],ims1[30],ims2[30],ims1[31],ims2[31],ims1[32],ims2[32],ims1[33],ims2[33],ims1[34],ims2[34],ims1[35],ims2[35],ims1[36],ims2[36],ims1[37],ims2[37],ims1[38],ims2[38],ims1[39],ims2[39],ims1[40],ims2[40],ims1[41],ims2[41],ims1[42],ims2[42],ims1[43],ims2[43],ims1[44],ims2[44],ims1[45],ims2[45],ims1[46],ims2[46],ims1[47],ims2[47],ims1[48],ims2[48],ims1[49],ims2[49],ims1[50],ims2[50],ims1[51],ims2[51],ims1[52],ims2[52],ims1[53],ims2[53],ims1[54],ims2[54],ims1[55],ims2[55],ims1[56],ims2[56],ims1[57],ims2[57],ims1[58],ims2[58],ims1[59],ims2[59],ims1[60],ims2[60],ims1[61],ims2[61],ims1[62],ims2[62]

# ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
# # tabletService.loadUrl(LOCAL_SERVER+"imageViewer.html")
# # tabletService.showWebview()
# fig.show()







def reshape_vis(vis, index):
    if len(vis.shape)>3:
        vis  = vis[0,:,:,index]
    print vis.max()
    vis  = vis * (vis>0) # ReLu
    vis  = np.minimum(vis,10)
    # vis  = vis * (vis<10) # ReLu top limit
    vis /= 10 #vis.max()
    vis  = resize(vis,[224,224])
    return vis

# ######################################
# ### Plotting time !!
# ######################################
# Retrieve the predictions
# def forward_one_image():
img = vStream.getFrame()
img = my_images.crop_from_center(img)
img = resize(img, [224,224])
named_preds, vis = model.forward_image(img,-1)
vis = reshape_vis(vis, model.mod_param.labels.index("coffee-mug"))

# argmax = np.array([p[1] for p in named_preds]).argmax()
# vis    = reshape_vis(vis, argmax)
# print  named_preds[argmax]


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
    # argmax = np.array([p[1] for p in named_preds]).argmax()
    # vis = reshape_vis(vis, argmax)
    # vis  = vis[0,:,:,model.mod_param.labels.index("coffee-mug")]
    # print  named_preds[argmax]
    vis = reshape_vis(vis, model.mod_param.labels.index("coffee-mug"))
    
    # Update the axis
    im1.set_array(img)
    im2.set_array(vis)
    
    # Upload to pepper
    # fig.savefig( LOCAL_SERVER_FOLDER+'new.jpg' )
    # tabletService.showImageNoCache("http://10.0.164.204:8081/new.jpg")
    return im1, im2
    # return ims1[0],ims2[0],ims1[1],ims2[1],ims1[2],ims2[2],ims1[3],ims2[3],ims1[4],ims2[4]
    # return ims1[0],ims2[0],ims1[1],ims2[1],ims1[2],ims2[2],ims1[3],ims2[3],ims1[4],ims2[4],ims1[5],ims2[5],ims1[6],ims2[6],ims1[7],ims2[7],ims1[8],ims2[8],ims1[9],ims2[9],ims1[10],ims2[10],ims1[11],ims2[11],ims1[12],ims2[12],ims1[13],ims2[13],ims1[14],ims2[14],ims1[15],ims2[15],ims1[16],ims2[16],ims1[17],ims2[17],ims1[18],ims2[18],ims1[19],ims2[19],ims1[20],ims2[20],ims1[21],ims2[21],ims1[22],ims2[22],ims1[23],ims2[23],ims1[24],ims2[24],ims1[25],ims2[25],ims1[26],ims2[26],ims1[27],ims2[27],ims1[28],ims2[28],ims1[29],ims2[29],ims1[30],ims2[30],ims1[31],ims2[31],ims1[32],ims2[32],ims1[33],ims2[33],ims1[34],ims2[34],ims1[35],ims2[35],ims1[36],ims2[36],ims1[37],ims2[37],ims1[38],ims2[38],ims1[39],ims2[39],ims1[40],ims2[40],ims1[41],ims2[41],ims1[42],ims2[42],ims1[43],ims2[43],ims1[44],ims2[44],ims1[45],ims2[45],ims1[46],ims2[46],ims1[47],ims2[47],ims1[48],ims2[48],ims1[49],ims2[49],ims1[50],ims2[50],ims1[51],ims2[51],ims1[52],ims2[52],ims1[53],ims2[53],ims1[54],ims2[54],ims1[55],ims2[55],ims1[56],ims2[56],ims1[57],ims2[57],ims1[58],ims2[58],ims1[59],ims2[59],ims1[60],ims2[60],ims1[61],ims2[61],ims1[62],ims2[62]

ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
# tabletService.loadUrl(LOCAL_SERVER+"imageViewer.html")
# tabletService.showWebview()
fig.show()
