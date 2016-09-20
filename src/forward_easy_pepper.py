# from util import load_image
from skimage.transform import resize
from stream import VideoStream
import my_images

import model_param
from forward_model import Forward_model


######################################
###  Parameter
######################################
mod_param  = model_param.Model_params('PERSO', "VGG16_CAM_W_S", 'rmsProp', 0.00001)
PEPPER_IP  = "10.0.160.236" # jarc.local


######################################
### Load video stream of pepper
######################################
vStream = VideoStream( PEPPER_IP )
img = vStream.getFrame()


######################################
# Load tensorflow's caltech model
######################################
model = Forward_model(mod_param, 24)


######################################
### guessing time !
######################################

while True:
    img = vStream.getFrame()
    img = my_images.crop_from_center(img)
    img = resize(img, [224,224])
    named_preds, vis = model.forward_image(img,-1)
    preds = sorted(named_preds, key=lambda a:a[1], reverse=True)[:5]
    for name, prob in preds:
        print "%-10s %.3f"%(name,prob)
    print '--'
    

