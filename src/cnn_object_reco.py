import sys
try: from importlib import reload # for python3
except: pass
reload(sys)
#sys.setdefaultencoding('utf8') # for python3  # but doesn't works

from singleton import Singleton

from forward_model import Forward_model
import model_param
import cv2
import sys
strUtilsPath = "../utils"
if strUtilsPath not in sys.path:
    sys.path.append( strUtilsPath )
import my_images


@Singleton
class DetectObject:
  def __init__(self):
    self.mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-4, 5e-5, 5e-7)
    self.model = None
    self.loadDetector()
    self.toDetect =["soda-can",           "people",
                    "fire-extinguisher",  "tennis-ball",
                    "lathe",              "t-shirt",
                    "jesus-christ",       "flashlight",
                    "soccer-ball",        "computer-monitor",
                    "lightbulb",          "bowling-ball",
                    "teapot",             "wine-bottle",
                    "spoon",              "microwave",
                    "tennis-shoes",       "dog",
                    "laptop-101",         "binoculars",
                    "cereal-box",         "human-skeleton",
                    "computer-keyboard",  "head-phones",
                    "eyeglasses",         "computer-mouse",
                    "top-hat",            "beer-mug",
                    "watch-101",          "coffee-mug",
                    "fried-egg",]
  
  def __del__( self ):
    #~ if loaded:
        # unload...
    pass
      
  
  
  def loadDetector( self ):
    """
    Load model on GPU
    """
    if self.model == None:
      self.model = Forward_model(self.mod_param, 44)
    print( "INF: loadDetector: model loaded..." )
  
  
  def detectInBuffer( self, npBuffer, restrict_detection=True, do_resize=True):
    """
    return une liste d'objet avec confiance
    """
    named_preds, vis = self.model.forward_image(npBuffer,-1, do_resize)
    if restrict_detection == True:
      print( vis[0].shape )
      
      tmp = [(p,vis[0][:,:,:,idx]) for idx,p in enumerate(named_preds) if p[0] in detect.toDetect]
      named_preds, vis = [p[0] for p in tmp], [v[1] for v in tmp]
    return named_preds, vis
  
  def getNativeImageProperies( self ):
    """
    return w,h,nbr_planes
    """
    return [224,224,3]
  
  def detectInFilename( self, strFilename, restrict_detection=True ):
    """
    return une liste d'objet avec confiance
    """
    # im = cv2.imread( strFilename )
    im = my_images.load_image( strFilename )
    return self.detectInBuffer( im, restrict_detection=restrict_detection )

# class DetectObject - end


#def main():
pictures = [
"image1474967223.33.jpg",  "image1474967224.13.jpg",  "image1474967224.33.jpg",
"image1474967249.83.jpg",  "image1474967378.31.jpg",  "image1474967390.56.jpg",
"image1474967402.23.jpg",  "image1474967413.34.jpg",  "image1474967422.34.jpg",
"image1474967433.6.jpg" ,  "image1474967445.84.jpg",  "image1474967457.37.jpg",
"image1474967486.98.jpg",]
# pictures = ["/home/mmoreaux/Pictures/"+p for p in pictures]
# pictures = ["/home/cuda/datasets/Perso_photo/"+p for p in pictures]
pictures = ["../test_images/"+p for p in pictures]



detect = DetectObject.Instance()
for path in pictures:
  preds = detect.detectInFilename(path)[0]
  preds = sorted(preds, key=lambda a:a[1], reverse=True)
  print( preds )

# results were:
"""
/home/gpu/work/object_saliency/src/forward_model.py:94: FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
if xx == None :
[('laptop-101', 1.9172316, -0.1428571428571429, 0.14285714285714279), ('flashlight', 1.8609635, 0.28571428571428581, 0.4285714285714286)]
[('soda-can', 1.99108, 0.4285714285714286, 0.0), ('laptop-101', 1.9789326, -0.1428571428571429, 0.14285714285714279)]
[('computer-monitor', 1.9407232, 0.28571428571428581, 0.14285714285714279), ('soda-can', 1.8740108, 0.4285714285714286, 0.0)]
[('people', 2.3289642, 0.28571428571428581, 0.4285714285714286), ('bowling-ball', 1.125162, -0.2857142857142857, -0.2857142857142857)]
[('people', 2.415684, 0.4285714285714286, 0.4285714285714286), ('jesus-christ', 1.3031428, 0.4285714285714286, 0.4285714285714286)]
[('people', 0.85815156, -0.5714285714285714, 0.5714285714285714), ('soda-can', 0.73164177, 0.4285714285714286, 0.0)]
[('head-phones', 2.2543364, 0.0, 0.0), ('soccer-ball', 0.62977815, -0.1428571428571429, -0.1428571428571429)]
[('lathe', 1.1616666, 0.5714285714285714, 0.0), ('fire-extinguisher', 0.56938893, -0.2857142857142857, 0.4285714285714286)]
[('lathe', 4.1666908, 0.14285714285714279, 0.14285714285714279), ('computer-monitor', 1.8748485, -0.5714285714285714, -0.4285714285714286)]
[('soda-can', 1.4000971, -0.2857142857142857, -0.1428571428571429), ('lathe', 0.9071579, -0.7142857142857143, 0.14285714285714279)]
[('lathe', 2.6814904, -0.4285714285714286, 0.14285714285714279), ('laptop-101', 1.091622, 0.0, 0.14285714285714279)]
[('fire-extinguisher', 0.53935093, -0.4285714285714286, -0.85714285714285721), ('lathe', 0.49073964, -0.5714285714285714, 0.14285714285714279)]
[('people', 1.4568579, -0.7142857142857143, -0.5714285714285714), ('soda-can', 1.0642385, -0.5714285714285714, -0.5714285714285714)]    
"""



#~ if __name__ == '__main__':
    #~ main()