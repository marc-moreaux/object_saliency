from singleton import Singleton

from forward_model import Forward_model
import model_param
import cv2
import my_images


@Singleton
class DetectObject:
  def __init__(self):
    self.mod_param  = model_param.Model_params("CALTECH256", "VGG16_CAM5b_S", 'rmsProp',   1e-4, 5e-5, 5e-7)
    self.model = None
    self.loadDetector()
    self.toDetect =["soda-can",           "people",
                    "fire-extinguisher",  "light-house",
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
                    "tennis-ball",]
  
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
  
  
  def detectInBuffer( self, npBuffer , do_resize=True):
    """
    return une liste d'objet avec confiance
    """
    named_preds, vis = self.model.forward_image(npBuffer,-1, do_resize)
    return named_preds, vis
  
  def getNativeImageProperies( self ):
    """
    return w,h,nbr_planes
    """
    return [224,224,3]
  
  def detectInFilename( self, strFilename ):
    """
    return une liste d'objet avec confiance
    """
    # im = cv2.imread( strFilename )
    im = my_images.load_image( strFilename )
    return self.detectInBuffer( im )

# class DetectObject - end


def main():
    pictures = [
    "image1474967223.33.jpg",  "image1474967224.13.jpg",  "image1474967224.33.jpg",
    "image1474967249.83.jpg",  "image1474967378.31.jpg",  "image1474967390.56.jpg",
    "image1474967402.23.jpg",  "image1474967413.34.jpg",  "image1474967422.34.jpg",
    "image1474967433.6.jpg" ,  "image1474967445.84.jpg",  "image1474967457.37.jpg",
    "image1474967486.98.jpg",]
    pictures = ["/home/mmoreaux/Pictures/"+p for p in pictures]
    # pictures = ["/home/cuda/datasets/Perso_photo/"+p for p in pictures]


    detect = DetectObject.Instance()
    for path in pictures:
      preds = detect.detectInFilename(path)[0]
      preds = [p for p in preds if p[0] in detect.toDetect]
      preds = sorted(preds, key=lambda a:a[1], reverse=True)
      print preds[:2]




if __name__ == '__main__':    
    main()