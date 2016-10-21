from singleton import Singleton

from forward_model import Forward_model
import model_param
import cv2


@Singleton
class DetectObject:
  def __init__(self):
    self.mod_param  = model_param.Model_params('PERSO', "VGG16_CAM_W_S", 'rmsProp', 0.00001)
    self.model = None
    self.loadDetector()
  
  
  def __del__( self ):
    #~ if loaded:
        # unload...
    pass
      
  
  
  def loadDetector( self ):
    """
    Load model on GPU
    """
    if self.model == None:
      self.model = Forward_model(self.mod_param, 24)
  
  
  def detectInBuffer( self, npBuffer ):
    """
    return une liste d'objet avec confiance
    """
    named_preds, vis = self.model.forward_image(npBuffer,-1)
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
    im = cv2.imread( strFilename )
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

  detect = DetectObject.Instance()
  for path in pictures:
    print detect.detectInFilename(path)[0][:5]




if __name__ == '__main__':    
    main()