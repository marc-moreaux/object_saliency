from naoqi import ALProxy
import numpy as np

ip   = "10.0.160.236"

class VideoStream():
  def __init__(self,ip,port=9559,camId=0):
    """Load the video proxy on robot
       
      Keyword arguments:
      ip -- ip of the robot
      port -- port where naoqi listen
      camId -- top cam is 0, bottom cam is 1
    """
    self.strId = "0"     # No idea
    resolution = 2  # VGA
    colorSpace = 11 # RGB
    
    self.video = ALProxy("ALVideoDevice",ip,port)
    self.status = self.video.subscribeCamera(self.strId,camId,resolution,colorSpace,30)
    if self.status == "":
        raise Exception("subscribtion to camera failed")
    else:
        self.strId = self.status
    
    nUsedResolution = self.video.getResolution( self.strId );
    self.nWidth, self.nHeight = self.video.resolutionToSizes( nUsedResolution );
    
  
  def getFrame(self):
    """"Retrieve a frame form the video proxy"""
    resultCamera = self.video.getImageRemote(self.strId)[6]
    if resultCamera is None:
      raise Exception("Cannot read a frame from the video feed")
    else:
      image = resultCamera;
      img = np.fromstring(image, dtype=np.uint8)
      img = np.reshape(img, ( self.nHeight, self.nWidth,3))
      
    return img
  
  def __del__(self):
    """Uninstanciate the video proxy"""
    self.video.unsubscribe(self.status)



if __name__ == "__main__":
  # test the class
  import matplotlib.pyplot as plt
  vStream = VideoStream("10.0.160.236")
  plt.imshow(vStream.getFrame())
  plt.show()
