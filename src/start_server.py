# -*- coding: utf-8 -*-

###########################################################
# Start le server qui fait tout ce qu'il faut

# Aldebaran Robotics (c) 2010 All Rights Reserved - This file is confidential.
###########################################################

import abcdk.socket_receive
import cnn_object_reco

def runServer():
    
    def analyseImage( npImage ):
      return cnn_object_reco.DetectObject.Instance().detectInBuffer( npImage )[0]
    
    s = abcdk.socket_receive.SocketReceiver()
    nPortNumber = 10000
    s.run( nPortNumber, analyseImage )
    #time.sleep( 100    
# runServer - end


runServer()