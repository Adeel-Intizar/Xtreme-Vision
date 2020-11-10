"""
MIT License

Copyright (c) 2020 Adeel <kingadeel2017@outlook.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from xtreme_vision.Detection.centernet import PoseEstimation as estimate
import numpy as np
import cv2
from PIL import Image


class Pose_Estimation:
    
    """
    This is Pose-Estimation Class in Xtreme-Vision Library, it provides the support of State-Of-The-Art Models 
    like CenterNet. After Instantiating this Class, you can set its properties and use pre-defined
    functions for performing segmentation Tasks out of the box.

    Note: Custom Segmenation only Supports Mask-RCNN

        Use_CenterNet()                             # To Specify which Model to Use
        Detect_From_Image()                         # To Detect from Images
        Detect_From_Video()                         # To Detect from Videos

    """

    def __init__(self):
    
        self.model = None
        self.modelLoaded = False
        self.modelType = None
  
    def Use_CenterNet(self):
        
        """[This function is used to set the Model Type to CenterNet, Automatically downloads the 
        weights file and Loads the Model.]
        """

        self.model = estimate()
        print('-' * 20)
        print('Loading the Model \n Please Wait...')
        print('-' * 20)
        self.model.load_model()
        self.modelLoaded = True
        self.modelType = 'centernet'   

    def Detect_From_Image(self, input_path:str, output_path:str):
        
        """[This Function is used to detect pose from Images.]
        
        Args:
            input_path: (str) [path to the input image with jpg/jpeg/png extension]
            output_path: (str) [path to save the output image with jpg/jpeg/png extension]
 
        Raises:
            RuntimeError: [If Model is not Loaded before using this Function]
            RuntimeError: [If any othre Model type is specified other than CenterNet]
        """
        
        if self.modelLoaded != True:
            raise RuntimeError ('Before calling this function, you have to call Use_CenterNet().')
              
        img = np.array(Image.open(input_path))[..., ::-1]
              
        if self.modelType == 'centernet':
            
            _ = self.model.predict(img = img, output_path = output_path, debug = True) 
                  
        else:
            raise RuntimeError ('Invalid ModelType: Valid Type Is "CenterNet"')

    def Detect_From_Video(self, input_path:str, output_path:str):
        
        """[This Function is used to detect pose from Videos.]
        
        Args:
            input_path: (str) [path to the input Video with mp4/avi extension]
            output_path: (str) [path to save the output Video with mp4/avi extension]
 
        Raises:
            RuntimeError: [If Model is not Loaded before using this Function]
            RuntimeError: [If any othre Model type is specified other than CenterNet]
        """
    
        if self.modelLoaded != True:
            raise RuntimeError ('Before calling this function, you have to call Use_CenterNet().')

        out = None
        cap = cv2.VideoCapture(input_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'\nThere are {length} Frames in this video')
        print('-' * 20)
        print('Detecting Objects in the Video... Please Wait...')
        print('-' * 20)

        while(cap.isOpened()):
            retreive, frame = cap.read()
            if not retreive:
                break

            frame = np.array(frame)[..., ::-1]      
            if self.modelType == 'centernet':
                im = self.model.predict(img = frame, output_path='./', debug = False)
                im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            else:
                raise RuntimeError ('Invalid ModelType: Valid Type Is "CenterNet"')
            
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
            
            out.write(im)
        print('Done. Processing has been Finished... Please Check Output Video.')
        out.release()
        cap.release()
