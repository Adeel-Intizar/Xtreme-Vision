from xtreme_vision.Estimation import Pose_Estimation

model = Pose_Estimation()
model.Use_CenterNet()
model.Detect_From_Image(input_path = 'pose.jpg', 
                        output_path = './out.jpg')

from PIL import Image
Image.open('out.jpg')