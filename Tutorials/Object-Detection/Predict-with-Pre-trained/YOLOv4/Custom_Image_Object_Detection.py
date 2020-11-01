from xtreme_vision.Detection import Object_Detection
from PIL import Image

model = Object_Detection()
model.Use_YOLOv4()
custom_objects = model.Custom_Objects(kite=True, person=False)
model.Detect_Custom_Objects_From_Image(custom_objects, 
                                       input_path = 'kite.jpg',
                                       output_path = './out.jpg')

Image.open('out.jpg')