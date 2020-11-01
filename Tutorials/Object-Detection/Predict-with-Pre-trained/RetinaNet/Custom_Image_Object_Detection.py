from xtreme_vision.Detection import Object_Detection
from PIL import Image

model = Object_Detection()
model.Use_RetinaNet()
custom_objects = model.Custom_Objects(kite=True, person=False)
model.Detect_Custom_Objects_From_Image(custom_objects, 
                                       input_path = 'kite.jpg',
                                       output_path = './custom_detection.jpg', 
                                       minimum_percentage_probability = 0.2,
                                       extract_objects=False)

Image.open('custom_detection.jpg')