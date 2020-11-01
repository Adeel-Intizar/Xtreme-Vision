from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_YOLOv4()
model.Detect_From_Image(input_path='kite.jpg',
                        output_path='./out.jpg')

from PIL import Image
Image.open('out.jpg')