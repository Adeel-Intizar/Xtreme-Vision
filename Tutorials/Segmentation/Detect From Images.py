from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
model.Detect_From_Image(input_path='img.jpg',
                        output_path='./out.jpg')

from PIL import Image
Image.open('out.jpg')