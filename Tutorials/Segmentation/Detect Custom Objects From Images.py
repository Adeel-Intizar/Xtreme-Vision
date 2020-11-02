from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
objects = model.Custom_Objects(car = True, person = False)
model.Detect_Custom_Objects_From_Image(custom_objects = objects,
                                       input_path='image.jpg',
                                       output_path='./out.jpg',
                                       show_names = False)

from PIL import Image
Image.open('out.jpg')