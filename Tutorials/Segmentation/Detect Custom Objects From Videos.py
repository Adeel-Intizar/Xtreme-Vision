from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
objects = model.Custom_Objects(car = True)
model.Detect_Custom_Objects_From_Video(custom_objects = objects,
                                       input_path='road.mp4',
                                       output_path='./out.mp4',
                                       show_names = False)