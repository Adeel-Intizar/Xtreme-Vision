from xtreme_vision.Segmentation import Segmentation

model = Segmentation()
model.Use_MaskRCNN()
model.Detect_From_Video(input_path='road.mp4',
                        output_path='./out.mp4')