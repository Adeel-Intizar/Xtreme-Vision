from xtreme_vision.Detection import Object_Detection

model = Object_Detection()
model.Use_RetinaNet()
model.Detect_From_Video(input_path = 'road.mp4', 
                        output_path='./retinanet.mp4', 
                        extract_objects=False)